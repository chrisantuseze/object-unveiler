#!/usr/bin/env python3
"""
Record workspace corner FK positions and update yaml/bhand.yml real_workspace.bounds.

Usage:
  rosrun dofbot_pro_info record_workspace_bounds.py --n 4

Do NOT run arm_driver.py alongside this script — it engages servo torque and
locks the joints. This script reads servo positions directly via Arm_Lib
(no writes, no torque engagement) so the arm stays free to move by hand.

Procedure:
  1. Move the gripper to a corner of the reachable workspace.
  2. Press ENTER — the script reads current joint angles from the hardware
     and calls get_kinemarics (FK) to get (x,y,z) in the robot base frame.
  3. Repeat for all corners.
  4. The script computes xmin/xmax, ymin/ymax and writes
     env.real_workspace.bounds into yaml/bhand.yml.
"""
import os
import math
import time
import yaml
import argparse
import rospy
from sensor_msgs.msg import JointState
from dofbot_pro_info.srv import kinemarics, kinemaricsRequest


# ---------------------------------------------------------------------------
# Hardware joint reading (no servo writes — joints stay free)
# ---------------------------------------------------------------------------

def release_servo_torque():
    """Release servo torque so joints can be moved by hand."""
    try:
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        arm.Arm_serial_set_torque(0)
        rospy.loginfo("Servo torque RELEASED — joints are free to move")
        return arm
    except Exception as e:
        rospy.logwarn(f"Could not release servo torque: {e}")
        return None


def engage_servo_torque(arm=None):
    """Re-engage servo torque (call after recording is done)."""
    try:
        if arm is None:
            from Arm_Lib import Arm_Device
            arm = Arm_Device()
        arm.Arm_serial_set_torque(1)
        rospy.loginfo("Servo torque ENGAGED")
    except Exception as e:
        rospy.logwarn(f"Could not engage servo torque: {e}")


def read_hw_joints_deg():
    """Read all 6 servo positions (degrees) directly from hardware via Arm_Lib."""
    try:
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        joints = []
        for i in range(6):
            time.sleep(0.01)
            v = arm.Arm_serial_servo_read(i + 1)
            joints.append(float(v) if v is not None else 90.0)
            time.sleep(0.01)
        return joints
    except Exception as e:
        rospy.logwarn(f"Arm_Lib read failed: {e}")
        return None


def read_joints_from_topic_deg(timeout=5.0):
    """Fallback: read /joint_states (radians) and convert to degrees."""
    try:
        msg = rospy.wait_for_message('/joint_states', JointState, timeout=timeout)
        return [math.degrees(j) for j in msg.position]
    except rospy.ROSException:
        return None


def get_joint_angles_deg():
    """Try hardware first; fall back to /joint_states topic."""
    joints = read_hw_joints_deg()
    if joints is not None:
        return joints, 'hardware'
    joints = read_joints_from_topic_deg()
    if joints is not None:
        return joints, '/joint_states'
    return None, None


def call_fk_service(joints_deg, service_name='get_kinemarics'):
    try:
        rospy.wait_for_service(service_name, timeout=3.0)
        client = rospy.ServiceProxy(service_name, kinemarics)
        req = kinemaricsRequest()
        req.kin_name = 'fk'
        # FK service expects degrees; joints_deg are already in degrees.
        defaults = [90.0, 90.0, 90.0, 0.0, 90.0, 30.0]
        for i in range(6):
            val = joints_deg[i] if i < len(joints_deg) else defaults[i]
            setattr(req, f'cur_joint{i+1}', float(val))
        resp = client.call(req)
        return resp.x, resp.y, resp.z
    except Exception as e:
        rospy.logwarn(f"FK service call failed: {e}")
        return None


def update_bhand_yaml(yaml_path, real_bounds):
    # Load existing YAML and update only the env.real_workspace.bounds key
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if 'env' not in data or data['env'] is None:
        data['env'] = {}

    # Preserve existing env.workspace and other keys; only set real_workspace.bounds
    real_ws = data['env'].get('real_workspace', {}) or {}
    # Ensure bounds are simple nested Python lists of floats
    rb = [[float(real_bounds[0][0]), float(real_bounds[0][1])],
          [float(real_bounds[1][0]), float(real_bounds[1][1])],
          [float(real_bounds[2][0]), float(real_bounds[2][1])]]
    real_ws['bounds'] = rb
    data['env']['real_workspace'] = real_ws

    # Write back with stable formatting but do not sort keys to keep file readable
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4, help='Number of corner samples')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'yaml', 'bhand.yml'), help='Path to bhand.yml to update')
    parser.add_argument('--fk-service', type=str, default='get_kinemarics', help='FK service name')
    args = parser.parse_args()

    rospy.init_node('record_workspace_bounds', anonymous=True)
    rospy.loginfo(f"Recording {args.n} workspace corners")

    arm = release_servo_torque()   # release torque so joints can be moved by hand
    print("Joints are free. Move the gripper by hand to each corner.")
    print("Torque will be re-engaged when recording is complete.\n")

    corners = []
    for i in range(args.n):
        input(f"\nMove gripper to corner {i+1}/{args.n} and press ENTER...")

        joints_deg, source = get_joint_angles_deg()
        if joints_deg is None:
            print("Could not read joints from hardware or /joint_states.")
            print("Enter joint values manually as comma-separated degrees (j1..j6):")
            manual = input('  j1,j2,j3,j4,j5,j6: ')
            joints_deg = [float(x.strip()) for x in manual.split(',')]
            source = 'manual'
        else:
            print(f"Joint angles [deg] from {source}: {[round(j,2) for j in joints_deg]}")

        fk = call_fk_service(joints_deg, service_name=args.fk_service)
        if fk is None:
            print("FK service unavailable — please enter the measured FK XYZ for this corner (metres):")
            manual_xyz = input('x,y,z: ')
            x, y, z = [float(x.strip()) for x in manual_xyz.split(',')]
        else:
            x, y, z = fk
            print(f"FK -> x={x:.4f}, y={y:.4f}, z={z:.4f}")

        corners.append((x, y, z))

    x_vals = [c[0] for c in corners]
    y_vals = [c[1] for c in corners]

    real_bounds = [ [min(x_vals), max(x_vals)], [min(y_vals), max(y_vals)], [0.004, 0.30] ]

    print(f"Computed real_bounds: {real_bounds}")
    out_path = os.path.abspath(args.out)
    print(f"Updating {out_path} ...")
    update_bhand_yaml(out_path, real_bounds)
    engage_servo_torque(arm)
    print("Done.")


if __name__ == '__main__':
    main()
