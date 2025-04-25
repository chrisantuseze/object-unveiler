import numpy as np
import pybullet as p
from your_robot_msgs.srv import Kinematics, KinematicsRequest

# 1) Joint limits
sim_joint_names = ['joint_x', 'joint_y', 'joint_z', 'joint_revolute']
sim_joint_limits = {
    'joint_x'        : (-1.0,  1.0),    # meters (use a realistic bound, not ±1000!)
    'joint_y'        : (-1.0,  1.0),
    'joint_z'        : ( 0.0,  1.0),
    'joint_revolute' : (np.radians(-100), np.radians(100)),  # radians
}

# DS-SY15A servos: ±150° around neutral
real_joint_names = ['axis_x','axis_y','axis_z','wrist','palm','gripper']
real_joint_limits = {
    'axis_x' : (-150.0, 150.0),  # degrees
    'axis_y' : (-150.0, 150.0),
    'axis_z' : (-150.0, 150.0),
    'wrist'  : (-150.0, 150.0),
    'palm'   : (  90.0,  90.0),  # locked at 90°
    'gripper': (  30.0, 140.0),  # your working aperture
}

# 2) Placeholder mapping from slide distances (m) → servo degrees
LEADSCREW_PITCH_M = 0.005  # e.g. 5 mm per 360°
def slide_m_to_servo_deg(x_m):
    return np.clip(x_m * (360.0 / LEADSCREW_PITCH_M),
                   real_joint_limits['axis_x'][0],
                   real_joint_limits['axis_x'][1])

# 3) Initialize connection to your IK/FK service
fk_client = rospy.ServiceProxy('/kinematics', Kinematics)
arm_id    = 0        # PyBullet robot id
ee_link   = 9        # your end-effector link idx

sim_pts_cm = []
rob_pts_cm = []

for _ in range(10):
    # a) Sample sim joint values
    sim_joints = []
    for name in sim_joint_names:
        lo, hi = sim_joint_limits[name]
        sim_joints.append(np.random.uniform(lo, hi))

    # b) Apply in PyBullet
    for idx, angle in enumerate(sim_joints):
        p.resetJointState(arm_id, idx, angle)
    for _ in range(50): p.stepSimulation()

    # c) Read sim end-effector pose (in meters), convert to cm
    ls = p.getLinkState(arm_id, ee_link, computeForwardKinematics=True)
    sim_xyz = np.array(ls[4])       # (x,y,z) in meters
    sim_xyz_cm = sim_xyz * 100.0

    # d) Build real-robot joint command:
    servo_x    = slide_m_to_servo_deg(sim_joints[0])
    servo_y    = slide_m_to_servo_deg(sim_joints[1])
    servo_z    = slide_m_to_servo_deg(sim_joints[2])
    servo_w    = np.clip(np.degrees(sim_joints[3]),
                         *real_joint_limits['wrist'])
    servo_palm = 90.0
    servo_grip = 30.0

    hw_joints = [servo_x, servo_y, servo_z, servo_w, servo_palm, servo_grip]

    # e) Call your FK service
    req = KinematicsRequest()
    req.joints = hw_joints
    res = fk_client(req)
    robot_xyz_cm = np.array([res.x, res.y, res.z])

    # f) Store the pair
    sim_pts_cm.append(sim_xyz_cm)
    rob_pts_cm.append(robot_xyz_cm)

# After this loop, sim_pts_cm and rob_pts_cm are N×3 arrays you can feed
# into your Kabsch/SVD routine to solve for R and t.
