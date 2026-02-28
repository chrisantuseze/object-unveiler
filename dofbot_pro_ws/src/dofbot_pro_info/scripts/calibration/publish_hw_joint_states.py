#!/usr/bin/env python3
"""
Read joint angles from the hardware via `Arm_Lib.Arm_Device` and publish to
`/joint_states` so other nodes (recorders) can consume real joint values.

Usage:
  rosrun dofbot_pro_info publish_hw_joint_states.py --rate 2

This script queries the servos for their current positions and publishes a
`sensor_msgs/JointState` message at the requested rate. It expects the same
servo numbering/conventions used by `arm_driver.py`.
"""
import time
import rospy
from sensor_msgs.msg import JointState
from Arm_Lib import Arm_Device
import numpy as np
from math import pi
import argparse


def read_hw_positions(arm):
    # Read servos 1..6 (Arm_Device returns degrees)
    pos = []
    for i in range(6):
        try:
            v = arm.Arm_serial_servo_read(i + 1)
            pos.append(float(v))
            time.sleep(0.01)
        except Exception:
            pos.append(90.0)
    return pos


def to_jointstate_positions(hw_positions):
    # arm_driver maps hardware degrees to a centered range around 90 and
    # converts to radians before publishing. Reproduce that mapping.
    mid = np.array([90, 90, 90, 90, 90, 90], dtype=float)
    arr = np.array(hw_positions, dtype=float) - mid
    # Convert degrees -> radians
    return list(np.deg2rad(arr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=float, default=1.0, help='Publish rate Hz')
    args = parser.parse_args()

    rospy.init_node('publish_hw_joint_states', anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
    arm = Arm_Device()
    rate = rospy.Rate(args.rate)

    rospy.loginfo(f"Publishing hardware joint_states at {args.rate} Hz")
    while not rospy.is_shutdown():
        hw_pos = read_hw_positions(arm)
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.header.frame_id = 'joint_states'
        js.name = ["Arm1_Joint", "Arm2_Joint", "Arm3_Joint", "Arm4_Joint", "Arm5_Joint", "grip_joint"]
        js.position = to_jointstate_positions(hw_pos)
        pub.publish(js)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
