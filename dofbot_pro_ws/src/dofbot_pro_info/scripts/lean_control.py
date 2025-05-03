#!/usr/bin/env python3

import numpy as np

import rospy
from dofbot_pro_info.msg import ArmJoint

class Controller:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('controller')

        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        self.gripper_angle = 30.0

    def move_arm_to_position(self, joint_positions, run_time=2000):
        """Send joint positions to the robot arm"""
        joint_positions[5] = self.gripper_angle
        arm_joint = ArmJoint()
        arm_joint.joints = joint_positions
        arm_joint.run_time = run_time
        self.pub_arm.publish(arm_joint)

    def gripper_control(self, aperture, run_time=1000):
        """Control the gripper (servo 6) based on aperture"""
        # Map aperture from your policy's range to the robot's range (assumed 30-180)
        # Adjust this mapping based on your specific aperture range
        gripper_angle = np.interp(aperture, [0, 1], [30, 140])
        self.gripper_angle = gripper_angle
        
        arm_joint = ArmJoint()
        arm_joint.id = 6  # Gripper servo ID
        arm_joint.angle = gripper_angle
        arm_joint.run_time = run_time
        arm_joint.joints = []
        self.pub_arm.publish(arm_joint)

    def run(self):
        joint_positions = [70.0, 36.0, 60.0, 20.0, 90.0, 30.0] #[90.0, 36.0, 60.0, 20.0, 90.0, 30.0]
        self.move_arm_to_position(joint_positions)
        rospy.sleep(3)
        
        # 4. Close gripper
        self.gripper_control(1)  # Fully closed
        rospy.sleep(2)

        home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 40]
        self.move_arm_to_position(home_position)
        rospy.sleep(3)

        # 7. Open gripper to release object
        self.gripper_control(0)  # Fully open
        rospy.sleep(3)

        rospy.is_shutdown()

if __name__ == '__main__':
    controller = Controller()
    controller.run()