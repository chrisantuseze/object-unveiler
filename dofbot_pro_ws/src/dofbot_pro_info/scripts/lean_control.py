#!/usr/bin/env python3

import numpy as np

from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import compute_post_grasp_joints, compute_pre_grasp_joints
import rospy
from dofbot_pro_info.msg import ArmJoint

class Controller:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('controller')

        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        # Robot arm parameters
        self.home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 40] #30.0]  # Default home position
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

    def run1(self):
        print("Starting controller...")
        # joint_positions = [90.0, 0.0, 35.0, 150.0, 90.0, 30.0] # peripheral object
        # joint_positions = [90.0, 0.0, 70.0, 100.0, 90.0, 30.0] # central/target object

        episode_actions = [[90.0, 0.0, 35.0, 150.0, 90.0, 30.0], [90.0, 0.0, 70.0, 100.0, 90.0, 30.0]]
        for joint_positions in episode_actions:
            self.step(joint_positions)
            rospy.sleep(5)
        
        rospy.is_shutdown()

    def run2(self):
        print("Starting controller...")
        joint_positions = [80.0, 0.0, 35.0, 150.0, 90.0, 30.0] # left peripheral object
        # joint_positions = [100.0, 0.0, 50.0, 120.0, 90.0, 30.0] # right peripheral object
        # joint_positions = [90.0, 0.0, 70.0, 100.0, 90.0, 30.0] # central/target object

        # episode_actions = [[90.0, 0.0, 35.0, 150.0, 90.0, 30.0], [90.0, 0.0, 70.0, 100.0, 90.0, 30.0]]
        # for joint_positions in episode_actions:
        #     self.step(joint_positions)
        #     rospy.sleep(5)

        self.step(joint_positions)
        
        rospy.is_shutdown()

    def step(self, joint_positions):
        """
        Execute a complete grasp sequence
        
        Args:
            joint_angles: Target joint angles for grasp position
        """
        
        # 1. Move to pre-grasp position
        pre_grasp_joints = compute_pre_grasp_joints(joint_positions)
        self.move_arm_to_position(pre_grasp_joints)
        rospy.sleep(3)  # Wait for movement to complete
        
        # 3. Move to grasp position
        self.move_arm_to_position(joint_positions)
        rospy.sleep(3)
        
        # 4. Close gripper
        self.gripper_control(1)  # Fully closed
        rospy.sleep(2)
        
        # 5. Lift object
        post_grasp_joints = compute_post_grasp_joints(joint_positions)
        self.move_arm_to_position(post_grasp_joints)
        rospy.sleep(3)

        # 6. Move to pre-home position
        pre_home = [180, 90, 45, 30.0, 90.0, 30.0]
        self.move_arm_to_position(pre_home)
        rospy.sleep(3)

        # 7. Open gripper to release object
        self.gripper_control(0)  # Fully open
        rospy.sleep(3)
        
        # 8. Return to home position
        self.move_arm_to_position(self.home_position)
        rospy.sleep(3)

if __name__ == '__main__':
    controller = Controller()
    controller.run2()