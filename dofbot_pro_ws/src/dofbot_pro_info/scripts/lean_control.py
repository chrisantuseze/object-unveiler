#!/usr/bin/env python3

import numpy as np

from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import compute_pre_grasp_joints, compute_post_grasp_joints
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
        gripper_angle = np.interp(aperture, [0, 0.5, 1], [30, 160, 140])
        self.gripper_angle = gripper_angle
        
        arm_joint = ArmJoint()
        arm_joint.id = 6  # Gripper servo ID
        arm_joint.angle = gripper_angle
        arm_joint.run_time = run_time
        arm_joint.joints = []
        self.pub_arm.publish(arm_joint)

    def compute_post_grasp_joints(self, grasp_joints):
        """Compute a post-grasp position"""
        post_grasp = grasp_joints.copy()
        post_grasp[1] = 60  # Adjust second joint to lift
        # post_grasp[2] = 5  # Adjust third joint to lift
        # post_grasp[3] = 270  # Adjust fourth joint to lift
        return post_grasp
    
    def run0(self):
        print("Starting controller...")
        # joint_positions = [90.0, 0.0, 15.0, 180.0, 90.0, 30.0] # peripheral object
        # joint_positions = [85.0, 0.0, 50.0, 130.0, 90.0, 30.0] # central/target object

        # self.step(joint_positions, 1)

        episode_actions = [
            {
                "direction": "left",
                "joint_positions": [90.0, 0.0, 15.0, 180.0, 90.0, 30.0],
                "aperture": 0.5,
            },
            {
                "direction": "right",
                "joint_positions": [85.0, 0.0, 50.0, 130.0, 90.0, 30.0],
                "aperture": 1,
            },
        ]
        for item in episode_actions:
            direction = item["direction"]
            joint_positions = item["joint_positions"]

            print(f"Moving to {direction} position: {joint_positions}")
            self.step(joint_positions, item["aperture"])
            rospy.sleep(3)
        
        rospy.is_shutdown()

    def run1(self):
        print("Starting controller...")
        joint_positions = [72.0, 0.0, 20.0, 160.0, 90.0, 30.0]

        self.step(joint_positions)

        rospy.is_shutdown()

    def run2(self):
        print("Starting controller...")
        episode_actions = [
            {
                "direction": "right",
                "joint_positions": [90.0, 0.0, 5.0, 180.0, 90.0, 30.0],
                "aperture": 1,
            },
            {
                "direction": "left", # closest to user
                "joint_positions": [75.0, 0.0, 10.0, 170.0, 90.0, 30.0],
                "aperture": 1,
            },
            {
                "direction": "central",
                "joint_positions": [85.0, 0.0, 50.0, 130.0, 90.0, 30.0],
                "aperture": 0.5,
            },
        ]
        for item in episode_actions:
            direction = item["direction"]
            joint_positions = item["joint_positions"]

            print(f"Moving to {direction} position: {joint_positions}")
            self.step(joint_positions, item["aperture"])
            rospy.sleep(3)
         
        rospy.is_shutdown()

    def step(self, joint_positions, aperture=1):
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
        self.gripper_control(aperture)  # Fully closed
        rospy.sleep(2)
        
        # 5. Lift object
        post_grasp_joints = self.compute_post_grasp_joints(joint_positions)
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
    controller.run0()