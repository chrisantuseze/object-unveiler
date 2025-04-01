#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from dofbot_pro_info.msg import ArmJoint
import time
from dofbot_pro_info.msg import *
from dofbot_pro_info.srv import *
# import your_policy_module  # Import your policy module

class PolicyRobotController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('policy_robot_controller')
        
        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        self.ik_client = rospy.ServiceProxy("get_kinemarics", kinemarics)
        
        # Initialize your policy
        # self.policy = your_policy_module.Policy()  # Replace with your actual policy initialization
        
        # Robot arm parameters
        self.home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 30.0]  # Default home position
        # self.home_position = [90.0, 70.0, 0.0, 0.0, 90.0, 30.0]  # Default home position
        self.gripper_angle = 30.0
        
        # Wait for publisher to connect
        rospy.sleep(1)
        
        # Move to home position at startup
        self.move_arm_to_position(self.home_position)
        
        print("Policy Robot Controller initialized")
    
    def grasp_object(self, state, target_mask):
        """
        Execute a grasp based on policy prediction
        
        Args:
            state: The current state for policy input
            target_mask: Target mask for policy input
        """
        try:
            # Get observation (replace with your actual observation method)
            # obs = self.get_observation()
            
            # Get action from policy
            # action = self.policy.exploit_unveiler(state, obs['color'][1], target_mask)
            
            # # Convert to 3D action
            # env_action3d = self.policy.action3d(action)
            
            # # Convert the 3D action to joint angles for dofbot
            # joint_angles = self.convert_action_to_joints(env_action3d)

            pos = [0.08, 0.07, 0.08]
            # pos = [-0.26, -0.10, 0.08]
            env_action3d = {'pos': np.array(pos), 'quat': None, 'aperture': 0.9601622467210791, 'push_distance': 0.12}

            # Convert to joint angles
            joint_angles = self.get_joint_angles_from_pose(
                env_action3d['pos'], 
                env_action3d['quat']
            )
            
            if joint_angles is None:
                rospy.logerr("Failed to compute joint angles, aborting grasp")
                return
            
            # joint_angles = [110.0, 36.0, 60.0, 20.0, 90.0, 30.0]
            # joint_angles = [90.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Left obstacle
            # joint_angles = [70.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Target
            # joint_angles = [60.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Right obstacle

            # Execute the grasp sequence
            self.step(joint_angles, env_action3d['aperture'])
                        
        except Exception as e:
            rospy.logerr(f"Error executing grasp: {str(e)}")
    
    def compute_pre_grasp_joints(self, grasp_joints):
        """Compute a pre-grasp position slightly above the grasp position"""
        pre_grasp = grasp_joints.copy()
        pre_grasp[2] += 20  # Adjust second joint to raise arm
        pre_grasp[3] += 10  # Adjust second joint to raise arm
        return pre_grasp
    
    def compute_post_grasp_joints(self, grasp_joints):
        """Compute a post-grasp position"""
        post_grasp = grasp_joints.copy()
        post_grasp[1] += 30  # Adjust second joint to lift
        post_grasp[2] -= 20  # Adjust second joint to lift
        return post_grasp

    def convert_sim_to_robot_pose(self, sim_pos, sim_quat):
        """Convert simulation position/orientation to robot coordinates"""
        # This is a placeholder - implement based on your coordinate systems
        # You may need to scale, offset, and/or rotate coordinates
        
        # Example conversion (adjust based on your setup):
        robot_x = 102.90 #sim_pos[0] * 100  # Convert to cm
        robot_y = 29.40 #sim_pos[1] * 100
        robot_z = 80 #sim_pos[2] * 100
        
        return robot_x, robot_y, robot_z
    
    def get_joint_angles_from_pose(self, pos, quat):
        """Use inverse kinematics to get joint angles for a pose"""
        x, y, z = self.convert_sim_to_robot_pose(pos, quat)
        
        request = kinemaricsRequest()
        request.tar_x = x
        request.tar_y = y
        request.tar_z = z
        request.kin_name = "ik"
        
        try:
            response = self.ik_client.call(request)
            
            # Check if response is valid (joint angles within limits)
            if response.joint1 < 0 or response.joint1 > 180 or \
               response.joint2 < 0 or response.joint2 > 180 or \
               response.joint3 < 0 or response.joint3 > 180 or \
               response.joint4 < 0 or response.joint4 > 180:
                rospy.logwarn("IK solution contains invalid joint angles")
                return None
            
            joint_angles = [
                response.joint1,
                response.joint2,
                response.joint3,
                response.joint4,
                90,  # Usually fixed at 90
                30   # Initial gripper position
            ]
            
            return joint_angles
            
        except rospy.ServiceException as e:
            rospy.logerr(f"IK service call failed: {e}")
            return None
    
    def step(self, joint_positions, aperture):
        """
        Execute a complete grasp sequence
        
        Args:
            joint_angles: Target joint angles for grasp position
            aperture: Gripper aperture (0-1 range)
        """
        
        # 1. Move to pre-grasp position
        pre_grasp_joints = self.compute_pre_grasp_joints(joint_positions)
        self.move_arm_to_position(pre_grasp_joints)
        rospy.sleep(3)  # Wait for movement to complete
        
        # 3. Move to grasp position
        self.move_arm_to_position(joint_positions)
        rospy.sleep(3)
        
        # 4. Close gripper
        self.gripper_control(1)  # Fully closed
        rospy.sleep(2)
        
        # 5. Lift object
        post_grasp_joints = self.compute_post_grasp_joints(joint_positions)
        self.move_arm_to_position(post_grasp_joints)
        rospy.sleep(3)
        
        # 6. Return to home position
        self.move_arm_to_position(self.home_position)
        rospy.sleep(3)
        
        # 7. Open gripper to release object
        self.gripper_control(0)  # Fully open

    def move_arm_to_position(self, joint_positions, run_time=2000):
        """Send joint positions to the robot arm"""
        joint_positions[5] = self.gripper_angle
        arm_joint = ArmJoint()
        arm_joint.joints = joint_positions
        arm_joint.run_time = run_time
        self.pub_arm.publish(arm_joint)

        print("joint_positions:", joint_positions)
    
    def gripper_control(self, aperture, run_time=1000):
        """Control the gripper (servo 6) based on aperture"""
        # Map aperture from your policy's range to the robot's range (assumed 30-180)
        # Adjust this mapping based on your specific aperture range
        # gripper_angle = np.interp(aperture, [0, 1], [30, 180])
        gripper_angle = np.interp(aperture, [0, 1], [30, 140])
        self.gripper_angle = gripper_angle
        
        arm_joint = ArmJoint()
        arm_joint.id = 6  # Gripper servo ID
        arm_joint.angle = gripper_angle
        arm_joint.run_time = run_time
        arm_joint.joints = []
        self.pub_arm.publish(arm_joint)
    
    def get_observation(self):
        """
        Get observation for policy input
        
        Returns:
            Observation dictionary
        """
        # Replace with your method to get camera/sensor data
        obs = {
            'color': [None, np.zeros((224, 224, 3))]  # Placeholder
        }
        return obs
    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(1)  # 1 Hz, adjust as needed
        
        # while not rospy.is_shutdown(): # This should be while target is not grasped
        #     try:
        #         # Get state and target info
        #         state = None  # Initialize your state
        #         target_mask = None  # Initialize your target mask
                
        #         # Execute grasp based on policy
        #         self.grasp_object(state, target_mask)
                
        #         rate.sleep()
                
        #     except KeyboardInterrupt:
        #         print("Shutting down")
        #         break
        #     except Exception as e:
        #         rospy.logerr(f"Error in main loop: {str(e)}")

        try:
            # Get state and target info
            state = None  # Initialize your state
            target_mask = None  # Initialize your target mask
            
            # Execute grasp based on policy
            self.grasp_object(state, target_mask)
            
            rate.sleep()
            
        except KeyboardInterrupt:
            print("Shutting down")
        except Exception as e:
            rospy.logerr(f"Error in main loop: {str(e)}")

        rospy.is_shutdown()

if __name__ == '__main__':
    try:
        controller = PolicyRobotController()
        controller.run()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"Error in calling PolicyRobotController: {str(e)}")