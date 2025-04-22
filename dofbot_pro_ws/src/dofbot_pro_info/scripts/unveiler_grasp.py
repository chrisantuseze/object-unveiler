#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import yaml
from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import compute_post_grasp_joints, compute_pre_grasp_joints, convert_numpy_masks_to_ros_image_list
from policy import grasping
import rospy
import cv2
import time
import copy
import numpy as np
import argparse

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from dofbot_pro_info.msg import ArmJoint, ObservData, ActionData
from dofbot_pro_info.msg import *
from dofbot_pro_info.srv import *
from dofbot_pro_ws.src.dofbot_pro_info.scripts.image_manip import HeightmapGenerator
from utils import general_utils
import utils.logger as logging

from std_msgs.msg import String

class PolicyRobotController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('policy_robot_controller')
        self.TEST_DIR = "dofbot_pro_ws/src/dofbot_pro_info/scripts/images"
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)
        
        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        self.ik_client = rospy.ServiceProxy("get_kinemarics", kinemarics)

        # Image Storage
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.point_cloud = None
        self.state = None
        self.intrinsics = None  # Camera intrinsics
        
        # Image acquisition locks and flags
        self.rgb_lock = False
        self.depth_lock = False
        self.camera_info_received = False
        
        # Subscribers - initialized but not active yet
        self.rgb_sub = None
        self.depth_sub = None
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)

        # self.action_sub = rospy.Subscriber('/action/data', ActionData, self.action_sub_callback)
        # self.observation_pub = rospy.Publisher("/action/obs", ObservData, queue_size=1)

        self.processed_masks, self.pred_mask, self.raw_masks, self.bboxes = [], None, [], []
        self.raw_color_image, self.raw_depth_image, self.target_mask = None, None, None
        self.action = None

        
        # Robot arm parameters
        self.home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 40] #30.0]  # Default home position
        self.gripper_angle = 30.0
        
        # Wait for publisher to connect and camera info to be received
        rospy.sleep(1)
        
        # Move to home position at startup
        self.move_arm_to_position(self.home_position)
        print("Policy Robot Controller initialized")
        
        # Wait for camera info to be received
        start_time = time.time()
        while not self.camera_info_received and time.time() - start_time < 10:
            rospy.sleep(0.1)
        
        if not self.camera_info_received:
            rospy.logwarn("Camera info not received within timeout. Some features may not work properly.")

        self.pub = rospy.Publisher('/robot_machine', String, queue_size=10)
        self.sub = None
        while self.pub.get_num_connections() == 0:
            rospy.loginfo("Waiting for local machine to subscribe...")
            rospy.sleep(0.5)

    def action_sub_callback(self, action_data):
        self.action = action_data.values
        self.target_mask = action_data.target_mask
        print("Received action data", self.action)

    def camera_info_callback(self, msg):
        """ Extract camera intrinsic parameters. """
        self.intrinsics = np.array(msg.K).reshape(3, 3)  # Intrinsic matrix (3x3)
        self.camera_info_received = True
        # We can keep this subscription active all the time as the camera parameters don't change

    def rgb_callback(self, msg):
        """ Callback to receive the RGB image. """
        if self.rgb_lock:
            try:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to OpenCV format
                self.rgb_image = cv2.flip(self.rgb_image, -1)

                # self.image_pub.publish(msg)
                self.raw_color_image = msg

                cv2.imwrite(os.path.join(self.TEST_DIR, "saved_rgb_image.png"), self.rgb_image)
                self.rgb_lock = False  # Release the lock
            except Exception as e:
                rospy.logerr(f"RGB conversion error: {e}")
                self.rgb_lock = False  # Make sure to release the lock even if there's an error

    def depth_callback(self, msg):
        """ Callback to receive the depth image. """
        if self.depth_lock:
            try:
                # Convert ROS depth image to OpenCV format
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Depth is in 16-bit unsigned int
                self.depth_image = cv2.flip(self.depth_image, -1)

                self.raw_depth_image = msg

                # Normalize depth to 0–255 and convert to 8-bit for visualization
                depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)
                cv2.imwrite(os.path.join(self.TEST_DIR, "saved_depth_image.png"), depth_vis)
                
                self.depth_lock = False  # Release the lock
            except Exception as e:
                rospy.logerr(f"Depth conversion error: {e}")
                self.depth_lock = False  # Make sure to release the lock even if there's an error

    def get_latest_image(self, timeout=5.0):
        """
        Get the latest RGB and depth images on demand
        
        Args:
            timeout: Maximum time to wait for images (seconds)
            
        Returns:
            True if both images were successfully acquired, False otherwise
        """
        # Reset image data
        self.rgb_image = None
        self.depth_image = None

        self.raw_color_image = None
        self.raw_depth_image = None
        
        # Set locks to acquire new images
        self.rgb_lock = True
        self.depth_lock = True
        
        # Create subscribers if they don't exist
        if self.rgb_sub is None:
            self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        # Wait for both images to be received
        start_time = time.time()
        while (self.rgb_lock or self.depth_lock) and time.time() - start_time < timeout:
            rospy.sleep(0.05)  # Short sleep to avoid CPU hogging
        
        # Check if both images were received
        if self.rgb_image is None or self.depth_image is None:
            rospy.logwarn(f"Failed to get images within timeout ({timeout}s)")
            return False
        
        return True
    
    def grasp_object(self, action):
        """
        Execute a grasp based on policy prediction
        
        Args:
            action: The predicted action from the policy
        """
        try:
            pos = [action[0], action[1], action[2]]
            aperture = action[3]

            # Convert to joint angles
            joint_angles = self.get_joint_angles_from_pose(pos)
            
            if joint_angles is None:
                rospy.logerr("Failed to compute joint angles, aborting grasp")
                return
            
            # joint_angles = [110.0, 36.0, 60.0, 20.0, 90.0, 30.0]
            joint_angles = [90.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Left obstacle
            # joint_angles = [70.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Target
            # joint_angles = [60.0, 36.0, 60.0, 20.0, 90.0, 30.0] # Right obstacle

            # Execute the grasp sequence
            self.step(joint_angles, aperture)
                        
        except Exception as e:
            rospy.logerr(f"Error executing grasp: {str(e)}")

        general_utils.delete_episodes_misc(self.TEST_DIR)

        # Get new observation after grasp
        return self.get_observation()
    
    def get_joint_angles_from_pose(self, pos):
        """Use inverse kinematics to get joint angles for a pose"""
        x, y, z = self.convert_sim_to_robot_pose(pos)
        
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
        gripper_angle = np.interp(aperture, [0, 1], [30, 140])
        self.gripper_angle = gripper_angle
        
        arm_joint = ArmJoint()
        arm_joint.id = 6  # Gripper servo ID
        arm_joint.angle = gripper_angle
        arm_joint.run_time = run_time
        arm_joint.joints = []
        self.pub_arm.publish(arm_joint)
    
    def get_observation(self, timeout=5.0):
        """
        Get observation for policy input
        
        Returns:
            Observation dictionary
        """
        print("Acquiring latest images")
        # Get the latest images on demand
        success = self.get_latest_image(timeout=5.0)
        
        if not success:
            print("Failed to get images")
            rospy.logerr("Failed to get observation")
            return None
        
        print("Latest images acquired")

        # Wait for both images to be received
        start_time = time.time()
        while self.pred_mask is None and time.time() - start_time < timeout:
            rospy.sleep(0.2)  # Short sleep to avoid CPU hogging

        # Create observation dictionary
        obs = {
            'color': self.rgb_image.copy(),  # Create copies to avoid reference issues
            'depth': self.depth_image.copy()
        }
        
        return obs
            
    def eval_agent(self, args):
        self.args = args

        rng = np.random.RandomState()
        rng.seed(args.seed)

        for i in range(args.n_scenes):
            episode_seed = rng.randint(0, pow(2, 32) - 1)
            logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

            self.run()
            # self.test(args)

        rospy.is_shutdown()

    def call_policy_manager(self, timeout=5.0):
        if self.raw_color_image is None or self.raw_depth_image is None:
            rospy.logerr("No images available")
            return
        
        obs_data = ObservData()
        obs_data.color_image = self.raw_color_image
        obs_data.depth_image = self.raw_depth_image 
        if self.target_mask is not None:
            obs_data.target_mask = self.target_mask

        self.observation_pub.publish(obs_data)
        print("Publishing observation data to policy manager for action data")

        # Reset segmentation data
        self.raw_color_image, self.raw_depth_image = None, None

        # Wait for both images to be received
        start_time = time.time()
        while self.action is None and time.time() - start_time < timeout:
            rospy.sleep(0.5)  # Short sleep to avoid CPU hogging

    def test(self, args):
        for i in range(10):
            processed_masks = self.get_masks()
            if len(processed_masks) == 0:
                rospy.logerr("Failed to get masks")
                return
            print("Got masks")
            print(f"Iter {i}: {len(processed_masks)} masks")

    
    def run(self):
        """Main control loop"""
        rate = rospy.Rate(1)  # 1 Hz, adjust as needed

        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            rospy.logerr("Failed to get initial observation")
            return
        
        max_steps = 6
        attempts = 0
        while attempts < max_steps:
            # self.call_policy_manager()

            if self.sub is None:
                self.sub = rospy.Subscriber('/machine_robot', String, self.callback)

            rate = rospy.Rate(1)
            while not rospy.is_shutdown():
                self.pub.publish(String(data="Hello from robot"))
                print("Published message")
                rate.sleep()



            if self.action is None:
                rospy.logerr("Failed to get action from policy manager")
                attempts += 1
                continue

            if self.action[0] == 0 and self.action[1] == 0 and self.action[2] == 0 and self.action[3] == 0:
                print("Action is zero. Target is not available")
                break
        
            try:
                # Execute grasp based on policy
                print("Executing grasp with action:", self.action)
                next_obs = self.grasp_object(self.action)
                if next_obs is None:
                    rospy.logerr("Failed to get observation after grasp")
                    attempts += 1
                    continue

                obs = copy.deepcopy(next_obs)
                cv2.imwrite(os.path.join(self.TEST_DIR, "color0.png"), obs['color'])
                cv2.imwrite(os.path.join(self.TEST_DIR, "depth0.png"), obs['depth'])
                
                attempts += 1
                rate.sleep()
                
            except KeyboardInterrupt:
                print("Shutting down")
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {str(e)}")
                attempts += 1

    def cleanup(self):
        """Clean up subscribers to prevent issues on shutdown"""
        if self.rgb_sub is not None:
            self.rgb_sub.unregister()
        if self.depth_sub is not None:
            self.depth_sub.unregister()
        if self.camera_info_sub is not None:
            self.camera_info_sub.unregister()

    def callback(self, msg):
        print("Received from machine:", msg.data)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='ae', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--ae_model', default='save/ae/ae_model_best.pt', type=str, help='')
    parser.add_argument('--sre_model', default='save/sre/sre_model_best.pt', type=str, help='')
    parser.add_argument('--reg_model', default='downloads/reg_model.pt', type=str, help='')
    parser.add_argument('--seed', default=16, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='save/pc-ou-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    parser.add_argument('--sequence_length', default=1, type=int, help='')
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=10, type=int, help='This should not be less than the maximum possible number of objects in the scene, which from list Environment.nr_objects is 9')
    parser.add_argument('--step', default=500, type=int, help='')

    # args for act
    parser.add_argument('--chunk_size', default=3, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    print(f"You are using {args.device}")

    controller = PolicyRobotController()
    controller.eval_agent(args)
    controller.cleanup()

# if __name__ == '__main__':
#     rospy.init_node('test_node')

#     def callback(msg):
#         print("Received from machine:", msg.data)

#     pub = rospy.Publisher('/robot_machine', String, queue_size=10)
#     while pub.get_num_connections() == 0:
#         rospy.loginfo("Waiting for local machine to subscribe...")
#         rospy.sleep(0.5)

#     rospy.Subscriber('/machine_robot', String, callback)

#     rate = rospy.Rate(1)
#     while not rospy.is_shutdown():
#         pub.publish(String(data="Hello from robot"))
#         print("Published message")
#         rate.sleep()

#     rospy.spin()
