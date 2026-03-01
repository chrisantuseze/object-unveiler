#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import yaml
from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import compute_R_and_t, compute_post_grasp_joints, compute_pre_grasp_joints, compute_real_pts, convert_sim_to_robot_pose, sim_to_robot
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
from utils import general_utils
import utils.logger as logging

from std_msgs.msg import String
from utils.orientation import Quaternion, rot_y

class PolicyRobotController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('policy_robot_controller')
        self.TEST_DIR = "dofbot_pro_ws/src/dofbot_pro_info/scripts/images"
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)
        
        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)

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
        self.camera_info_sub = None

        self.raw_color_image, self.raw_depth_image, self.target_mask = None, None, None
        self.action = None

        # Wait for publisher to connect and camera info to be received
        rospy.sleep(1)

        # Robot arm parameters
        self.home_position = [90.0, 60.0, 60.0, 20.0, 90.0, 30.0] #[90.0, 90.0, 60.0, 0.0, 90.0, 40] #30.0]  # Default home position
        self.gripper_angle = 30.0
        
        # Move to home position at startups
        self.move_arm_to_position(self.home_position)
        print("Policy Robot Controller initialized")
        
        # Wait for camera info to be received
        start_time = time.time()
        while not self.camera_info_received and time.time() - start_time < 10:
            rospy.sleep(0.1)
        
        if not self.camera_info_received:
            rospy.logwarn("Camera info not received within timeout. Some features may not work properly.")

    def camera_info_callback(self, msg):
        """ Extract camera intrinsic parameters. """
        if self.camera_info_received:
            self.intrinsics = np.array(msg.K).reshape(3, 3)  # Intrinsic matrix (3x3)
            self.camera_info_received = False
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
        self.camera_info_received = True
        
        # Create subscribers if they don't exist
        if self.rgb_sub is None:
            self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        if self.camera_info_sub is None:
            self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        
        # Wait for both images to be received
        start_time = time.time()
        while (self.rgb_lock or self.depth_lock) and time.time() - start_time < timeout:
            rospy.sleep(0.05)  # Short sleep to avoid CPU hogging
        
        # Check if both images were received
        if self.rgb_image is None or self.depth_image is None:
            rospy.logwarn(f"Failed to get images within timeout ({timeout}s)")
            return False
        
        return True
    
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

        # Create observation dictionary
        obs = {
            'color': self.rgb_image.copy(),  # Create copies to avoid reference issues
            'depth': self.depth_image.copy()
        }
        
        return obs
    
    def move_arm_to_position(self, joint_positions, run_time=2000):
        """Send joint positions to the robot arm"""
        joint_positions[5] = self.gripper_angle
        arm_joint = ArmJoint()
        arm_joint.joints = joint_positions
        arm_joint.run_time = run_time
        self.pub_arm.publish(arm_joint)

        print("joint_positions:", joint_positions)
    
    def eval_agent(self, args):
        self.args = args

        rng = np.random.RandomState()
        rng.seed(args.seed)

        for i in range(args.n_scenes):
            episode_seed = rng.randint(0, pow(2, 32) - 1)
            logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

            self.run(i)

        rospy.is_shutdown()

    def run(self, i):
        """Main control loop"""
        rate = rospy.Rate(1)  # 1 Hz, adjust as needed

        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            rospy.logerr("Failed to get initial observation")
            return
        
        np.save(os.path.join(self.TEST_DIR, f"color_image_{i}.npy"), obs['color'])
        cv2.imwrite(os.path.join(self.TEST_DIR, f"rgb_image_{i}.png"), obs['color'])

        # Wait for both images to be received
        start_time = time.time()
        while self.action is None and time.time() - start_time < 20:
            rospy.sleep(0.5)  # Short sleep to avoid CPU hogging
        

    def cleanup(self):
        """Clean up subscribers to prevent issues on shutdown"""
        if self.rgb_sub is not None:
            self.rgb_sub.unregister()
        if self.depth_sub is not None:
            self.depth_sub.unregister()
        if self.camera_info_sub is not None:
            self.camera_info_sub.unregister()

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