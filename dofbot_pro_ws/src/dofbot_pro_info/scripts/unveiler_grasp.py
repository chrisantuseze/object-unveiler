#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import yaml
from policy import grasping
import rospy
import cv2
import time
import copy
import numpy as np
import argparse

from PIL import Image

import open3d as o3d  # For point cloud operations
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from dofbot_pro_info.msg import ArmJoint, SegmentationData, Image_Msg
from dofbot_pro_info.msg import *
from dofbot_pro_info.srv import *
from dofbot_pro_ws.src.dofbot_pro_info.scripts.image_manip import HeightmapGenerator
from utils import general_utils
import utils.logger as logging
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy

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

        self.segment_sub = rospy.Subscriber('/segmentation/data', SegmentationData, self.segment_callback)
        self.image_pub = rospy.Publisher('/image_data', Image_Msg, queue_size=1)

        self.processed_masks, self.pred_mask, self.raw_masks, self.bboxes = [], None, [], []

        
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

        self.hmap_generator = HeightmapGenerator()

    def request_image_segmentation(self, raw_data):
        """
        Request image segmentation from the segmenter
        """
        self.img = self.bridge.imgmsg_to_cv2(raw_data, "bgr8")
        size = self.img.shape
        
        image = Image_Msg()
        image.height = size[0] # 480
        image.width = size[1] # 640
        image.channels = size[2] # 3
        image.data = raw_data.data

        print("Requesting image segmentation...")
        
        self.image_pub.publish(image)

    def segment_callback(self, msg):
        print("Received segmentation data")
        self.pred_mask = self.bridge.imgmsg_to_cv2(msg.pred_mask, desired_encoding='mono8')
        self.raw_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding='mono8') for m in msg.raw_masks]
        self.processed_masks = [self.bridge.imgmsg_to_cv2(m, desired_encoding='mono8') for m in msg.processed_masks]
        self.bboxes = list(zip(msg.bbox_x1, msg.bbox_y1, msg.bbox_x2, msg.bbox_y2))

    def camera_info_callback(self, msg):
        """ Extract camera intrinsic parameters. """
        self.intrinsics = np.array(msg.K).reshape(3, 3)  # Intrinsic matrix (3x3)
        self.camera_info_received = True
        # We can keep this subscription active all the time as the camera parameters don't change

    def rgb_callback(self, msg):
        """ Callback to receive the RGB image. """
        if self.rgb_lock:
            # try:
            #     self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to OpenCV format
            #     self.rgb_image = cv2.flip(self.rgb_image, -1)

            #     print("Received RGB image")
            #     self.request_image_segmentation(self.rgb_image)
            #     cv2.imwrite(os.path.join(self.TEST_DIR, "saved_rgb_image.png"), self.rgb_image)
                
            #     self.rgb_lock = False  # Release the lock
            # except Exception as e:
            #     rospy.logerr(f"RGB conversion error: {e}")
            #     self.rgb_lock = False  # Make sure to release the lock even if there's an error

            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to OpenCV format
            self.rgb_image = cv2.flip(self.rgb_image, -1)

            print("Received RGB image")
            self.request_image_segmentation(msg)
            cv2.imwrite(os.path.join(self.TEST_DIR, "saved_rgb_image.png"), self.rgb_image)
            
            self.rgb_lock = False  # Release the lock

    def depth_callback(self, msg):
        """ Callback to receive the depth image. """
        if self.depth_lock:
            try:
                # Convert ROS depth image to OpenCV format
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Depth is in 16-bit unsigned int
                self.depth_image = cv2.flip(self.depth_image, -1)

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

    def generate_point_cloud(self, color_img, depth_img):
        """ Generates and saves a point cloud using depth and RGB data. """
        height, width = depth_img.shape
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]  # Focal lengths
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]  # Optical center

        points = []
        colors = []

        for v in range(height):
            for u in range(width):
                Z = depth_img[v, u]
                if Z > 0:  # Ignore zero-depth points
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append((X, Y, Z))

                    # Get RGB color from the RGB image
                    color = color_img[v, u] / 255.0  # Normalize to [0, 1]
                    colors.append((color[2], color[1], color[0]))  # Convert BGR to RGB

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save the point cloud
        self.point_cloud = pcd
        return pcd

    def get_fused_heightmap(self, obs):
        pcd = self.generate_point_cloud(obs['color'], obs['depth'])

        bounds = [[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]]
        pixel_size = 0.005

        xyz = np.asarray(pcd.points)
        seg_class = np.asarray(pcd.colors)

        # Compute heightmap size
        heightmap_size = np.round(((bounds[1][1] - bounds[1][0]) / pixel_size,
                                (bounds[0][1] - bounds[0][0]) / pixel_size)).astype(int)

        height_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)
        seg_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)

        for i in range(xyz.shape[0]):
            x = xyz[i][0]
            y = xyz[i][1]
            z = xyz[i][2]

            idx_x = int(np.floor((x + bounds[0][1]) / pixel_size))
            idx_y = int(np.floor((y + bounds[1][1]) / pixel_size))

            if 0 < idx_x < heightmap_size[0] - 1 and 0 < idx_y < heightmap_size[1] - 1:
                if height_grid[idx_y][idx_x] < z:
                    height_grid[idx_y][idx_x] = z
                    seg_grid[idx_y][idx_x] = seg_class[i, 0]

        return cv2.flip(height_grid, 1)
    
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

    def convert_sim_to_robot_pose(self, sim_pos):
        """Convert simulation position/orientation to robot coordinates"""
        # This is a placeholder - implement based on your coordinate systems
        # You may need to scale, offset, and/or rotate coordinates
        
        # Example conversion (adjust based on your setup):
        robot_x = 102.90 #sim_pos[0] * 100  # Convert to cm
        robot_y = 29.40 #sim_pos[1] * 100
        robot_z = 80 #sim_pos[2] * 100
        
        return robot_x, robot_y, robot_z
    
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
            
    def eval_agent(self, args):
        self.args = args
        print("Running eval...")
        with open('yaml/bhand.yml', 'r') as stream:
            params = yaml.safe_load(stream)

        # policy = Policy(args, params)
        # policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model=args.sre_model)

        rng = np.random.RandomState()
        rng.seed(args.seed)

        for i in range(args.n_scenes):
            episode_seed = rng.randint(0, pow(2, 32) - 1)
            logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

            # self.run(policy, rng)
            self.test(args)

        rospy.is_shutdown()

    def get_masks(self, timeout=5.0):
        obs = self.get_observation()
        if obs is None:
            rospy.logerr("Failed to get initial observation")
            return
        
        print("Got initial observation. And now getting segmentations...")

        
        # Wait for both images to be received
        start_time = time.time()
        while self.pred_mask is None and time.time() - start_time < timeout:
            rospy.sleep(0.05)  # Short sleep to avoid CPU hogging

        return self.processed_masks

    def test(self, args):
        for i in range(10):
            processed_masks = self.get_masks()
            if processed_masks is None:
                rospy.logerr("Failed to get masks")
                return
            print("Got masks")
            print(f"Iter {i}: {len(processed_masks)} masks")

    
    def run(self, policy: Policy, rng):
        """Main control loop"""
        rate = rospy.Rate(1)  # 1 Hz, adjust as needed

        # Get initial observation
        obs = self.get_observation()
        if obs is None:
            rospy.logerr("Failed to get initial observation")
            return
        
        print("Got initial observation. And now getting segmentations...")

        args_ = copy.deepcopy(self.args)
        args_.device = torch.device("cpu")
        segmenter = ObjectSegmenter(args_, is_real=True)

        processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'], dir=self.TEST_DIR, bbox=True, dim=(240, 320))#(480, 640))
        cv2.imwrite(os.path.join("dofbot_pro_ws/src/dofbot_pro_info/scripts", "initial_scene.png"), pred_mask)
        cv2.imwrite(os.path.join(self.TEST_DIR, "color0.png"), obs['color'])
        cv2.imwrite(os.path.join(self.TEST_DIR, "depth0.png"), obs['depth'])

        print("len(processed_masks):", len(processed_masks))
        target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'], rng)
        print("Target ID:", target_id)
        cv2.imwrite(os.path.join("dofbot_pro_ws/src/dofbot_pro_info/scripts", "initial_target_mask.png"), target_mask)

        max_steps = 6
        attempts = 0
        while attempts < max_steps:
            # state = policy.state_representation(obs)
            # np.save(os.path.join(self.TEST_DIR, 'color.npy'), obs['color'])
            # np.save(os.path.join(self.TEST_DIR, 'depth.npy'), obs['depth'])
            # np.save(os.path.join(self.TEST_DIR, 'intrinsics.npy'), self.intrinsics)
            
            # state = self.hmap_generator.generate_heightmap(obs['color'], obs['depth'], self.intrinsics)
            state = policy.get_dmap(obs['color'], obs['depth'], self.intrinsics)
            # np.save(os.path.join(self.TEST_DIR, 'state.npy'), state)
            print("Gotten the state")

            torch.cuda.empty_cache()

            print("Getting actions...")
            # action = policy.exploit_unveiler(state, obs['color'], target_mask, processed_masks, bboxes)
            action = policy.exploit_real_robot(state, target_mask)
            print("Gotten the action:", action)

            torch.cuda.empty_cache()
        
            try:
                # Execute grasp based on policy
                next_obs = self.grasp_object(action)
                if obs is None:
                    rospy.logerr("Failed to get observation after grasp")
                    attempts += 1
                    continue

                obs = copy.deepcopy(next_obs)

                color_image = obs['color']
                cv2.imwrite(os.path.join(self.TEST_DIR, "maskrcnn_image.png"), color_image)

                rospy.sleep(0.2)

                print("Getting fresh segmentations...")
                segmenter = ObjectSegmenter(args_, is_real=True)
                processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(color_image, dir=self.TEST_DIR, bbox=True, dim=(240, 320))
                cv2.imwrite(os.path.join(self.TEST_DIR, "color0.png"), obs['color'])
                cv2.imwrite(os.path.join(self.TEST_DIR, "depth0.png"), obs['depth'])

                print("len(processed_masks):", len(processed_masks))
                target_id, target_mask = grasping.find_target(processed_masks, target_mask)

                if target_id == -1:
                    res = input("\nDo you think the target is available? (y/n) ")
                    if res.lower() == "y":
                        target_id = int(input("\nWhat is the index? "))
                        target_mask = processed_masks[target_id]
                    else:
                        print("Target not available. Exiting.")
                        break
                
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
    # try:
    #     args = parse_args()
    #     args.device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     print(f"You are using {args.device}")

    #     controller = PolicyRobotController()
    #     try:
    #         controller.eval_agent(args)
    #     except Exception as e:
    #         rospy.logerr(f"Error in eval_agent: {str(e)}")
    #     finally:
    #         controller.cleanup()
    # except rospy.ROSInterruptException as e:
    #     rospy.logerr(f"Error in calling PolicyRobotController: {str(e)}")

    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
    print(f"You are using {args.device}")

    controller = PolicyRobotController()
    controller.eval_agent(args)
    controller.cleanup()