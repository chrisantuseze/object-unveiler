#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
from policy import grasping
import rospy
import cv2
import time
import copy
import numpy as np
import open3d as o3d  # For point cloud operations
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from dofbot_pro_info.msg import ArmJoint
from dofbot_pro_info.msg import *
from dofbot_pro_info.srv import *
from utils import general_utils
import utils.logger as logging
from env.environment import Environment
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy

class PolicyRobotController:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('policy_robot_controller')
        self.TEST_DIR = "dofbot_pro_ws/src/dofbot_pro_info/scripts"
        
        # Publisher to control the robot arm
        self.pub_arm = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        self.ik_client = rospy.ServiceProxy("get_kinemarics", kinemarics)

        # Subscribers
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)

        # Image Storage
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.point_cloud = None
        self.state = None
        self.intrinsics = None  # Camera intrinsics

        self.get_point_cloud = False
        self.get_rgb_image = False
        
        # Robot arm parameters
        self.home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 30.0]  # Default home position
        # self.home_position = [90.0, 70.0, 0.0, 0.0, 90.0, 30.0]  # Default home position
        self.gripper_angle = 30.0
        
        # Wait for publisher to connect
        rospy.sleep(1)
        
        # Move to home position at startup
        self.move_arm_to_position(self.home_position)
        print("Policy Robot Controller initialized")

    def camera_info_callback(self, msg):
        """ Extract camera intrinsic parameters. """
        self.intrinsics = np.array(msg.K).reshape(3, 3)  # Intrinsic matrix (3x3)

    def rgb_callback(self, msg):
        """ Callback to receive the RGB image. """
        if not self.get_rgb_image:
            rospy.loginfo("Waiting for depth image...")
            return
        
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to OpenCV format
            self.get_rgb_image = False
            cv2.imwrite("saved_rgb_image.png", self.rgb_image)
        except Exception as e:
            rospy.logerr(f"RGB conversion error: {e}")

    def depth_callback(self, msg):
        """ Callback to receive the depth image and compute the point cloud. """
        if not self.get_point_cloud:
            rospy.loginfo("Waiting for RGB image...")
            return 
        
        try:
            # Convert ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Depth is in 16-bit unsigned int
            self.depth_image = self.depth_image.astype(np.float32) / 1000.0  # Convert to meters
            cv2.imwrite("saved_depth_image.png", self.depth_image)

            if self.rgb_image is not None and self.intrinsics is not None:
                self.generate_point_cloud()
                self.state = self.get_fused_heightmap()
                cv2.imwrite("state.png", self.state)

            self.get_point_cloud = False
        
        except Exception as e:
            rospy.logerr(f"Depth conversion error: {e}")

    def generate_point_cloud(self):
        """ Generates and saves a point cloud using depth and RGB data. """
        height, width = self.depth_image.shape
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]  # Focal lengths
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]  # Optical center

        points = []
        colors = []

        for v in range(height):
            for u in range(width):
                Z = self.depth_image[v, u]
                if Z > 0:  # Ignore zero-depth points
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append((X, Y, Z))

                    # Get RGB color from the RGB image
                    color = self.rgb_image[v, u] / 255.0  # Normalize to [0, 1]
                    colors.append((color[2], color[1], color[0]))  # Convert BGR to RGB

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save the point cloud
        # o3d.io.write_point_cloud("pointcloud.ply", pcd)
        # rospy.loginfo("Point cloud saved as pointcloud.ply")
        self.point_cloud = pcd

    def get_fused_heightmap(self):
        bounds = [[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]]
        pixel_size = 0.005

        xyz = np.asarray(self.point_cloud.points)
        seg_class = np.asarray(self.point_cloud.colors)

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

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(height_grid)
        # ax[1].imshow(seg_grid)
        # plt.show()

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
            self.get_point_cloud = True
            self.get_rgb_image = True
                        
        except Exception as e:
            rospy.logerr(f"Error executing grasp: {str(e)}")

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

        if self.rgb_image and self.state:
            obs = {
                'color': self.rgb_image,
                'depth': self.depth_image
            }
            return obs
        
        return None
    
    def eval_agent(self, args):
        print("Running eval...")
        with open('yaml/bhand.yml', 'r') as stream:
            params = yaml.safe_load(stream)

        env = Environment(params)

        policy = Policy(args, params)
        policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model=args.sre_model)

        segmenter = ObjectSegmenter()

        rng = np.random.RandomState()
        rng.seed(args.seed)

        for i in range(args.n_scenes):
            episode_seed = rng.randint(0, pow(2, 32) - 1)
            logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

            self.run(policy, env, segmenter, rng)

        rospy.is_shutdown()
    
    def run(self, policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng):
        """Main control loop"""
        rate = rospy.Rate(1)  # 1 Hz, adjust as needed
        
        self.get_point_cloud = True
        self.get_rgb_image = True

        obs = self.get_observation()

        processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'], dir=self.TEST_DIR, bbox=True)
        cv2.imwrite(os.path.join(self.TEST_DIR, "initial_scene.png"), pred_mask)
        cv2.imwrite(os.path.join(self.TEST_DIR, "color0.png"), obs['color'])

        target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'], rng)

        max_steps = 6
        attempts = 0
        while attempts < max_steps:
            state = policy.state_representation(obs)
            action = policy.exploit_unveiler(state, obs['color'], target_mask, processed_masks, bboxes)
        
            try:
                # Execute grasp based on policy
                obs = self.grasp_object(action)

                processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'], dir=self.TEST_DIR, bbox=True)
                target_id, target_mask = grasping.find_target(processed_masks, target_mask)

                if target_id == -1:
                    res = input("\nDo you think the target is available? (y/n) ")
                    if res.lower() == "y":
                        target_id = int(input("\nWhat is the index? "))
                        target_mask = processed_masks[target_id]
                        continue
                
                rate.sleep()
                
            except KeyboardInterrupt:
                print("Shutting down")
            except Exception as e:
                rospy.logerr(f"Error in main loop: {str(e)}")

if __name__ == '__main__':
    try:
        controller = PolicyRobotController()
        controller.eval_agent()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"Error in calling PolicyRobotController: {str(e)}")