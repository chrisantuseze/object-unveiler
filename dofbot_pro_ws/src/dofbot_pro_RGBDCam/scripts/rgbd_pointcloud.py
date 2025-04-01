#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2 as cv
import numpy as np
import open3d as o3d  # For point cloud operations
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

class RGBD_PointCloud:
    def __init__(self):
        rospy.init_node("rgbd_pointcloud", anonymous=False)
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)

        # Image Storage
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.point_cloud = None
        self.intrinsics = None  # Camera intrinsics

    def camera_info_callback(self, msg):
        """ Extract camera intrinsic parameters. """
        self.intrinsics = np.array(msg.K).reshape(3, 3)  # Intrinsic matrix (3x3)

    def rgb_callback(self, msg):
        """ Callback to receive the RGB image. """
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert to OpenCV format
            cv.imwrite("saved_rgb_image.png", self.rgb_image)
        except Exception as e:
            rospy.logerr(f"RGB conversion error: {e}")

    def depth_callback(self, msg):
        """ Callback to receive the depth image and compute the point cloud. """
        try:
            # Convert ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # Depth is in 16-bit unsigned int
            self.depth_image = self.depth_image.astype(np.float32) / 1000.0  # Convert to meters
            cv.imwrite("saved_depth_image.png", self.depth_image)

            if self.rgb_image is not None and self.intrinsics is not None:
                self.generate_point_cloud()
                state = self.get_fused_heightmap()
                cv.imwrite("state.png", state)
        
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

        return cv.flip(height_grid, 1)

if __name__ == '__main__':
    rgbd_pc = RGBD_PointCloud()
    rospy.spin()
