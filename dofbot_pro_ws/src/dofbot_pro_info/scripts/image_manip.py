import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import time

class HeightmapGenerator:
    def __init__(self):        
        # Define workspace bounds (x_min, x_max, y_min, y_max, z_min, z_max)
        # Adjust these values based on your actual robot workspace
        self.bounds = np.array([[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.3]])
        # self.bounds_ = np.array([[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]])

        self.pix_size = 0.002  # 2mm per pixel, adjust as needed
        # self.pix_size_ = 0.005
        
        # Camera pose (adjust these values based on your real camera setup)
        self.camera_pos = np.array([0.0, 0.5, 0.5])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])
        
    def get_camera_pose(self):
        """
        Computes the camera pose w.r.t. world
        """
        # Compute z axis
        z = self.camera_target - self.camera_pos
        z /= np.linalg.norm(z)

        # Compute y axis
        dist = np.dot(z, -self.camera_up)
        y = -self.camera_up - dist * z
        y /= np.linalg.norm(y)

        # Compute x axis as the cross product of y and z
        x = np.cross(y, z)
        x /= np.linalg.norm(x)

        camera_pose = np.eye(4)
        camera_pose[0:3, 0] = x
        camera_pose[0:3, 1] = y
        camera_pose[0:3, 2] = z
        camera_pose[0:3, 3] = self.camera_pos.reshape(3)
        return camera_pose
    
    def get_pointcloud(self, color_img, depth_img, camera_intrinsics):
        """
        Convert RGB-D images to point cloud
        """
        # Get depth image size
        im_h = depth_img.shape[0]
        im_w = depth_img.shape[1]

        # Project depth into 3D point cloud in camera coordinates
        pix_x, pix_y = np.meshgrid(np.linspace(0, im_w-1, im_w), np.linspace(0, im_h-1, im_h))
        
        # Extract intrinsic parameters
        fx = camera_intrinsics[0, 0]  # Focal length x
        fy = camera_intrinsics[1, 1]  # Focal length y
        cx = camera_intrinsics[0, 2]  # Principal point x
        cy = camera_intrinsics[1, 2]  # Principal point y
        
        # Project to 3D
        cam_pts_x = np.multiply(pix_x-cx, depth_img/fx)
        cam_pts_y = np.multiply(pix_y-cy, depth_img/fy)
        cam_pts_z = depth_img.copy()
        
        # Reshape for point cloud
        cam_pts_x = cam_pts_x.reshape(-1, 1)
        cam_pts_y = cam_pts_y.reshape(-1, 1)
        cam_pts_z = cam_pts_z.reshape(-1, 1)

        # Reshape image into colors for 3D point cloud
        rgb_pts_r = color_img[:,:,2]  # OpenCV uses BGR
        rgb_pts_g = color_img[:,:,1]
        rgb_pts_b = color_img[:,:,0]
        rgb_pts_r = rgb_pts_r.reshape(-1, 1)
        rgb_pts_g = rgb_pts_g.reshape(-1, 1)
        rgb_pts_b = rgb_pts_b.reshape(-1, 1)

        # Combine points and colors
        cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
        rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

        return cam_pts, rgb_pts
    
    def generate_heightmap(self, color_image, depth_image, intrinsics):
        """
        Generate color and depth heightmaps from RGB-D images
        """
        if intrinsics is None:
            rospy.logerr("Camera intrinsics not yet received")
            return None, None
            
        if color_image is None or depth_image is None:
            rospy.logerr("RGB or depth image not available")
            return None, None
            
        # Get point cloud from RGB-D images
        surface_pts, color_pts = self.get_pointcloud(color_image, depth_image, intrinsics)
        
        # Get camera pose
        cam_pose = self.get_camera_pose()
        
        # Transform 3D point cloud from camera coordinates to robot coordinates
        surface_pts = np.transpose(
            np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts)) + 
            np.tile(cam_pose[0:3, 3].reshape(3, 1), (1, surface_pts.shape[0]))
        )
        
        # Compute heightmap size
        heightmap_size = np.round(((self.bounds[1][1] - self.bounds[1][0]) / self.pix_size,
                                  (self.bounds[0][1] - self.bounds[0][0]) / self.pix_size)).astype(int)
        
        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:, 2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = color_pts[sort_z_ind]
        
        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.logical_and(
                        surface_pts[:, 0] >= self.bounds[0][0], 
                        surface_pts[:, 0] < self.bounds[0][1]
                    ), 
                    surface_pts[:, 1] >= self.bounds[1][0]
                ), 
                surface_pts[:, 1] < self.bounds[1][1]
            ), 
            surface_pts[:, 2] < self.bounds[2][1]
        )
        
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]
        
        # Create heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        
        heightmap_pix_x = np.floor((surface_pts[:, 0] - self.bounds[0][0]) / self.pix_size).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:, 1] - self.bounds[1][0]) / self.pix_size).astype(int)
        
        color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
        color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
        color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
        
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
        
        # Set minimum height
        z_bottom = self.bounds[2][0]
        depth_heightmap = depth_heightmap - z_bottom
        depth_heightmap[depth_heightmap < 0] = 0
        depth_heightmap[depth_heightmap == -z_bottom] = np.nan
        
        # Flip heightmaps to match robot coordinate system
        color_heightmap = cv2.flip(color_heightmap, 1)
        depth_heightmap = cv2.flip(depth_heightmap, 1)
        
        return depth_heightmap
        
    def save_heightmaps(self, color_heightmap, depth_heightmap):
        """
        Save heightmaps to disk
        """
        if color_heightmap is not None and depth_heightmap is not None:
            cv2.imwrite("color_heightmap.png", color_heightmap)
            
            # Normalize depth for visualization
            depth_vis = cv2.normalize(depth_heightmap, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            cv2.imwrite("depth_heightmap.png", depth_vis)
            
            # Save actual depth values (for use in algorithms)
            np.save("depth_heightmap.npy", depth_heightmap)
            
            return True
        return False
    
    def run(self):
        """
        Main process to capture images and generate heightmaps
        """
        if self.get_latest_image():
            color_heightmap, depth_heightmap = self.generate_heightmap()
            if self.save_heightmaps(color_heightmap, depth_heightmap):
                rospy.loginfo("Successfully generated and saved heightmaps")
                return color_heightmap, depth_heightmap
            else:
                rospy.logerr("Failed to save heightmaps")
        else:
            rospy.logerr("Failed to capture images")
        
        return None, None

# if __name__ == "__main__":
#     rospy.init_node("dofbot_heightmap_generator")
#     generator = HeightmapGenerator()
    
#     # Set camera position and orientation
#     generator.camera_pos = np.array([0.0, 0.5, 0.5])
#     generator.camera_target = np.array([0.0, 0.0, 0.0])
#     generator.camera_up = np.array([0.0, 0.0, 1.0])
    
#     # Set workspace bounds
#     generator.bounds = np.array([[-0.3, 0.3], [-0.3, 0.3], [0.0, 0.3]])
    
#     # Generate heightmaps
#     color_map, depth_map = generator.run()
    
#     # Keep the node running
#     rospy.spin()