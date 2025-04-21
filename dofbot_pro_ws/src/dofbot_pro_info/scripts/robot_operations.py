import open3d as o3d  # For point cloud operations
import numpy as np
import cv2
import torch  # For image processing


def generate_point_cloud(color_img, depth_img, intrinsics, point_cloud):
        """ Generates and saves a point cloud using depth and RGB data. """
        height, width = depth_img.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Focal lengths
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]  # Optical center

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
        point_cloud = pcd
        return pcd

def get_fused_heightmap(obs):
    pcd = generate_point_cloud(obs['color'], obs['depth'])

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


def compute_pre_grasp_joints(grasp_joints):
    """Compute a pre-grasp position slightly above the grasp position"""
    pre_grasp = grasp_joints.copy()
    pre_grasp[2] += 20  # Adjust second joint to raise arm
    pre_grasp[3] += 10  # Adjust second joint to raise arm
    return pre_grasp

def compute_post_grasp_joints(grasp_joints):
    """Compute a post-grasp position"""
    post_grasp = grasp_joints.copy()
    post_grasp[1] += 30  # Adjust second joint to lift
    post_grasp[2] -= 20  # Adjust second joint to lift
    return post_grasp

def convert_sim_to_robot_pose(sim_pos):
    """Convert simulation position/orientation to robot coordinates"""
    # This is a placeholder - implement based on your coordinate systems
    # You may need to scale, offset, and/or rotate coordinates
    
    # Example conversion (adjust based on your setup):
    robot_x = 102.90 #sim_pos[0] * 100  # Convert to cm
    robot_y = 29.40 #sim_pos[1] * 100
    robot_z = 80 #sim_pos[2] * 100
    
    return robot_x, robot_y, robot_z


def convert_numpy_masks_to_ros_image_list(masks, bridge):
    image_msgs = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        m = np.squeeze(m)  # Remove channel dim if present
        if m.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {m.shape}")
        image_msgs.append(bridge.cv2_to_imgmsg((m.astype('uint8') * 255), encoding='mono8'))
    return image_msgs