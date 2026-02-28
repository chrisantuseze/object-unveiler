import open3d as o3d  # For point cloud operations
import numpy as np
import cv2
import yaml
import os
import torch  # For image processing
from dofbot_pro_info.srv import *

_EXTRINSICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'yaml', 'cam_extrinsics.yaml'
)

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


# ---------------------------------------------------------------------------
# Real-robot camera extrinsics & heightmap
# ---------------------------------------------------------------------------

def load_T_cam_base():
    """Load the measured camera-to-robot-base 4x4 transform from yaml/cam_extrinsics.yaml."""
    path = _EXTRINSICS_PATH
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['T_cam_base'], dtype=np.float64)


def get_real_heightmap(color_bgr, depth_m, intrinsics_3x3, T_cam_base, bounds, pix_size):
    """
    Build a top-down depth heightmap from a single RGB-D frame.

    Parameters
    ----------
    color_bgr      : (H, W, 3) uint8   — BGR image from the Orbbec camera
    depth_m        : (H, W)    float32 — depth in metres (raw 16UC1 / 1000)
    intrinsics_3x3 : (3, 3)    float64 — camera K matrix
    T_cam_base     : (4, 4)    float64 — camera→robot-base rigid transform
    bounds         : (3, 2)    float64 — [[xmin,xmax],[ymin,ymax],[zmin,zmax]] metres
    pix_size       : float              — metres per pixel in the output heightmap

    Returns
    -------
    height_grid : (H_map, W_map) float32 — top-down heightmap in robot base frame
    """
    H, W = depth_m.shape
    fx = intrinsics_3x3[0, 0]
    fy = intrinsics_3x3[1, 1]
    cx = intrinsics_3x3[0, 2]
    cy = intrinsics_3x3[1, 2]

    # Back-project depth pixels to 3-D points in camera frame (vectorised)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_m > 0.05) & (depth_m < 1.5)

    z_c = depth_m[valid].astype(np.float64)
    x_c = (u[valid] - cx) * z_c / fx
    y_c = (v[valid] - cy) * z_c / fy

    pts_cam  = np.vstack([x_c, y_c, z_c, np.ones_like(z_c)])  # 4 × N
    pts_base = (T_cam_base @ pts_cam)[:3, :]                   # 3 × N

    px_arr = pts_base[0]
    py_arr = pts_base[1]
    pz_arr = pts_base[2]

    # Filter to workspace x/y bounds and z upper bound (mirrors utils/general_utils.py)
    in_bounds = (
        (px_arr >= bounds[0][0]) & (px_arr < bounds[0][1]) &
        (py_arr >= bounds[1][0]) & (py_arr < bounds[1][1]) &
        (pz_arr < bounds[2][1])
    )
    px_arr = px_arr[in_bounds]
    py_arr = py_arr[in_bounds]
    pz_arr = pz_arr[in_bounds]

    h_map = int(np.round((bounds[1][1] - bounds[1][0]) / pix_size))
    w_map = int(np.round((bounds[0][1] - bounds[0][0]) / pix_size))
    height_grid = np.zeros((h_map, w_map), dtype=np.float32)

    if px_arr.size > 0:
        # Sort by z ascending so higher points overwrite lower (matches sim)
        sort_z = np.argsort(pz_arr)
        px_arr = px_arr[sort_z]
        py_arr = py_arr[sort_z]
        pz_arr = pz_arr[sort_z]

        idx_x = np.clip(np.floor((px_arr - bounds[0][0]) / pix_size).astype(int), 0, w_map - 1)
        idx_y = np.clip(np.floor((py_arr - bounds[1][0]) / pix_size).astype(int), 0, h_map - 1)
        height_grid[idx_y, idx_x] = pz_arr.astype(np.float32)

    # Normalise heights relative to workspace floor so output is always >= 0
    # (mirrors the depth_heightmap - z_bottom step in utils/general_utils.py)
    z_bottom = float(bounds[2][0])
    height_grid = height_grid - z_bottom
    height_grid[height_grid < 0] = 0
    # Cells that were never filled still equal -z_bottom after subtraction; mark as NaN
    unfilled = np.isclose(height_grid, -z_bottom, atol=1e-4)
    height_grid[unfilled] = np.nan

    return cv2.flip(height_grid, 1)


def compute_pre_grasp_joints(grasp_joints):
    """Compute a pre-grasp position slightly above the grasp position"""
    pre_grasp = grasp_joints.copy()
    pre_grasp[2] += 20  # Adjust third joint to raise arm
    pre_grasp[3] += 10  # Adjust fourth joint to raise arm
    return pre_grasp

def compute_post_grasp_joints(grasp_joints):
    """Compute a post-grasp position"""
    post_grasp = grasp_joints.copy()
    post_grasp[1] += 30  # Adjust second joint to lift
    post_grasp[2] -= 20  # Adjust third joint to lift
    post_grasp[3] += 30  # Adjust fourth joint to lift
    return post_grasp

def convert_sim_to_robot_pose(sim_pos):
    """Convert simulation position/orientation to robot coordinates"""
    # This is a placeholder - implement based on your coordinate systems
    # You may need to scale, offset, and/or rotate coordinates
    
    # Example conversion (adjust based on your setup):
    # robot_x = 102.90 #sim_pos[0] * 100  # Convert to cm
    # robot_y = 29.40 #sim_pos[1] * 100
    # robot_z = 80 #sim_pos[2] * 100

    robot_x = sim_pos[0] * 2  # Convert to cm
    robot_y = sim_pos[1] * 5
    robot_z = sim_pos[2] * 5

    print(f"Converted sim position {sim_pos} to robot coordinates: ({robot_x}, {robot_y}, {robot_z})")
    
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

def compute_real_pts(fk_client):
    # DS-SY15A servos: ±150° around neutral
    real_joint_names = ['axis_x','axis_y','axis_z','wrist','palm','gripper']
    real_joint_limits = {
        'axis_x' : (-150.0, 150.0),  # degrees
        'axis_y' : (-150.0, 150.0),
        'axis_z' : (-150.0, 150.0),
        'wrist'  : (-150.0, 150.0),
        'palm'   : (  90.0,  90.0),  # locked at 90°
        'gripper': (  30.0, 140.0),  # your working aperture
    }

    # 2) Placeholder mapping from slide distances (m) → servo degrees
    LEADSCREW_PITCH_M = 0.005  # e.g. 5 mm per 360°
    def slide_m_to_servo_deg(x_m):
        return np.clip(x_m * (360.0 / LEADSCREW_PITCH_M),
                    real_joint_limits['axis_x'][0],
                    real_joint_limits['axis_x'][1])

    rob_pts_cm = []

    all_sim_joints = [
        [-0.07604594951759225, 0.8412918801978999, 0.22231107132687644, 0.8893918705226926],
        [0.10729796364462763, 0.3117763098558053, 0.4397511753189053, -1.1051408421572297],
        [0.38259759340501365, 0.3514039544648102, 0.39252284505554513, 0.21740771092392852],
        [0.4689198954821414, 0.8334303137593444, 0.9478650166945917, 1.6766761761596298],
        [-0.23664286393688005, -0.6943020955281285, 0.8739982521963279, -1.088581667368953]
    ]

    for i in range(5):
        # a) Sample sim joint values
        sim_joints = all_sim_joints[i]
        
        # d) Build real-robot joint command:
        servo_x    = slide_m_to_servo_deg(sim_joints[0])
        servo_y    = slide_m_to_servo_deg(sim_joints[1])
        servo_z    = slide_m_to_servo_deg(sim_joints[2])
        servo_w    = np.clip(np.degrees(sim_joints[3]),
                            *real_joint_limits['wrist'])
        servo_palm = 90.0
        servo_grip = 30.0

        hw_joints = [servo_x, servo_y, servo_z, servo_w, servo_palm, servo_grip]

        # e) Call your FK service
        # req = kinemaricsRequest()
        # req.joints = hw_joints
        # res = fk_client(req)
        # robot_xyz_cm = np.array([res.x, res.y, res.z])

        robot_xyz_cm = []

        fk_client.wait_for_service()
        request = kinemaricsRequest()
        request.cur_joint1 = hw_joints[0]
        request.cur_joint2 = hw_joints[1]
        request.cur_joint3 = hw_joints[2]
        request.cur_joint4 = hw_joints[3]
        request.cur_joint5 = hw_joints[4]
        request.kin_name = "fk"
        response = fk_client.call(request)
        if isinstance(response, kinemaricsResponse):
            robot_xyz_cm = np.array([response.x, response.y, response.z])

        # f) Store the pair
        rob_pts_cm.append(robot_xyz_cm)
        print("robot_xyz_cm", robot_xyz_cm)


def compute_R_and_t():

    # Example sim→robot correspondences (sim in m, robot in cm)
    # sim_pts   = np.array([[0.7,0,0.2],[0.8,0,0.2],[0.7,0.1,0.2]])
    # robot_pts = np.array([[ 95,120, 25],[105,120, 25],[ 95,130, 25]])

    sim_pts   = np.array([
        [70.55, 12.00, 20.00], 
        [68.53, 11.69, 20.00], 
        [63.70, 3.78, 20.00], 
        [64.68, 1.25, 20.00], 
        [63.96, 2.76, 20.00]
        ])
    robot_pts = np.array([
        [0.26867949, 0.15442937, 0.14680159], 
        [0.04553918, -0.02698488, 0.26418912], 
        [0.2236476, -0.12981582, 0.24510726], 
        [0.24058956, -0.13959727, 0.02176884], 
        [0.13161792, 0.07529682, 0.14620293]
        ])

    # 1) Check collinearity on sim points:
    v1 = sim_pts[1] - sim_pts[0]
    v2 = sim_pts[2] - sim_pts[0]
    if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
        raise RuntimeError("Points are collinear—pick different points")

    # 2) Solve for R,t via Kabsch (scale s=100)
    s = 100.0 
    c_sim   = sim_pts.mean(axis=0)
    c_robot = robot_pts.mean(axis=0)

    # A = (sim_pts - c_sim) * s
    # B = (robot_pts - c_robot)

    A = (sim_pts - c_sim)
    B = (robot_pts - c_robot) * s

    H = A.T.dot(B)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T.dot(U.T)

    t = c_robot - R.dot(c_sim * s)

    tt = c_robot - R.dot(c_sim)

    print(f"R: {R}")
    print(f"t: {t}")
    print(f"tt: {tt}")

    return R, t

def sim_to_robot(sim_pos):
    R = np.array([
        [0.34825496, -0.4036592, 0.84603648],
        [0.84328404, 0.52905984, -0.09469802],
        [-0.4093782, 0.74642812, 0.52464708]
        ])
    # t = np.array([-3746.12030864, -5733.33729857, 1194.44197742])
    t = np.array([-37.28100848, -57.346574, 12.10758558])

    A = R.dot(np.array(sim_pos))
    print(f"sim_pos: {sim_pos}")
    print(f"R: {R}")
    print(f"A: {A}")
    print(f"t: {t}")

    return A + t #@From Chris: Since the sim points are in cm, do we need to scale them?
