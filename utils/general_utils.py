import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import pickle
import yaml
from PIL import Image
import torch
from skimage import transform

import utils.pybullet_utils as p_utils
from utils.constants import *
import utils.logger as logging
import policy.grasping as grasping

try:
    import open3d as o3d
except:
    print("Couldn't import open3d")


def get_pointcloud(depth, seg, intrinsics):
    """
    Creates a point cloud from a depth image given the camera intrinsics parameters.

    Parameters
    ----------
    depth: np.array
        The input image.
    intrinsics: np.array(3, 3)
        Intrinsics parameters of the camera.

    Returns
    -------
    o3d.geometry.PointCloud
        The point cloud of the scene.
    """
    depth = depth
    width, height = depth.shape
    c, r = np.meshgrid(np.arange(height), np.arange(width), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, 0)
    x = np.where(valid, z * (c - intrinsics[0, 2]) / intrinsics[0, 0], 0)
    y = np.where(valid, z * (r - intrinsics[1, 2]) / intrinsics[1, 1], 0)
    pcd = np.dstack((x, y, z))

    colors = np.zeros((seg.shape[0], seg.shape[1], 3))
    colors[:, :, 0] = seg / np.max(seg)
    colors[:, :, 1] = seg / np.max(seg)
    colors[:, :, 2] = np.zeros((seg.shape[0], seg.shape[1]))

    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

    return point_cloud


def get_aligned_point_cloud(color, depth, seg, configs, bounds, pixel_size, plot=False):
    """
    Returns the scene point cloud aligned with the center of the workspace.
    """
    full_point_cloud = o3d.geometry.PointCloud()
    for color, depth, seg, config in zip(color, depth, seg, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        point_cloud = get_pointcloud(depth, seg, intrinsics)

        transform = p_utils.get_camera_pose(config['pos'],
                                                   config['target_pos'],
                                                   config['up_vector'])
        point_cloud = point_cloud.transform(transform)

        crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([bounds[0, 0], bounds[1, 0], bounds[2, 0]]),
                                                       max_bound=np.array([bounds[0, 1], bounds[1, 1], bounds[2, 1]]))
        point_cloud = point_cloud.crop(crop_box)
        
        full_point_cloud += point_cloud

    # full_point_cloud.estimate_normals()
    if plot:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([full_point_cloud, mesh_frame])
    return full_point_cloud


def get_fused_heightmap(obs, configs, bounds, pix_size):
    point_cloud = get_aligned_point_cloud(obs['color'], obs['depth'], obs['seg'], configs, bounds, pix_size)
    xyz = np.asarray(point_cloud.points)
    seg_class = np.asarray(point_cloud.colors)

    # Compute heightmap size
    heightmap_size = np.round(((bounds[1][1] - bounds[1][0]) / pix_size,
                               (bounds[0][1] - bounds[0][0]) / pix_size)).astype(int)

    height_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)
    seg_grid = np.zeros((heightmap_size[0], heightmap_size[0]), dtype=np.float32)

    for i in range(xyz.shape[0]):
        x = xyz[i][0]
        y = xyz[i][1]
        z = xyz[i][2]

        idx_x = int(np.floor((x + bounds[0][1]) / pix_size))
        idx_y = int(np.floor((y + bounds[1][1]) / pix_size))

        if 0 < idx_x < heightmap_size[0] - 1 and 0 < idx_y < heightmap_size[1] - 1:
            if height_grid[idx_y][idx_x] < z:
                height_grid[idx_y][idx_x] = z
                seg_grid[idx_y][idx_x] = seg_class[i, 0]

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(height_grid)
    # ax[1].imshow(seg_grid)
    # plt.show()

    return cv2.flip(height_grid, 1)


def rgb2bgr(rgb):
    """
    Converts a rgb image to bgr

    Parameters
    ----------
    rgb : np.array
        The rgb image

    Returns
    -------
    np.array:
        The image in bgr format
    """
    h, w, c = rgb.shape
    bgr = np.zeros((h, w, c), dtype=np.uint8)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """
    def __init__(self, width, height, fx, fy, cx, cy):

        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    def get_intrinsic_matrix(self):
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def get_focal_length(self):
        return self.fx, self.fy

    def get_principal_point(self):
        return self.cx, self.cy

    def back_project(self, p, z):
        x = (p[0] - self.cx) * z / self.fx
        y = (p[1] - self.cy) * z / self.fy
        return np.array([x, y, z])
    

def min_max_scale(x, range, target_range):
    assert range[1] > range[0]
    assert target_range[1] > target_range[0]

    range_min = range[0] * np.ones(x.shape)
    range_max = range[1] * np.ones(x.shape)
    target_min = target_range[0] * np.ones(x.shape)
    target_max = target_range[1] * np.ones(x.shape)
    

    return target_min + ((x - range_min) * (target_max - target_min)) / (range_max - range_min)


def sample_distribution(prob, rng, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    rand_ind = rng.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())

def get_target_mask(processed_masks, obs, rng):
    id = 0
    if len(processed_masks) > 1:
        rand_id = rng.randint(0, len(processed_masks) - 1)
        # id = grasping.find_topmost_right_object(processed_masks)
        mid_id = grasping.find_central_object(processed_masks)

        # Randomly decide between mid_id and the generated number
        id = rng.choice([mid_id, rand_id])

        target_mask = processed_masks[id]
    elif len(processed_masks) == 1:
        target_mask = processed_masks[id]
    else:
        target_mask = obs['color'][id]

    return target_mask, id

def delete_episodes_misc(path):
    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(path)
    except OSError as e:
        logging.info("Error: %s - %s." % (e.filename, e.strerror))

    if not os.path.exists(path):
        os.mkdir(path)
    pass

def recreate_train():
    path = TRAIN_EPISODES_DIR

    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(path)
    except OSError as e:
        logging.info("Error: %s - %s." % (e.filename, e.strerror))
        
    if not os.path.exists(path):
        os.makedirs(path)

def recreate_test():
    path = TEST_EPISODES_DIR

    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(path)
    except OSError as e:
        logging.info("Error: %s - %s." % (e.filename, e.strerror))
        
    if not os.path.exists(path):
        os.makedirs(path)

def create_dirs():
    recreate_test()
    recreate_train()

    demo_path = 'save/ppg-dataset'
    if not os.path.exists(demo_path):
        os.makedirs(demo_path)

def rad_to_deg(radians):
    return (radians * 180) / np.pi

def deg_to_rad(degrees):
    return (degrees * np.pi) / 180

def accuracy(loss, corrects, loader):
    epoch_loss = loss / len(loader.dataset)
    epoch_acc = corrects.double() / len(loader.dataset)

    return epoch_loss, epoch_acc

def resize_mask(transform, mask, new_size = (100, 100)):
    # Resize the image using seam carving to match with the heightmap
    resized = transform.resize(mask, new_size, mode='reflect', anti_aliasing=True, order=1)
    return resized

def resize_bbox(bbox):
    old_width, old_height = (400, 400)
    new_width, new_height = (100, 100)

    width_scale_factor = new_width / old_width
    height_scale_factor = new_height / old_height

    resized_bbox = [
        bbox[0] * width_scale_factor,  # New x-coordinate
        bbox[1] * height_scale_factor,  # New y-coordinate
        bbox[2] * width_scale_factor,  # New width
        bbox[3] * height_scale_factor  # New height
    ]
    return resized_bbox

def postprocess_single(q_maps, padding_width):
    """
    Remove extra padding
    """

    w = int(q_maps.shape[2] - 2 * padding_width)
    h = int(q_maps.shape[3] - 2 * padding_width)
    remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], w, h))

    for i in range(q_maps.shape[0]):
        for j in range(q_maps.shape[1]):

            # remove extra padding
            q_map = q_maps[i, j, padding_width:int(q_maps.shape[2] - padding_width),
                            padding_width:int(q_maps.shape[3] - padding_width)]
            
            remove_pad[i][j] = q_map.detach().cpu().numpy()

    return remove_pad

def postprocess_multi(q_maps, padding_width):
    """
    Remove extra padding
    """

    w = int(q_maps.shape[3] - 2 * padding_width)
    h = int(q_maps.shape[4] - 2 * padding_width)

    remove_pad = np.zeros((q_maps.shape[0], q_maps.shape[1], q_maps.shape[2], w, h))

    for i in range(q_maps.shape[0]):
        for j in range(q_maps.shape[1]):
            for k in range(q_maps.shape[2]):
                # remove extra padding
                q_map = q_maps[i, j, k, padding_width:int(q_maps.shape[3] - padding_width),
                            padding_width:int(q_maps.shape[4] - padding_width)]
                
                remove_pad[i][j][k] = q_map.detach().cpu().numpy()

    return remove_pad

def preprocess_aperture_image(state, p1, theta, crop_size, plot=False):
    """
    Add extra padding, rotate image so the push always points to the right, crop around
    the initial push (something like attention) and finally normalize the cropped image.
    """

    # add extra padding (to handle rotations inside network)
    diag_length = float(state.shape[0]) * np.sqrt(2)
    diag_length = np.ceil(diag_length/16) * 16
    padding_width = int((diag_length - state.shape[0])/2)
    depth_heightmap = np.pad(state, padding_width, 'constant')
    padded_shape = depth_heightmap.shape

    p1 += padding_width
    action_theta = -((theta + (2 * np.pi)) % (2 * np.pi))
    
    # rotate image (push always on the right)
    rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                    action_theta * 180 / np.pi, 1.0)
    rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]))

    # compute the position of p1 on the rotated heightmap
    rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
    rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # crop heightmap
    cropped_map = np.zeros((2 * crop_size, 2 * crop_size), dtype=np.float32)
    y_start = max(0, rotated_pt[1] - crop_size)
    y_end = min(padded_shape[0], rotated_pt[1] + crop_size)
    x_start = rotated_pt[0]
    x_end = min(padded_shape[0], rotated_pt[0] + 2 * crop_size)
    cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

    # logging.info( action['opening']['min_width'])
    if plot:
        p2 = np.array([0, 0])
        p2[0] = p1[0] + 20 * np.cos(theta)
        p2[1] = p1[1] - 20 * np.sin(theta)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(depth_heightmap)
        ax[0].plot(p1[0], p1[1], 'o', 2)
        ax[0].plot(p2[0], p2[1], 'x', 2)
        ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

        rotated_p2 = np.array([0, 0])
        rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
        rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
        ax[1].imshow(rotated_heightmap)
        ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
        ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
        ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1],
                    width=1)

        ax[2].imshow(cropped_map)
        plt.show()

    # normalize maps 
    image_mean = 0.01
    image_std = 0.03
    cropped_map = (cropped_map - image_mean)/image_std
    cropped_map = np.expand_dims(cropped_map, axis=0)

    three_channel_img = np.zeros((3, cropped_map.shape[1], cropped_map.shape[2]))
    three_channel_img[0], three_channel_img[1], three_channel_img[2] = cropped_map, cropped_map, cropped_map
    
    p1 -= padding_width
    
    return three_channel_img

def preprocess_heightmap(heightmap):
    return preprocess_image(heightmap, skip_transform=True)

def preprocess_target(target, state=None):
    if state is not None:
        resized_target = resize_mask(transform, target)
        full_crop = extract_target_crop(resized_target, state)
        return preprocess_image(full_crop, skip_transform=True)[0]

    return preprocess_image(target, skip_transform=False)[0]

def preprocess_image(image, skip_transform=False):
    """
        Pre-process heightmap (padding and normalization)
    """
    
    if not skip_transform:
        image = resize_mask(transform, image)

    # add extra padding (to handle rotations inside the network)
    diagonal_length = float(image.shape[0]) * np.sqrt(2)
    diagonal_length = np.ceil(diagonal_length / 16) * 16
    padding_width = int((diagonal_length - image.shape[0]) / 2)
    padded_image = np.pad(image, padding_width, 'constant', constant_values=-0.01)
    padded_image = padded_image.astype(np.float32)

    # normalize heightmap
    image_mean = 0.01
    image_std = 0.03
    padded_image = (padded_image - image_mean)/image_std

    # add extra channel
    padded_image = np.expand_dims(padded_image, axis=0)

    return padded_image, padding_width

def get_index(index, min):
    if min:
        return index if index < 1 else index - 2
    return index if index > 98 else index + 2

def extract_target_crop(resized_target, heightmap):
        non_zero_indices = np.nonzero(resized_target)
        xmin = get_index(np.min(non_zero_indices[1]), min=True)
        xmax = get_index(np.max(non_zero_indices[1]), min=False)
        ymin = get_index(np.min(non_zero_indices[0]), min=True)
        ymax = get_index(np.max(non_zero_indices[0]), min=False)

        full_crop = np.zeros((100, 100))
        full_crop[ymin:ymax, xmin:xmax] = heightmap[ymin:ymax, xmin:xmax]

        if np.all(full_crop == 0):
            full_crop = heightmap

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(heightmap)
        # ax[1].imshow(resized_target)
        # ax[2].imshow(full_crop)

        # # bg1, bg2 = overlay_images(heightmap, resized_target)
        # # ax[3].imshow(bg1)
        # # ax[4].imshow(bg2)
        # plt.show()   
        # # plt.close('all')

        return full_crop

def overlay_images(heightmap, overlay):
    import copy

    # Deep copy
    background1 = copy.deepcopy(heightmap)
    background2 = copy.deepcopy(heightmap)

    # Overlay the images
    for i in range(100):
        for j in range(100):
            if overlay[i, j] != 0:
                background1[i, j] = overlay[i, j]
            else:
                background2[i, j] = overlay[i, j]

    return background1, background2

def apply_softmax(optimal_nodes):
    # Find the indices of non-zero elements
    non_zero_indices = np.nonzero(optimal_nodes)

    # Extract non-zero elements
    non_zero_values = optimal_nodes[non_zero_indices]

    # Apply softmax to non-zero elements
    softmax_values = np.exp(non_zero_values) / np.sum(np.exp(non_zero_values))

    # Replace non-zero elements with softmax values in the original tensor
    optimal_nodes[non_zero_indices] = softmax_values

    return np.array(optimal_nodes)

def get_pointcloud_(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[2],depth_img/camera_intrinsics[0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[5],depth_img/camera_intrinsics[4])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap_(obs, configs, bounds, pix_size):
    color_img, depth_img = obs['color'][0], obs['depth'][0]
    cam_intrinsics = configs[0]['intrinsics']

    # Compute heightmap size
    heightmap_size = np.round(((bounds[1][1] - bounds[1][0]) / pix_size,
                               (bounds[0][1] - bounds[0][0]) / pix_size)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud_(color_img, depth_img, cam_intrinsics)

    cam_pose = p_utils.get_camera_pose(configs[0]['pos'], configs[0]['target_pos'], configs[0]['up_vector'])

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= bounds[0][0], 
                                                                                      surface_pts[:,0] < bounds[0][1]), 
                                                                                      surface_pts[:,1] >= bounds[1][0]), 
                                                                                      surface_pts[:,1] < bounds[1][1]), surface_pts[:,2] < bounds[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - bounds[0][0])/pix_size).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - bounds[1][0])/pix_size).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = bounds[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    color_heightmap, depth_heightmap = cv2.flip(color_heightmap, 1), cv2.flip(depth_heightmap, 1)

    return color_heightmap, depth_heightmap

def resize_image(image, target_size=(128, 128)):#(480, 640)):
    # # Get the shape of the input image
    input_shape = image.shape

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    if len(input_shape) == 2:
        resized = cv2.cvtColor(resized.astype(np.float32), cv2.COLOR_GRAY2RGB)

    elif len(input_shape) != 3:
        raise ValueError("Unexpected image shape. Expected 2D or 3D array.")
    
    return resized