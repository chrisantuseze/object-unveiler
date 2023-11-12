import math
import numpy as np
import pybullet as p

import utilities.pybullet_utils as p_utils
import utilities.general_utils as general_utils
from utils.constants import *
"""Camera configs."""


class RealSense():
  """Default configuration with 2 RealSense RGB-D cameras."""

  image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

  # Set default camera poses. (w.r.t. workspace center)
  front_position = np.array([0.0, 0.5, 0.5])
  front_target_pos = np.array([0.0, 0.0, 0.0])
  front_up_vector = np.array([0.0, 0.0, 1.0])

  top_position = np.array([0.0, 0.0, 0.5])
  top_target_pos = np.array([0.0, 0.0, 0.0])
  top_up_vector = np.array([0.0, -1.0, 0.0])


  # Default camera configs.
  CONFIG = [
    {
      'type': 'front',
      'image_size': (480, 640),
      'intrinsics': intrinsics,
      'pos': front_position,
      'target_pos': front_target_pos,
      'up_vector': front_up_vector,
      'zrange': (0.01, 10.),
      'noise': False,
    }, 
    {
        'type': 'top',
        'image_size': image_size,
        'intrinsics': intrinsics,
        'pos': top_position,
        'target_pos': top_target_pos,
        'up_vector': top_up_vector,
        'zrange': (0.01, 10.),
        'noise': False,
    }
  ]


class SimCamera:
    def __init__(self, config) -> None:
        self.config = config
        self.pos = np.array(config['pos'])
        self.target_pos = np.array(config['target_pos'])
        self.up_vector = np.array(config['up_vector'])

        self.type = config['type']

        self.view_matrix = p.computeViewMatrix(cameraEyePosition=self.pos,
                                                cameraTargetPosition=self.target_pos,
                                                cameraUpVector=self.up_vector)
        
        self.z_near = config['zrange'][0]
        self.z_far = config['zrange'][1]
        self.width, self.height = config['image_size'][1], config['image_size'][0]
        self.fy = config['intrinsics'][0]

        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width/self.height, 
                                                                nearVal=self.z_near, farVal=self.z_far)
        
    def get_pose(self):
        """
        Returns the camera pose w.r.t. world

        Returns
        -------
        np.array()
            4x4 matrix representing the camera pose w.r.t. world
        """
        return p_utils.get_camera_pose(self.pos, self.target_pos, self.up_vector)

    def get_depth(self, depth_buffer):
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth

    def get_data(self):
        """
        Returns
        -------
        np.array(), np.array(), np.array()
            The rgb, depth and segmentation images
        """
        _, _, color, depth, segm = p.getCameraImage(self.width, self.height,
                                    self.view_matrix, self.projection_matrix,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                )
        
        # Get color image.
        color_image_size = (self.config["image_size"][0], self.config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if self.config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, self.config["image_size"]))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (self.config["image_size"][0], self.config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = self.z_far + self.z_near - (2.0 * zbuffer - 1.0) * (self.z_far - self.z_near)
        depth = (2.0 * self.z_near * self.z_far) / depth
        if self.config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm
        
        # return color, self.get_depth(depth), segm

