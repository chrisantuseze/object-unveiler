#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import yaml
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

# Adjust paths: compute repo root (object-unveiler) and point to dofbot_pro_ws/src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
BHAND_PATH = os.path.join(REPO_ROOT, 'yaml', 'bhand.yml')

rospy.init_node('heightmap_smoke_test', anonymous=True)
bridge = CvBridge()
print('Waiting for color, depth, camera_info topics...')
color_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10.0)
depth_msg = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=10.0)
info_msg  = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=10.0)
print('Messages received')

color = bridge.imgmsg_to_cv2(color_msg, 'bgr8')
depth = bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
K = np.array(info_msg.K).reshape(3, 3)

# load functions
import sys
import importlib.util
# Import robot_operations.py directly (package not installed as a Python package)
robot_ops_path = os.path.join(REPO_ROOT, 'dofbot_pro_ws', 'src', 'dofbot_pro_info', 'scripts', 'robot_operations.py')
spec = importlib.util.spec_from_file_location('robot_operations', robot_ops_path)
robot_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_ops)
get_real_heightmap = robot_ops.get_real_heightmap
load_T_cam_base = robot_ops.load_T_cam_base

T = load_T_cam_base()
with open(BHAND_PATH, 'r') as f:
    params = yaml.safe_load(f)
real_bounds = np.array(params['env']['real_workspace']['bounds'], dtype=np.float64)
pxl_size = float(params['env']['pixel_size'])

print('Computing heightmap...')
hmap = get_real_heightmap(color, depth, K, T, real_bounds, pxl_size)

out = '/tmp/hmap.png'
cv2.imwrite(out, cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
print('Saved', out, 'max=', float(np.nanmax(hmap)))
