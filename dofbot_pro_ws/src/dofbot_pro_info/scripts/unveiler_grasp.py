#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import time
import argparse

import rospy
import cv2
import numpy as np

# Resolve project root so all top-level modules are importable.
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_operations import (compute_post_grasp_joints, compute_pre_grasp_joints,
                              sim_to_robot, load_T_cam_base, get_real_heightmap)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from dofbot_pro_info.msg import ArmJoint, ObservData, ActionData
from dofbot_pro_info.srv import *
from utils import general_utils
import utils.logger as logging
import utils.orientation as ori
from utils.orientation import Quaternion, rot_y


# ---------------------------------------------------------------------------
# Coordinate conversion (no ML models required)
# ---------------------------------------------------------------------------

def action3d_from_params(action, params):
    """Convert 4-DoF pixel action → 3-D robot pose using yaml workspace params."""
    pxl_size = params['env']['pixel_size']
    bounds   = np.array(params['env']['workspace']['bounds'])
    x    = -(pxl_size * action[0] - bounds[0][1])
    y    =   pxl_size * action[1] - bounds[1][1]
    quat = Quaternion.from_rotation_matrix(
        np.matmul(ori.rot_y(-np.pi / 2), ori.rot_x(action[2]))
    )
    return {'pos': np.array([x, y, 0.08]), 'quat': quat,
            'aperture': action[3], 'push_distance': 0.10}


# ---------------------------------------------------------------------------
# Robot controller
# ---------------------------------------------------------------------------

class PolicyRobotController:
    def __init__(self, args):
        self.args = args

        rospy.init_node('policy_robot_controller')

        self.TEST_DIR = os.path.join(_SCRIPT_DIR, 'images')
        self.TEST_EPISODES_DIR = os.path.join(self.TEST_DIR, 'episodes')
        os.makedirs(self.TEST_DIR, exist_ok=True)
        os.makedirs(self.TEST_EPISODES_DIR, exist_ok=True)

        # Arm control
        self.pub_arm       = rospy.Publisher("TargetAngle", ArmJoint, queue_size=10)
        self.ik_client     = rospy.ServiceProxy("get_kinemarics", kinemarics)
        self.home_position = [90.0, 120.0, 0.0, 0.0, 90.0, 40.0]
        self.gripper_angle = 30.0

        # Camera state
        self.bridge      = CvBridge()
        self.rgb_image   = None
        self.depth_image = None
        self.intrinsics  = None
        self.rgb_lock    = False
        self.depth_lock  = False
        self.camera_info_received = False

        # On-demand camera subscribers (created once, reused)
        self.rgb_sub         = None
        self.depth_sub       = None
        self.camera_info_sub = None

        # Communication with lab-computer inference server
        self.observation_pub = rospy.Publisher("/action/obs", ObservData, queue_size=1)
        # Subscribe immediately so the callback is registered before the first publish.
        self.action_sub      = rospy.Subscriber('/action/data', ActionData, self._action_callback)
        self.raw_color_image = None
        self.raw_depth_image = None
        self.target_mask     = None
        self.action          = None

        # Workspace params for pixel → 3-D coordinate conversion
        params_path = os.path.join(_PROJECT_ROOT, 'yaml', 'bhand.yml')
        with open(params_path, 'r') as f:
            self.params = yaml.safe_load(f)

        # Load camera-to-robot extrinsics and real workspace bounds
        self.T_cam_base  = load_T_cam_base()
        self.real_bounds = np.array(
            self.params['env']['real_workspace']['bounds'], dtype=np.float64
        )
        rospy.loginfo(f"[Jetson] T_cam_base loaded; real_bounds = {self.real_bounds.tolist()}")

        rospy.sleep(1)
        self.move_arm_to_position(self.home_position)

        # Wait until the lab-computer inference_server is subscribed to /action/obs.
        # Note: roslibpy.Ros.run() is non-blocking — the WebSocket handshake and
        # rosbridge subscription propagation can take a few seconds after the
        # inference_server prints "Connected".  Give it a minimum 3 s head-start
        # before polling, then keep polling until rosbridge has created the ROS
        # subscriber (get_num_connections > 0).
        rospy.loginfo("Waiting for inference server to connect…")
        rospy.sleep(3.0)          # minimum grace period for roslibpy / rosbridge
        timeout_wait = rospy.Time.now() + rospy.Duration(60.0)
        while self.observation_pub.get_num_connections() == 0:
            if rospy.Time.now() > timeout_wait:
                rospy.logwarn("Timed out waiting for inference server — proceeding anyway.")
                break
            rospy.sleep(0.5)
        rospy.loginfo("PolicyRobotController ready — inference server connected.")

    # ------------------------------------------------------------------
    # Subscribers / callbacks
    # ------------------------------------------------------------------

    def _ensure_camera_subscribers(self):
        if self.rgb_sub is None:
            self.rgb_sub = rospy.Subscriber(
                "/camera/color/image_raw", Image, self._rgb_callback)
        if self.depth_sub is None:
            self.depth_sub = rospy.Subscriber(
                "/camera/depth/image_raw", Image, self._depth_callback)
        if self.camera_info_sub is None:
            self.camera_info_sub = rospy.Subscriber(
                "/camera/depth/camera_info", CameraInfo, self._camera_info_callback)

    def _rgb_callback(self, msg):
        if not self.rgb_lock:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.rgb_image       = cv2.flip(img, -1)
            self.raw_color_image = msg
            cv2.imwrite(os.path.join(self.TEST_DIR, "saved_rgb.png"), self.rgb_image)
        except Exception as e:
            rospy.logerr(f"RGB callback error: {e}")
        finally:
            self.rgb_lock = False

    def _depth_callback(self, msg):
        if not self.depth_lock:
            return
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image     = cv2.flip(depth, -1)
            self.raw_depth_image = msg
            depth_vis = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(self.TEST_DIR, "saved_depth.png"), depth_vis)
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")
        finally:
            self.depth_lock = False

    def _camera_info_callback(self, msg):
        if self.camera_info_received:
            self.intrinsics = np.array(msg.K).reshape(3, 3)
            self.camera_info_received = False

    def _action_callback(self, msg):
        self.action = list(msg.values)
        rospy.loginfo(f"[Jetson] Action received: {self.action}")

    # ------------------------------------------------------------------
    # Image acquisition
    # ------------------------------------------------------------------

    def get_latest_image(self, timeout=10.0):
        """Acquire a fresh RGB + depth frame. Returns True on success."""
        self.rgb_image = self.depth_image = None
        self.raw_color_image = self.raw_depth_image = None
        self.rgb_lock = self.depth_lock = self.camera_info_received = True

        self._ensure_camera_subscribers()

        deadline = time.time() + timeout
        while (self.rgb_lock or self.depth_lock or self.camera_info_received) and time.time() < deadline:
            rospy.sleep(0.05)

        if self.rgb_image is None or self.depth_image is None:
            rospy.logwarn(f"Images not received within {timeout}s")
            return False
        if self.intrinsics is None:
            rospy.logwarn(f"Camera intrinsics not received within {timeout}s")
            return False
        return True
    
    # ------------------------------------------------------------------
    # Lab-computer communication
    # ------------------------------------------------------------------

    def call_policy_manager(self, timeout=30.0):
        """Publish current observation and block until an action is received."""
        if self.raw_color_image is None or self.raw_depth_image is None:
            rospy.logerr("[Jetson] No images to publish")
            return
        if self.intrinsics is None:
            rospy.logerr("[Jetson] Camera intrinsics not available")
            return

        obs_connections = self.observation_pub.get_num_connections()
        rospy.loginfo(f"[Jetson] /action/obs connections: {obs_connections}")
        if obs_connections == 0:
            rospy.logwarn("[Jetson] No subscribers on /action/obs; observation may be dropped.")

        obs = ObservData()
        obs.color_image    = self.raw_color_image
        obs.depth_image    = self.raw_depth_image
        obs.cam_intrinsics = self.intrinsics.flatten()
        # target_mask is not included: the lab server tracks the episode
        # target internally to avoid sending large image payloads over rosbridge.

        self.observation_pub.publish(obs)
        rospy.loginfo("[Jetson] Observation published — waiting for action…")

        # Stale images consumed; prevent re-sending on the next call
        self.raw_color_image = self.raw_depth_image = None

        deadline = time.time() + timeout
        while self.action is None and time.time() < deadline:
            rospy.sleep(0.1)

        if self.action is None:
            rospy.logwarn(f"[Jetson] No action received within {timeout}s")

    # ------------------------------------------------------------------
    # Robot execution
    # ------------------------------------------------------------------

    def real_action3d(self, action, real_bounds, pix_size):
        """
        Convert a 4-DOF pixel action [px, py, theta, aperture] to joint angles
        via the IK service using real workspace bounds.

        Returns (joint_angles list, aperture) or (None, aperture) on failure.
        """
        px, py, theta, aperture = action
        # Clamp x/y to the calibrated workspace so out-of-range pixels don't
        # produce poses outside the robot's kinematic envelope.
        x = np.clip(real_bounds[0][0] + pix_size * float(px),
                    real_bounds[0][0], real_bounds[0][1])
        y = np.clip(real_bounds[1][0] + pix_size * float(py),
                    real_bounds[1][0], real_bounds[1][1])

        # Clamp combined horizontal reach to the arm's physical limit (~22 cm).
        # The workspace corners can push sqrt(x²+y²) beyond the arm length,
        # causing the IK solver to return degenerate solutions with joint2 < 0.
        MAX_REACH = 0.1163   # metres — tune to match your robot's actual reach
        d = np.hypot(x, y)
        if d > MAX_REACH:
            scale = MAX_REACH / d
            x, y = x * scale, y * scale
            rospy.logwarn(
                f"real_action3d: clamped reach from {d:.3f} m → {MAX_REACH:.3f} m "
                f"(x={x:.3f}, y={y:.3f})"
            )

        # z = table surface (real_bounds[2][0]) + small standoff.
        # real_bounds[2][0] ≈ −0.14 m, i.e. the table is 14 cm below the
        # robot base-frame origin — this is correct and expected.
        z = float(real_bounds[2][0]) + 0.03   # ~3 cm above table surface
        rospy.loginfo(
            f"real_action3d → XYZ: ({x:.3f}, {y:.3f}, {z:.3f})  "
            f"theta={np.degrees(float(theta)):.1f}°"
        )
        req = kinemaricsRequest()
        req.kin_name = "ik"
        req.tar_x    = x
        req.tar_y    = y
        req.tar_z    = z
        req.Roll     = float(theta)
        try:
            self.ik_client.wait_for_service(timeout=3.0)
            resp = self.ik_client.call(req)
            joints = [
                resp.joint1, resp.joint2, resp.joint3,
                min(resp.joint4, 90.0),
                90.0,   # palm fixed
                30.0,   # gripper — overridden by gripper_control
            ]
            if any(j < 0 or j > 180 for j in joints[:4]):
                rospy.logwarn(f"IK result out of range: {joints}")
                return None, aperture
            return joints, aperture
        except rospy.ServiceException as e:
            rospy.logerr(f"IK service failed: {e}")
            return None, aperture

    def grasp_object(self, action_4dof):
        """Execute grasp from raw 4-DOF policy action [px, py, theta, aperture]."""
        pix_size = self.params['env']['pixel_size']
        joint_angles, aperture = self.real_action3d(
            action_4dof, self.real_bounds, pix_size
        )
        if joint_angles is None:
            rospy.logerr("IK failed — skipping grasp step")
            general_utils.delete_episodes_misc(self.TEST_EPISODES_DIR)
            return
        try:
            self.gripper_control(aperture)
            rospy.sleep(0.5)
            self.execute_grasp_sequence(joint_angles)
            rospy.sleep(2)
        except Exception as e:
            rospy.logerr(f"Grasp error: {e}")
        general_utils.delete_episodes_misc(self.TEST_EPISODES_DIR)

    def grasp_object_from_pose(self, action_dict):
        """Legacy: Set gripper aperture, solve IK from 3-D pos dict, execute grasp."""
        try:
            self.gripper_control(action_dict['aperture'])
            rospy.sleep(0.5)
            joint_angles = self.get_joint_angles_from_pose(action_dict['pos'])
            if joint_angles is not None:
                self.execute_grasp_sequence(joint_angles)
            else:
                rospy.logerr("IK failed — skipping grasp")
            rospy.sleep(2)
        except Exception as e:
            rospy.logerr(f"Grasp error: {e}")
        general_utils.delete_episodes_misc(self.TEST_EPISODES_DIR)

    def get_joint_angles_from_pose(self, pos):
        """Call IK service and return joint angles, or None on failure."""
        x, y, z = sim_to_robot(pos)
        req = kinemaricsRequest()
        req.tar_x, req.tar_y, req.tar_z = x, y, z
        req.kin_name = "ik"
        try:
            resp = self.ik_client.call(req)
            angles = [resp.joint1, resp.joint2, resp.joint3, resp.joint4]
            if any(a < 0 or a > 180 for a in angles):
                rospy.logwarn("IK returned out-of-range joint angles")
                return None
            return angles + [90, 30]
        except rospy.ServiceException as e:
            rospy.logerr(f"IK service failed: {e}")
            return None

    def execute_grasp_sequence(self, joint_positions):
        """Pre-grasp → grasp → close gripper → lift → release → home."""
        self.move_arm_to_position(compute_pre_grasp_joints(joint_positions))
        rospy.sleep(3)
        self.move_arm_to_position(joint_positions)
        rospy.sleep(3)
        self.gripper_control(1)   # close
        rospy.sleep(2)
        self.move_arm_to_position(compute_post_grasp_joints(joint_positions))
        rospy.sleep(3)
        pre_home = self.home_position[::]
        pre_home[0], pre_home[1], pre_home[2] = 180.0, 80.0, 20.0
        self.move_arm_to_position(pre_home)
        rospy.sleep(3)
        self.gripper_control(0)   # open / release
        rospy.sleep(3)
        self.move_arm_to_position(self.home_position)
        rospy.sleep(3)

    def move_arm_to_position(self, joint_positions, run_time=2000):
        joint_positions[5] = self.gripper_angle
        msg = ArmJoint()
        msg.joints   = joint_positions
        msg.run_time = run_time
        self.pub_arm.publish(msg)
        rospy.loginfo(f"Arm target: {joint_positions}")

    def gripper_control(self, aperture, run_time=1000):
        self.gripper_angle = np.interp(aperture, [0, 1], [30, 140])
        msg = ArmJoint()
        msg.id       = 6
        msg.angle    = self.gripper_angle
        msg.run_time = run_time
        msg.joints   = []
        self.pub_arm.publish(msg)
            
    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run(self):
        """One episode: capture → send to lab → receive action → execute → repeat."""
        if not self.get_latest_image():
            rospy.logerr("Failed to acquire initial images")
            return

        for step in range(1, 7):  # max 6 steps
            self.action = None
            self.call_policy_manager(timeout=30.0)

            if self.action is None:
                rospy.logerr("[Jetson] No action received — aborting episode.")
                break

            action_arr = np.array(self.action, dtype=np.float32)
            rospy.loginfo(
                f"Step {step}: px={action_arr[0]:.1f}, py={action_arr[1]:.1f}, "
                f"theta={np.degrees(action_arr[2]):.1f}°, aperture={action_arr[3]:.3f}"
            )
            self.grasp_object(action_arr)

            if self.rgb_image is not None:
                cv2.imwrite(os.path.join(self.TEST_DIR, f'color_step{step}.png'), self.rgb_image)

            if input("\nContinue? (y/n): ").strip().lower() == 'n':
                break

            # Re-acquire fresh images for the next observation
            if not self.get_latest_image():
                rospy.logwarn("Could not re-acquire images after grasp.")

        rospy.loginfo(f"Episode finished after {step} steps.")

    def eval_agent(self):
        for i in range(self.args.n_scenes):
            logging.info(f'Episode: {i}')
            self.run()

    def cleanup(self):
        for sub in [self.rgb_sub, self.depth_sub, self.camera_info_sub, self.action_sub]:
            if sub is not None:
                sub.unregister()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed',     default=16,  type=int, help='Random seed')
    parser.add_argument('--n_scenes', default=100, type=int, help='Number of episodes to run')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    controller = PolicyRobotController(args)
    try:
        controller.eval_agent()
    finally:
        controller.cleanup()