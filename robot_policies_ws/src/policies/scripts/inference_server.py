#!/usr/bin/env python3
"""
inference_server.py — runs on the lab computer (192.168.0.113).

Connects to rosbridge_server running on the Jetson (192.168.0.8:9090) via
roslibpy (plain WebSocket — no ROS installation required on this machine).

Flow:
  1. Jetson publishes RGB + Depth observation on  /action/obs
  2. This server receives it, decodes the images, runs the full
     exploit_unveiler_rl pipeline, and publishes the 4-DoF action back on
     /action/data
  3. Jetson receives the action, executes it on the robot arm

Run on the lab computer:
    cd /path/to/object-unveiler
    python robot_policies_ws/src/policies/scripts/inference_server.py
"""

# ── Force a non-interactive matplotlib backend BEFORE any other import that
#    could trigger a display (policy.py imports plt at module level).
import matplotlib
matplotlib.use('Agg')

import os
import sys
import base64
import argparse
import time
import copy
import threading

import numpy as np
import cv2

# ── Make sure the project root is on the path so all local modules resolve ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> policies/ -> src/ -> robot_policies_ws/ -> object-unveiler/
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml
import roslibpy

from policy.policy import Policy
from mask_rg.object_segmenter import ObjectSegmenter
from utils import general_utils
import policy.grasping as grasping

# Topic names must match what unveiler_grasp.py publishes / subscribes to.
OBS_TOPIC    = "/action/obs"
OBS_MSG_TYPE = "dofbot_pro_info/ObservData"

ACTION_TOPIC    = "/action/data"
ACTION_MSG_TYPE = "dofbot_pro_info/ActionData"

# Directory for saving debug images on the lab computer
DEBUG_DIR = os.path.join(_PROJECT_ROOT, "robot_policies_ws", "src", "policies", "scripts", "images")
os.makedirs(DEBUG_DIR, exist_ok=True)


# ── Image encode / decode helpers ─────────────────────────────────────────────

def ros_image_to_cv2(img_msg: dict) -> np.ndarray:
    """Convert a rosbridge sensor_msgs/Image dict → OpenCV numpy array.

    rosbridge may encode uint8[] `data` either as:
      - base64 string, or
      - JSON number array (list[int]).
    """
    h        = img_msg['height']
    w        = img_msg['width']
    encoding = img_msg['encoding']
    data     = img_msg['data']

    if isinstance(data, str):
        raw = base64.b64decode(data)
    elif isinstance(data, list):
        raw = bytes(data)
    else:
        raise ValueError(f"Unsupported image data type: {type(data)}")

    if encoding in ('bgr8', 'rgb8'):
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        if encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding in ('mono8', '8UC1'):
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w).copy()
    elif encoding in ('16UC1', 'mono16'):
        img = np.frombuffer(raw, dtype=np.uint16).reshape(h, w).copy()
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    return img


def cv2_to_ros_image(img: np.ndarray, encoding: str = 'mono8') -> dict:
    """Convert an OpenCV numpy array → rosbridge sensor_msgs/Image dict."""
    if img.ndim == 2:
        h, w = img.shape
        step = w * img.itemsize
    else:
        h, w, c = img.shape
        step = w * c * img.itemsize

    # For publishing through rosbridge, use uint8[] list payload to match
    # sensor_msgs/Image.data type exactly.
    data_list = np.frombuffer(img.tobytes(), dtype=np.uint8).tolist()

    return {
        'header': {
            'seq': 0,
            'stamp': {'secs': 0, 'nsecs': 0},
            'frame_id': ''
        },
        'height':       h,
        'width':        w,
        'encoding':     encoding,
        'is_bigendian': False,
        'step':         step,
        'data':         data_list,
    }


# ── Main inference server class ───────────────────────────────────────────────

class InferenceServer:
    """
    Connects to rosbridge on the Jetson, subscribes to observations, and
    publishes actions computed by the SRE-RL policy.
    """

    def __init__(self, args):
        self.args = args
        self.rng  = np.random.RandomState(args.seed)
        self._inference_lock = threading.Lock()
        # Persists the chosen target across steps within one episode so we
        # don't re-advertise a fresh target every call.
        self._episode_target_mask: np.ndarray | None = None

        # ── Load models ───────────────────────────────────────────────────
        print("Loading models…")
        with open(os.path.join(_PROJECT_ROOT, 'yaml', 'bhand.yml'), 'r') as f:
            params = yaml.safe_load(f)

        self.policy = Policy(args, params)
        self.policy.load(
            ae_model  = os.path.join(_PROJECT_ROOT, args.ae_model),
            reg_model = os.path.join(_PROJECT_ROOT, args.reg_model),
            sre_model = os.path.join(_PROJECT_ROOT, args.sre_model),
            sre_rl    = os.path.join(_PROJECT_ROOT, args.sre_rl),
        )
        self.segmenter = ObjectSegmenter(args)

        # ── Load camera extrinsics (T_cam_base) ───────────────────────────
        _extrinsics_path = os.path.join(_PROJECT_ROOT, 'yaml', 'cam_extrinsics.yaml')
        if os.path.exists(_extrinsics_path):
            with open(_extrinsics_path, 'r') as _f:
                _ext = yaml.safe_load(_f)
            self.T_cam_base = np.array(_ext['T_cam_base'], dtype=np.float64)
            print(f"Loaded T_cam_base from {_extrinsics_path}")
        else:
            self.T_cam_base = np.eye(4, dtype=np.float64)
            print(
                "[WARNING] yaml/cam_extrinsics.yaml not found — "
                "using identity T_cam_base.  Heightmap will be in camera frame; "
                "complete camera calibration (Step 1 in REAL_ROBOT_SETUP_PLAN.md) "
                "to get correct robot-frame coordinates."
            )
        print("Models loaded.")

        # ── Connect to rosbridge on Jetson ────────────────────────────────
        print(f"Connecting to rosbridge at ws://{args.jetson_ip}:{args.jetson_port} …")
        self.client = roslibpy.Ros(host=args.jetson_ip, port=args.jetson_port)
        # run() is non-blocking: it starts the WebSocket handshake in a background
        # thread and returns immediately.  Poll until the handshake completes.
        self.client.run()
        deadline = time.time() + 15.0
        while not self.client.is_connected and time.time() < deadline:
            time.sleep(0.2)

        if not self.client.is_connected:
            raise RuntimeError("Failed to connect to rosbridge on the Jetson.")
        print(f"Connected: {self.client.is_connected}")

        # ── Publisher: action → Jetson ────────────────────────────────────
        # Advertise eagerly so rosbridge registers the topic before any observation
        # arrives and before we need to publish.
        self.action_pub = roslibpy.Topic(
            self.client, ACTION_TOPIC, ACTION_MSG_TYPE
        )
        self.action_pub.advertise()

        # ── Subscriber: observation ← Jetson ──────────────────────────────
        self.obs_sub = roslibpy.Topic(
            self.client, OBS_TOPIC, OBS_MSG_TYPE
        )
        self.obs_sub.subscribe(self.on_observation)

        # Give rosbridge time to process the subscribe command and create the
        # ROS-level subscriber on the Jetson, so that get_num_connections() in
        # unveiler_grasp.py returns > 0 as soon as we print the ready message.
        print("Waiting for rosbridge to propagate subscription…")
        time.sleep(2.0)
        print(f"Subscribed to {OBS_TOPIC}. Waiting for observations…")

    # ── Callback ──────────────────────────────────────────────────────────────

    def on_observation(self, msg: dict):
        """
        Triggered each time the Jetson publishes an ObservData message.

        IMPORTANT: this callback runs on the roslibpy WebSocket thread.  Blocking
        here (e.g. running GPU inference) freezes the event loop so rosbridge
        never processes the subsequent publish() call.  Dispatch to a daemon
        thread and return immediately.
        """
        if not self._inference_lock.acquire(blocking=False):
            print("[InferenceServer] Busy with previous frame — dropping observation.")
            return

        print("[InferenceServer] Observation received — dispatching inference thread…")
        threading.Thread(
            target=self._run_inference,
            args=(msg,),
            daemon=True,
        ).start()

    def _run_inference(self, msg: dict):
        """
        Heavy inference pipeline — runs in a background thread so the roslibpy
        WebSocket event loop is never blocked.

        Steps:
          1. Decode color / depth / (optional) target_mask images.
          2. Segment the scene to get object masks and bounding boxes.
          3. Build the depth-map state.
          4. Run exploit_unveiler_rl → 4-DoF action.
          5. Publish ActionData back to the Jetson.
        """
        print("[InferenceServer] Running inference…")

        try:
            # ── 1. Decode images ──────────────────────────────────────────
            color_image = ros_image_to_cv2(msg['color_image'])   # BGR uint8  HxWx3
            depth_image = ros_image_to_cv2(msg['depth_image'])   # uint16     HxW

            # Normalise depth to 8-bit for visualisation (matches policy_manager.py)
            depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)

            # Metric depth in metres — used by get_dmap_real for heightmap building
            depth_m = depth_image.astype(np.float32) / 1000.0

            # Debug: log depth image statistics so we can triage all-zero state
            try:
                print(f"[InferenceServer] depth_image dtype={depth_image.dtype} "
                      f"min={depth_image.min()} max={depth_image.max()} "
                      f"nonzero={np.count_nonzero(depth_image)}")
                print(f"[InferenceServer] depth_vis dtype={depth_vis.dtype} "
                      f"min={depth_vis.min()} max={depth_vis.max()} "
                      f"nonzero={np.count_nonzero(depth_vis)}")
            except Exception:
                print("[InferenceServer] Failed to compute depth statistics")

            # Target mask is optional — Jetson may send an all-zero Image when absent
            target_mask_ros = msg.get('target_mask', None)
            target_mask = None
            if (target_mask_ros is not None
                    and target_mask_ros.get('encoding', '') != ''):
                target_mask_raw = ros_image_to_cv2(target_mask_ros)
                if not np.all(target_mask_raw == 0):
                    target_mask = target_mask_raw

            # Camera intrinsics — float64[9] comes through as a plain list
            intrinsics = np.array(msg['cam_intrinsics'], dtype=np.float64) # shape (9,)
            print(f"[InferenceServer] intrinsics: {intrinsics}")

            # Save debug images
            cv2.imwrite(os.path.join(DEBUG_DIR, "color_image_data.png"), color_image)
            cv2.imwrite(os.path.join(DEBUG_DIR, "depth_vis.png"), depth_vis)

            # ── 2. Segment scene ──────────────────────────────────────────
            processed_masks, pred_mask, _, bboxes = self.segmenter.from_maskrcnn(
                color_image, dir=DEBUG_DIR, bbox=True, dim=(480, 640)
            )

            if not processed_masks:
                print("[InferenceServer] No masks detected — skipping this frame.")
                return

            cv2.imwrite(os.path.join(DEBUG_DIR, "pred_mask.png"), pred_mask)

            # ── 3. Pick target mask ───────────────────────────────────────
            # Use episode-level target if already established; otherwise pick one
            # or match the hint sent from the Jetson.
            if self._episode_target_mask is not None:
                # Continue tracking the same target within this episode.
                target_id, target_mask = grasping.find_target(
                    processed_masks, self._episode_target_mask
                )
                if target_id == -1:
                    print("[InferenceServer] Episode target lost — re-selecting.")
                    self._episode_target_mask = None

            if self._episode_target_mask is None:
                if target_mask is not None:
                    # Jetson supplied a hint from a previous session.
                    target_id, target_mask = grasping.find_target(
                        processed_masks, target_mask
                    )
                    if target_id == -1:
                        target_mask, target_id = general_utils.get_target_mask(
                            processed_masks, color_image, self.rng
                        )
                        print(f"[InferenceServer] Hint not matched; auto-selected target ID: {target_id}")
                    else:
                        print(f"[InferenceServer] Using Jetson-supplied target ID: {target_id}")
                else:
                    target_mask, target_id = general_utils.get_target_mask(
                        processed_masks, color_image, self.rng
                    )
                    print(f"[InferenceServer] Auto-selected target ID: {target_id}")
                self._episode_target_mask = target_mask

            cv2.imwrite(os.path.join(DEBUG_DIR, "target_mask.png"), target_mask)

            # ── 4. Build state (depth-map heightmap) ──────────────────────
            # Use real T_cam_base (loaded from yaml/cam_extrinsics.yaml, or
            # identity if calibration hasn't been done yet).
            # depth_m was already computed above (uint16 mm → float32 m).
            try:
                print(f"[InferenceServer] depth_m min={depth_m.min():.4f} max={depth_m.max():.4f} nonzero={np.count_nonzero(depth_m)}")
            except Exception:
                pass

            state = self.policy.get_dmap_real(
                color_image,
                depth_m,
                intrinsics,
                self.T_cam_base,
            )
            print(f"[InferenceServer] State all-zero: {np.all(state == 0)}")
            try:
                print(f"[InferenceServer] state min={state.min()} max={state.max()} nonzero={np.count_nonzero(state)}")
            except Exception:
                pass

            # scene_mask for exploit_unveiler_rl is the raw segmentation overlay
            scene_mask = general_utils.resize_mask(pred_mask)

            # ── 5. Run exploit_unveiler_rl ────────────────────────────────
            # Returns a 4-element vector: [pixel_x, pixel_y, theta, aperture]
            action = self.policy.exploit_unveiler_rl(
                state,
                scene_mask,
                color_image,
                target_mask,
                copy.deepcopy(processed_masks),
                bboxes,
            )
            print(f"[InferenceServer] Action: {action}")

            # ── 6. Publish ActionData back to Jetson ──────────────────────
            # Do NOT include target_mask here: serialising a 480×640 mono image
            # as a JSON integer array (~307 K elements) silently exceeds
            # rosbridge's buffer and the entire message is dropped.  The lab
            # tracks the target across steps internally (_episode_target_mask).
            action_msg = {
                'values':      [float(v) for v in action],
            }
            self.action_pub.publish(roslibpy.Message(action_msg))
            print(f"[InferenceServer] Action published: {action}")

        except Exception as exc:
            import traceback
            print(f"[InferenceServer] ERROR during inference:\n{traceback.format_exc()}")
        finally:
            self._inference_lock.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset_episode(self):
        """Call between episodes to allow automatic target re-selection."""
        self._episode_target_mask = None
        print("[InferenceServer] Episode target reset.")

    def spin(self):
        """Block until the connection drops or the user interrupts."""
        try:
            while self.client.is_connected:
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[InferenceServer] Shutting down…")
        finally:
            self.obs_sub.unsubscribe()
            self.client.terminate()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SRE-RL inference server — runs on lab computer, "
                    "connects to Jetson via roslibpy/rosbridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths (relative to project root)
    parser.add_argument('--ae_model',  default='save/ae/ae_model_best.pt',   type=str)
    parser.add_argument('--sre_model', default='save/sre/sre_model_best.pt', type=str)
    parser.add_argument('--sre_rl',    default='save/sre/sre_rl_best.pt',    type=str)
    parser.add_argument('--reg_model', default='downloads/reg_model.pt',     type=str)

    # Jetson rosbridge endpoint
    parser.add_argument('--jetson_ip',   default='192.168.0.8', type=str)
    parser.add_argument('--jetson_port', default=9090,          type=int)

    # Policy hyper-params (must match training config)
    parser.add_argument('--sequence_length', default=1,   type=int)
    parser.add_argument('--patch_size',      default=64,  type=int)
    parser.add_argument('--num_patches',     default=10,  type=int,
                        help='≥ max number of objects in the scene')
    parser.add_argument('--chunk_size',      default=3,   type=int)
    parser.add_argument('--temporal_agg',    action='store_true')
    parser.add_argument('--step',            default=500, type=int)

    # Misc
    parser.add_argument('--seed',      default=16,   type=int)
    parser.add_argument('--mode',      default='rl', type=str)

    # Unused by inference but required by Policy.__init__
    parser.add_argument('--n_scenes',      default=100,  type=int)
    parser.add_argument('--object_set',    default='seen', type=str)
    parser.add_argument('--dataset_dir',   default='save/pc-ou-dataset', type=str)
    parser.add_argument('--epochs',        default=100,  type=int)
    parser.add_argument('--lr',            default=0.0001, type=float)
    parser.add_argument('--batch_size',    default=1,    type=int)
    parser.add_argument('--split_ratio',   default=0.9,  type=float)
    parser.add_argument('--momentum',      default=0.9,  type=float)
    parser.add_argument('--weight_decay',  default=1e-3, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    server = InferenceServer(args)
    server.spin()