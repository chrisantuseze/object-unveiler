#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robot-side scene capture node.

Usage (after roscore is running):
    rosrun dofbot_pro_RGBDCam scene_capture.py \
        --out /home/jetson/echris/object-unveiler/save/real-eval-scenes \
        --count 50

Press Enter in the terminal for each scene you want to capture.
The node saves each scene image in the ReplayBuffer folder format:
    <out>/transition_000000/scene_image.png
    <out>/transition_000001/scene_image.png
    ...
so that the lab-side pipeline can locate and segment them immediately.
"""

import argparse
import os
import sys

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class SceneCapture:
    def __init__(self, out_dir: str, count: int, start_idx: int = 0):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        rospy.init_node("scene_capture", anonymous=False)
        rospy.Subscriber("/camera/color/image_raw", Image, self._color_cb)

        rospy.loginfo("SceneCapture ready. Waiting for first camera frame...")
        # Block until at least one frame has arrived before prompting
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.latest_frame is None:
            rate.sleep()
        rospy.loginfo("Camera feed active.")

        captured = 0
        idx = start_idx
        while not rospy.is_shutdown() and captured < count:
            # Determine target folder, skip already-captured ones
            folder = os.path.join(out_dir, f"transition_{idx:06d}")
            if os.path.exists(os.path.join(folder, "scene_image.png")):
                rospy.loginfo(f"  scene {idx:06d} already exists, skipping.")
                idx += 1
                continue

            try:
                input(
                    f"\n[{captured + 1}/{count}] Arrange the scene, then press Enter to capture..."
                )
            except EOFError:
                # Non-interactive mode (e.g. piped input) — just proceed
                pass

            if rospy.is_shutdown():
                break

            frame = cv2.flip(self.latest_frame.copy(), 0)  # flip vertically (upside-down fix)
            os.makedirs(folder, exist_ok=True)
            save_path = os.path.join(folder, "scene_image.png")
            cv2.imwrite(save_path, frame)
            rospy.loginfo(f"  Saved: {save_path}")

            captured += 1
            idx += 1

        rospy.loginfo(f"Done. {captured} scene(s) captured to '{out_dir}'.")

    def _color_cb(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture scene images from the robot camera one at a time."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/home/jetson/echris/object-unveiler/save/real-eval-scenes",
        help="Output directory for captured scenes.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of scenes to capture.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting transition index (useful to resume a collection session).",
    )
    # ROS appends extra args — strip them before argparse sees them
    argv = rospy.myargv(argv=sys.argv)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args()
    SceneCapture(out_dir=args.out, count=args.count, start_idx=args.start)
