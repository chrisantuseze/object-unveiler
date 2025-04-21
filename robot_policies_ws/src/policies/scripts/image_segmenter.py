#!/usr/bin/env python3

#ros
from mask_rg.object_segmenter import ObjectSegmenter
import rospy
import cv2
import numpy as np
import os
import torch

from policies.msg import SegmentationData
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSegmenter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('segmentation_policy')#, anonymous=True)

        self.TEST_DIR = "robot_policies_ws/src/policies/scripts/images"
        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)  # 1 Hz

        self.segmenter = ObjectSegmenter(is_real=True)

        self.segmenter_publisher = rospy.Publisher('/segmentation/data', SegmentationData, queue_size=1)
        self.image_subscriber = rospy.Subscriber("/image_data", Image, self.image_sub_callback)
        
    def convert_numpy_masks_to_ros_image_list(self, masks):
        image_msgs = []
        for m in masks:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = np.squeeze(m)  # Remove channel dim if present
            if m.ndim != 2:
                raise ValueError(f"Expected 2D mask, got shape {m.shape}")
            image_msgs.append(self.bridge.cv2_to_imgmsg((m.astype('uint8') * 255), encoding='mono8'))
        return image_msgs


    def image_sub_callback(self, image_data):
        print("Image subscriber callback triggered.")

        # Convert ROS image message to OpenCV image (NumPy array)
        image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding="bgr8")
        
        # Convert from BGR (ROS standard) to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(os.path.join(self.TEST_DIR, "received_image.png"), image)

        processed_masks, pred_mask, raw_masks, bboxes = self.segmenter.from_maskrcnn(image, dir=self.TEST_DIR, bbox=True, dim=(480, 640))

        msg = SegmentationData()
        msg.header.stamp = rospy.Time.now()

        # Predicted mask
        msg.pred_mask = self.bridge.cv2_to_imgmsg(pred_mask.astype('uint8') * 255, encoding='mono8')

        # Raw and processed masks
        msg.raw_masks = self.convert_numpy_masks_to_ros_image_list([mask for mask in raw_masks])
        msg.processed_masks = self.convert_numpy_masks_to_ros_image_list(processed_masks)

        # Bounding boxes
        msg.bbox_x1 = [int(x1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y1 = [int(y1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_x2 = [int(x2) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y2 = [int(y2) for (x1, y1, x2, y2) in bboxes]

        print("Segmentation done and now sending data to robot...")

        # Publish
        self.segmenter_publisher.publish(msg)

if __name__ == '__main__':
    try:
        image_segmenter = ImageSegmenter()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
