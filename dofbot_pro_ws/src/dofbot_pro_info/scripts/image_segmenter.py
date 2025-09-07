#ros
from mask_rg.object_segmenter import ObjectSegmenter
import rospy
import cv2
import numpy as np

from dofbot_pro_info.msg import SegmentationData, Image_Msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSegmenter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('segmentation_publisher', anonymous=True)

        self.publisher = rospy.Publisher('/segmentation/mask', SegmentationData, queue_size=10)
        self.image_subscriber = rospy.Subscriber("/image_data", Image_Msg, self.image_sub_callback)
        
        self.bridge = CvBridge()
        self.rate = rospy.Rate(1)  # 1 Hz

        self.segmenter = ObjectSegmenter(is_real=True)

    def convert_numpy_masks_to_ros_image_list(self, masks):
        return [self.bridge.cv2_to_imgmsg(m.astype('uint8') * 255, encoding='mono8') for m in masks]

    def image_sub_callback(self, image_data):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        np_image = np.ndarray(shape=(image_data.height, image_data.width, image_data.channels), dtype=np.uint8, buffer=image_data.data)
        image[:,:,0], image[:,:,1], image[:,:,2] = np_image[:,:,2], np_image[:,:,1], np_image[:,:,0] #rgb

        processed_masks, pred_mask, raw_masks, bboxes = self.segmenter.from_maskrcnn(image, dir="/", bbox=True, dim=(480, 640))

        msg = SegmentationData()
        msg.header.stamp = rospy.Time.now()

        # Predicted mask
        msg.pred_mask = self.bridge.cv2_to_imgmsg(pred_mask.astype('uint8') * 255, encoding='mono8')

        # Raw and processed masks
        msg.raw_masks = self.convert_numpy_masks_to_ros_image_list(raw_masks)
        msg.processed_masks = self.convert_numpy_masks_to_ros_image_list(processed_masks)

        # Bounding boxes
        msg.bbox_x1 = [int(x1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y1 = [int(y1) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_x2 = [int(x2) for (x1, y1, x2, y2) in bboxes]
        msg.bbox_y2 = [int(y2) for (x1, y1, x2, y2) in bboxes]

        print("Segmentation done and now sending data to robot...")

        # Publish
        self.publisher.publish(msg)

if __name__ == '__main__':
    try:
        image_segmenter = ImageSegmenter()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
