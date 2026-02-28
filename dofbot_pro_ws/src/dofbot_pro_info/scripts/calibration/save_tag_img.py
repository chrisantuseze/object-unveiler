#!/usr/bin/env python3

import rospy, cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
rospy.init_node('save_tag_img', anonymous=True)
bridge = CvBridge()
img = rospy.wait_for_message('/unveiler/tag_detections_image', Image, timeout=5.0)
cv2.imwrite('tag_detections_image.png', bridge.imgmsg_to_cv2(img, 'bgr8'))
print('Wrote tag_detections_image.png')
