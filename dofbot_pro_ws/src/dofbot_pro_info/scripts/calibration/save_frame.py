#!/usr/bin/env python3

import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
rospy.init_node('save_frame', anonymous=True)
bridge = CvBridge()
img = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=5.0)
cv2.imwrite('camera_color.png', bridge.imgmsg_to_cv2(img, 'bgr8'))
print('Wrote camera_color.png')
