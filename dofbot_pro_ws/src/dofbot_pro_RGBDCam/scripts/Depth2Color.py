#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
encoding = ['16UC1', '32FC1']
class Depth_To_Color:
	def __init__(self):
		rospy.init_node("get_depth_info", anonymous=False)
		self.sub_depth = rospy.Subscriber("/camera/depth/image_raw", Image, self.topic_depth)
		self.sub_color = rospy.Subscriber("/camera/color/image_raw", Image, self.topic_color)
		self.window_name = "depth_image"
		self.depth_bridge = CvBridge()
	
	def topic_depth(self,msg):
		depth_image_orin = self.depth_bridge.imgmsg_to_cv2(msg, encoding[1])
		depth_to_color_image = cv.applyColorMap(cv.convertScaleAbs(depth_image_orin, alpha=0.65), cv.COLORMAP_JET)
		
		# Save the image as a new file
		cv.imwrite("saved_image.png", depth_to_color_image)
		cv.imshow(self.window_name, depth_to_color_image)
		cv.waitKey(1)

	def topic_color(self, msg):
		try:
			# Convert ROS Image message to OpenCV format
			rgb_image = self.depth_bridge.imgmsg_to_cv2(msg, "bgr8")  # Ensure correct encoding
			
			# Save the RGB image
			cv.imwrite("saved_rgb_image.png", rgb_image)
		except Exception as e:
			rospy.logerr(f"Error processing image: {e}")	

if __name__ == '__main__':
    depth_to_color = Depth_To_Color()
    rospy.spin()
