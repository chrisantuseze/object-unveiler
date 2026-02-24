# Jetson terminal #1
cd ~/echris/object-unveiler
source ~/.bashrc
source /opt/ros/noetic/setup.bash
source ~/.bashrc
source dofbot_pro_ws/devel/setup.bash 

# start the driver (leave this terminal open)
rosrun dofbot_pro_info arm_driver.py
clear




roscore

rosrun dofbot_pro_info kinemarics_dofbot_pro
rosrun dofbot_pro_info arm_driver.py
rosrun dofbot_pro_info unveiler_grasp.py 
rosrun dofbot_pro_info lean_control.py 
rosrun dofbot_pro_info scene_image_collector.py 


roslaunch orbbec_camera dabai_dcw2.launch
rosrun dofbot_pro_RGBDCam rgbd_pointcloud.py

rosrun dofbot_pro_RGBDCam Depth2Color.py


catkin_make # navigate to the dofbot_pro_ws and run that to rebuild projects


watch -n 1 "free -h && sudo tegrastats"

Ctrl + Z to stop a program from running


source dofbot_pro_ws/devel/setup.bash 