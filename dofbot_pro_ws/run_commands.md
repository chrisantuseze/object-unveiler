cd echris/object-unveiler
source ~/.bashrc
source /opt/ros/noetic/setup.bash
source ~/.bashrc
clear

roscore

rosrun dofbot_pro_info kinemarics_dofbot_pro
rosrun dofbot_pro_info arm_driver.py
rosrun dofbot_pro_info unveiler_grasp.py 


roslaunch orbbec_camera dabai_dcw2.launch
rosrun dofbot_pro_RGBDCam rgbd_pointcloud.py

rosrun dofbot_pro_RGBDCam Depth2Color.py


catkin_make # navigate to the dofbot_pro_ws and run that to rebuild projects


watch -n 1 "free -h && sudo tegrastats"

Ctrl + Z to stop a program from running


source devel/setup.bash 