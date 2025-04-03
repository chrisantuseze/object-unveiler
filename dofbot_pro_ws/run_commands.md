source ~/.bashrc
source /opt/ros/noetic/setup.bash
source ~/.bashrc

roscore

rosrun dofbot_pro_info kinemarics_dofbot_pro
rosrun dofbot_pro_info arm_driver.py
rosrun dofbot_pro_info unveiler_grasp.py 



roslaunch orbbec_camera dabai_dcw2.launch
rosrun dofbot_pro_RGBDCam Depth2Color.py

rosrun dofbot_pro_RGBDCam rgbd_pointcloud.py