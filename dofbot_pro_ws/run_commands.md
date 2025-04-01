source ~/.bashrc
source /opt/ros/noetic/setup.bash
source ~/.bashrc

roscore

rosrun dofbot_pro_info kinemarics_dofbot_pro
rosrun dofbot_pro_info arm_driver.py
rosrun dofbot_pro_info unveiler_grasp.py 