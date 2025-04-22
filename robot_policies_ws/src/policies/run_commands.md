source ~/.bashrc
source robot_policies_ws/devel/setup.bash

catkin_make clean
catkin_make
source devel/setup.bash

rosrun policies policy_manager.py
