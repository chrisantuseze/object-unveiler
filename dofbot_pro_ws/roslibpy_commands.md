# General commands
cd ~/echris/object-unveiler
source ~/.bashrc
source /opt/ros/noetic/setup.bash
source ~/.bashrc
source dofbot_pro_ws/devel/setup.bash 

## Jetson terminal #1 
roscore

## Jetson terminal #2 - Starts the ros bridge websocket
roslaunch rosbridge_server rosbridge_websocket.launch

## Lab Computer 
python robot_policies_ws/src/policies/scripts/inference_server.py \
    --ae_model save/ae/ae_model_best.pt \
    --sre_model save/sre/sre_model_best.pt \
    --sre_rl save/sre/sre_rl_best.pt \
    --reg_model downloads/reg_model.pt

## Jetson terminal #3 - Computes (inverse) kinematics
rosrun dofbot_pro_info kinemarics_dofbot_pro

## Jetson terminal #4 - Publishes joint values to drive robot joints
rosrun dofbot_pro_info arm_driver.py

## Jetson terminal #5 - Start camera topic
roslaunch orbbec_camera dabai_dcw2.launch

## Jetson terminal #6 - Start robot control
rosrun dofbot_pro_info unveiler_grasp.py 


Ctrl + Z to stop a program from running