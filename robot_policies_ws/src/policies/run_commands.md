source ~/.bashrc
source devel/setup.bash

catkin_make clean
catkin_make
source devel/setup.bash

rosrun policies image_segmenter.py