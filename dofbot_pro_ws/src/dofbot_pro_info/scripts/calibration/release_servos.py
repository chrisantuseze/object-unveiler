#!/usr/bin/env python3
"""Release servo torque so robot joints can be moved by hand.

Usage:
  python3 release_servos.py
  # or
  rosrun dofbot_pro_info release_servos.py
"""
from Arm_Lib import Arm_Device
arm = Arm_Device()
arm.Arm_serial_set_torque(0)
print("Servo torque OFF — joints are now free to move by hand.")
