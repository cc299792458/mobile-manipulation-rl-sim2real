from utils import *

if __name__ == '__main__':
    dynamixel_arm = DynamixelArm()
    present_position = dynamixel_arm.get_position(if_print=True)
    present_vel_i_gain = dynamixel_arm.get_vel_i_gain(if_print=True)
    present_vel_p_gain = dynamixel_arm.get_vel_p_gain(if_print=True)
    while True:
        # dynamixel_arm.set_position(goal_position=[1800, 2047-180, 2047-291, 2047+472], id=[11, 13, 14, 2])
        # dynamixel_arm.set_position(goal_position=[2300], id=[11])
        # dynamixel_arm.set_position(goal_position=[1700], id=[11])
        dynamixel_arm.set_ee_pos(ee_pos=[0, 0.3, 0.1, 0, 0.03])
        # dynamixel_arm.set_ee_pos(ee_pos=[0, 0.3, 0.2, 0, 0])
        # dynamixel_arm.set_position(goal_position=[1600], id=[15])
        # dynamixel_arm.set_position(goal_position=[600], id=[15])
        # dynamixel_arm.set_position(goal_position=[1800], id=[11])
        # dynamixel_arm.set_position(goal_position=[2200], id=[11])
        # dynamixel_arm.get_current(if_print=True)
        # dynamixel_arm.get_position(if_print=True)
        # dynamixel_arm.set_position(goal_position=[1800], id=[13])

        # dynamixel_arm.set_position(goal_position=[2047-132], id=[13])
        # dynamixel_arm.set_position(goal_position=[2047-219], id=[14])
        # dynamixel_arm.set_position(goal_position=[2047+351], id=[2])