import os
import time
import numpy as np
from scipy.optimize import fsolve

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
def clamp(value, min_value, max_value):
    return [max(min_value, min(v, max_value)) for v in value]

from dynamixel_sdk import *                    # Uses Dynamixel SDK library

DEVICENAME                  = '/dev/ttyACM0'
BAUDRATE                    = 1000000
PROTOCOL_VERSION            = 2.0

ADDR_OPERATING_MODE         = 11
ADDR_TORQUE_ENABLE          = 64
ADDR_VELOCITY_I_GAIN        = 76
ADDR_VELOCITY_P_GAIN        = 78
ADDR_GOAL_VELOCITY          = 104
ADDR_GOAL_POSITION          = 116
ADDR_PRESENT_CURRENT        = 126   
ADDR_PRESENT_VELOCITY       = 128
ADDR_PRESENT_POSITION       = 132

DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual

VELOCITY_MODE               = 1
POSITION_MODE               = 3 

TORQUE_ENABLE               = 1     # Value for enabling the torque
TORQUE_DISABLE              = 0     # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold

LEN_VEL_I_Gain              = 2
LEN_VEL_P_Gain              = 2
LEN_PRESENT_CURRENT         = 2
LEN_GOAL_VELOCITY           = 4
LEN_PRESENT_POSITION        = 4
LEN_PRESENT_VELOCITY        = 4

class DynamixelArm:
    def __init__(self, id=[11, 13, 14, 2, 15], 
                 initial_position=[2047, 2559, 2047, 1535, 1023],   # 180, 225, 180, 135, 90 
                 control_freq=20) -> None:
        self.id = id
        self.num = len(id)
        self.initial_position = initial_position
        self.min_position_limit = 0
        self.max_position_limit = 4095
        self.position_threshold = 3
        self.velocity_limit = 200
        self.min_velocity_const = 1
        self.max_velocity_const = 22    # approximately 30 degree/s
        self.interpolate_freq = 100
        self.control_freq = control_freq
        self.current_limit = 200
        
        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        self.portHandler = PortHandler(DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Initialize GroupSyncWrite instance for Goal Velocity
        self.velocitySyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY)

        self.velIGainSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_VELOCITY_I_GAIN, LEN_VEL_I_Gain)
        self.velPGainSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, ADDR_VELOCITY_P_GAIN, LEN_VEL_P_Gain)

        # Initialize GroupSyncRead instace for Present Position and Present Velocity
        self.positionSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        self.velocitySyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)

        self.currentSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)

        self.velIGainSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_VELOCITY_I_GAIN, LEN_VEL_I_Gain)
        self.velPGainSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, ADDR_VELOCITY_P_GAIN, LEN_VEL_P_Gain)

        self.presentPosition = [0 for _ in range(self.num)]
        self.presentVelocity = [0 for _ in range(self.num)]

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()

        for i in range(self.num):
            # Disable Torque
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id[i], ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                break
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                break
            # else:
            #     print("Dynamixel#%d has been successfully connected" % id[i])
            
            # Switch to Velocity Mode
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id[i], ADDR_OPERATING_MODE, VELOCITY_MODE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                break
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                break
            # else:
            #     print("Dynamixel#%d has been successfully switched to velocity mode" % id[i])
            
            # Enable Torque
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id[i], ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                break
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
                break
            # else:
            #     print("Dynamixel#%d has been successfully connected" % id[i])
            
            # Setup SyncRead
            dxl_addparam_result = self.positionSyncRead.addParam(id[i])
            if dxl_addparam_result != True:
                print("[ID:%03d] positionSyncRead addparam failed" % id[i])
                quit()
            dxl_addparam_result = self.velocitySyncRead.addParam(id[i])
            if dxl_addparam_result != True:
                print("[ID:%03d] velocitySyncRead addparam failed" % id[i])
                quit()
            dxl_addparam_result = self.currentSyncRead.addParam(id[i])
            if dxl_addparam_result != True:
                print("[ID:%03d] currentSyncRead addparam failed" % id[i])
                quit()
            dxl_addparam_result = self.velIGainSyncRead.addParam(id[i])
            if dxl_addparam_result != True:
                print("[ID:%03d] velIGainSyncRead addparam failed" % id[i])
                quit()
            dxl_addparam_result = self.velPGainSyncRead.addParam(id[i])
            if dxl_addparam_result != True:
                print("[ID:%03d] velPGainSyncRead addparam failed" % id[i])
                quit()
            
        print("All Dynamixel are succesfully initialized.")
        # print("Current Control Mode: Control Position via control velocity")

        self.set_vel_i_gain(vel_i_gain=[1920, 500, 700, 700, 1000])
        self.set_vel_p_gain(vel_p_gain=[400, 150, 200, 200, 100])
        self.initialize_position()

    def get_position(self, if_print=False, id=None):
        # Syncread present position
        dxl_comm_result = self.positionSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for i in range(self.num):
            # Check if groupsyncread data of Dynamixel i is available
            dxl_getdata_result = self.positionSyncRead.isAvailable(self.id[i], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.id[i])
                quit()
        
        for i in range(self.num):
            dxl_present_position = self.positionSyncRead.getData(self.id[i], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if dxl_present_position > 2**16:    # velocity is represented as an unsigned int, this is to re-represent the negtive velocity.
                dxl_present_position -= 2**32
            self.presentPosition[i] = dxl_present_position
            if if_print:
                print(f"Present position for dynamixel {self.id[i]} is {dxl_present_position}")

        if id != None:
            present_position = [self.presentPosition[j] for j in [self.id.index(item) for item in id]]
            return present_position

        return self.presentPosition

    def get_velocity(self, if_print=False, id=None):
        # Syncread present velocity
        dxl_comm_result = self.velocitySyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for i in range(self.num):
            # Check if groupsyncread data of Dynamixel i is available
            dxl_getdata_result = self.velocitySyncRead.isAvailable(self.id[i], ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.id[i])
                quit()
        
        for i in range(self.num):
            dxl_present_velocity = self.velocitySyncRead.getData(self.id[i], ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
            if dxl_present_velocity > 2**16:    # velocity is represented as an unsigned int, this is to re-represent the negtive velocity.
                dxl_present_velocity -= 2**32
            self.presentVelocity[i] = dxl_present_velocity
            if if_print:
                print(f"Present velocity for dynamixel {self.id[i]} is {dxl_present_velocity}")
        
        if id != None:
            present_velocity = [self.presentVelocity[j] for j in [self.id.index(item) for item in id]]
            return present_velocity

        return self.presentVelocity
    
    def get_current(self, if_print=False, id=None):
        # Syncread present current
        dxl_comm_result = self.currentSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for i in range(self.num):
            # Check if groupsyncread data of Dynamixel i is available
            dxl_getdata_result = self.currentSyncRead.isAvailable(self.id[i], ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.id[i])
                quit()
        
        presentCurrent = [0 for _ in range(self.num)]
        for i in range(self.num):
            dxl_present_current = self.currentSyncRead.getData(self.id[i], ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
            if dxl_present_current > 2**8:    # current is represented as an unsigned int, this is to re-represent the negtive current.
                dxl_present_current = 2**16 - dxl_present_current
            presentCurrent[i] = dxl_present_current
            if if_print:
                print(f"Present current for dynamixel {self.id[i]} is {dxl_present_current}")

        if id != None:
            present_current = [presentCurrent[j] for j in [self.id.index(item) for item in id]]
            return present_current

        return presentCurrent

    def get_vel_i_gain(self, if_print=False, id=None):
        # Syncread present velocity
        dxl_comm_result = self.velIGainSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for i in range(self.num):
            # Check if groupsyncread data of Dynamixel i is available
            dxl_getdata_result = self.velIGainSyncRead.isAvailable(self.id[i], ADDR_VELOCITY_I_GAIN, LEN_VEL_I_Gain)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.id[i])
                quit()
        
        presentVelIGain = [0 for _ in range(self.num)]
        for i in range(self.num):
            dxl_present_vel_i_gain = self.velIGainSyncRead.getData(self.id[i], ADDR_VELOCITY_I_GAIN, LEN_VEL_I_Gain)
            presentVelIGain[i] = dxl_present_vel_i_gain
            if if_print:
                print(f"Present velocity I Gain for dynamixel {self.id[i]} is {dxl_present_vel_i_gain}")

        return presentVelIGain
    
    def get_vel_p_gain(self, if_print=False, id=None):
        # Syncread present velocity
        dxl_comm_result = self.velPGainSyncRead.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        for i in range(self.num):
            # Check if groupsyncread data of Dynamixel i is available
            dxl_getdata_result = self.velPGainSyncRead.isAvailable(self.id[i], ADDR_VELOCITY_P_GAIN, LEN_VEL_P_Gain)
            if dxl_getdata_result != True:
                print("[ID:%03d] groupSyncRead getdata failed" % self.id[i])
                quit()
        
        presentVelPGain = [0 for _ in range(self.num)]
        for i in range(self.num):
            dxl_present_vel_p_gain = self.velPGainSyncRead.getData(self.id[i], ADDR_VELOCITY_P_GAIN, LEN_VEL_P_Gain)
            presentVelPGain[i] = dxl_present_vel_p_gain
            if if_print:
                print(f"Present velocity P Gain for dynamixel {self.id[i]} is {dxl_present_vel_p_gain}")

        return presentVelPGain

    def set_velocity(self, goal_velocity: list, id=None):
        assert max(goal_velocity) <= self.velocity_limit
        if id == None:
            assert len(goal_velocity) == self.num
        else:
            assert len(goal_velocity) == len(id)
            for i in range(self.num):
                if self.id[i] not in id:
                    goal_velocity.insert(i, 0)

        for i in range(self.num):
            # Allocate goal velocity value into byte array
            param_goal_velocity = [DXL_LOBYTE(DXL_LOWORD(goal_velocity[i])), DXL_HIBYTE(DXL_LOWORD(goal_velocity[i])), DXL_LOBYTE(DXL_HIWORD(goal_velocity[i])), DXL_HIBYTE(DXL_HIWORD(goal_velocity[i]))]

            # Add Dynamixel goal velocity value to the Syncwrite parameter storage
            dxl_addparam_result = self.velocitySyncWrite.addParam(self.id[i], param_goal_velocity)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % self.id[i])
                quit()

        # Syncwrite goal velocity
        dxl_comm_result = self.velocitySyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self.velocitySyncWrite.clearParam()

    def set_position(self, goal_position: list, id=None):
        assert min(goal_position) >= self.min_position_limit and max(goal_position) <= self.max_position_limit
        if id == None:
            assert len(goal_position) == self.num
        else:
            assert len(goal_position) == len(id)
        present_position = self.get_position(id=id)
        gripper_current = self.get_current(id=[self.id[-1]])
        if id==None:
            if gripper_current[0] > self.current_limit:
                goal_position[-1] = present_position[-1]
        elif self.id[-1] in id:
            if gripper_current[0] > self.current_limit:
                goal_position[id.index(self.id[-1])] = present_position[id.index(self.id[-1])]
        delta_position = [goal_position[i] - present_position[i] for i in range(len(goal_position))]
        sign = [1 if x > 0 else -1 for x in delta_position]
        value = [abs(x) for x in delta_position]
        while max(value) > self.position_threshold:
            # if id == self.id[-1]:
            #     pass
            value = clamp(value, self.min_velocity_const, self.max_velocity_const)
            goal_velocity = [sign[i] * value[i] for i in range(len(value))]
            cur_time = start_time = time.time()
            if id == None:
                self.set_velocity(goal_velocity=goal_velocity)
            else:
                self.set_velocity(goal_velocity=goal_velocity, id=id)
            # end_time = time.time()
            # print(end_time - start_time)  # about 3e-4 second.
            while cur_time - start_time < 1 / self.interpolate_freq:
                time.sleep(0.0001)
                cur_time = time.time()
            present_position = self.get_position(id=id) 
            gripper_current = self.get_current(id=[self.id[-1]])
            if id==None:
                if gripper_current[0] > self.current_limit:
                    goal_position[-1] = present_position[-1]
            elif self.id[-1] in id:
                if gripper_current[0] > self.current_limit:
                    goal_position[id.index(self.id[-1])] = present_position[id.index(self.id[-1])]
            delta_position = [goal_position[i] - present_position[i] for i in range(len(goal_position))]
            sign = [1 if x > 0 else -1 for x in delta_position]
            value = [abs(x) for x in delta_position]
            # print(present_position)
            # self.get_velocity(if_print=True)
            # self.get_current(if_print=True)
        self.set_velocity([0 for _ in range(self.num)])

    def set_vel_i_gain(self, vel_i_gain: list, id=None):
        if id == None:
            assert len(vel_i_gain) == self.num
        else:
            assert len(vel_i_gain) == len(id)
            for i in range(self.num):
                if self.id[i] not in id:
                    vel_i_gain.insert(i, 0)

        for i in range(self.num):
            # Allocate velocity i gain into byte array
            # param_vel_i_gain = [DXL_LOBYTE(DXL_LOWORD(vel_i_gain[i])), DXL_HIBYTE(DXL_LOWORD(vel_i_gain[i])), DXL_LOBYTE(DXL_HIWORD(vel_i_gain[i])), DXL_HIBYTE(DXL_HIWORD(vel_i_gain[i]))]
            param_vel_i_gain = [DXL_LOBYTE(vel_i_gain[i]), DXL_HIBYTE(vel_i_gain[i])]

            # Add Dynamixel velocity i gain to the Syncwrite parameter storage
            dxl_addparam_result = self.velIGainSyncWrite.addParam(self.id[i], param_vel_i_gain)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % self.id[i])
                quit()

        # Syncwrite goal velocity
        dxl_comm_result = self.velIGainSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self.velIGainSyncWrite.clearParam()

    def set_vel_p_gain(self, vel_p_gain: list, id=None):
        if id == None:
            assert len(vel_p_gain) == self.num
        else:
            assert len(vel_p_gain) == len(id)
            for i in range(self.num):
                if self.id[i] not in id:
                    vel_p_gain.insert(i, 0)

        for i in range(self.num):
            # Allocate velocity i gain into byte array
            # param_vel_i_gain = [DXL_LOBYTE(DXL_LOWORD(vel_i_gain[i])), DXL_HIBYTE(DXL_LOWORD(vel_i_gain[i])), DXL_LOBYTE(DXL_HIWORD(vel_i_gain[i])), DXL_HIBYTE(DXL_HIWORD(vel_i_gain[i]))]
            param_vel_p_gain = [DXL_LOBYTE(vel_p_gain[i]), DXL_HIBYTE(vel_p_gain[i])]

            # Add Dynamixel velocity i gain to the Syncwrite parameter storage
            dxl_addparam_result = self.velPGainSyncWrite.addParam(self.id[i], param_vel_p_gain)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % self.id[i])
                quit()

        # Syncwrite goal velocity
        dxl_comm_result = self.velPGainSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        self.velPGainSyncWrite.clearParam()

    def set_ee_pos(self, ee_pos):
        """
            alpha_target and beta_target are in radian,
            r_target, z_target, and gripper_target are in meter,
        """
        alpha_target, r_target, z_target, beta_target, gripper_target = ee_pos
        goal_position = self._compensite4origin(self._rad2int32([alpha_target]))+\
                        self._compensite4origin(self.inverse_kinematics(r_target, z_target, beta_target))+\
                        self._meter2int32(gripper_target)
        self.set_position(goal_position)

    def set_joint_qpos(self, joint_qpos):
        pass

    def initialize_position(self):
        assert self.id[0:4] == [11, 13, 14, 2]  # NOTE: Hard-code with current arm.
        # Phase 1
        present_position = self.get_position(id=[13])
        goal_position = [int((self.initial_position[self.id.index(13)] + present_position[0]) / 2)]
        self.set_position(goal_position=goal_position, id=[13])
        # Phase 2
        present_position = self.get_position(id=[14])
        goal_position = [self.initial_position[self.id.index(13)], 
                         int((self.initial_position[self.id.index(14)] + present_position[0]) / 2)]
        self.set_position(goal_position=goal_position, id=[13, 14])
        # Phase 3
        present_position = self.get_position(id=[2])
        goal_position = [self.initial_position[self.id.index(13)],
                          self.initial_position[self.id.index(14)],
                         int((self.initial_position[self.id.index(2)] + present_position[0]) / 2)]
        self.set_position(goal_position=goal_position, id=[13, 14, 2])
        # Phase 4
        self.set_position(goal_position=self.initial_position[1:4], id=self.id[1:4])
        # Phase 5
        self.set_position(goal_position=self.initial_position[0:4], id=self.id[0:4])
        # Phase 5
        self.set_position(goal_position=self.initial_position, id=self.id)

    def forward_kinematics(self):
        a, b1, b2, c1, d1, d2 = 0.08025, 0.1425, 0.02075, 0.1375, 0.138, 0.0205 # geometry information of the arm
        qpos = self._int322rad(self.get_position[:-1])
        alpha, theta1, theta2, theta3 = qpos[0], qpos[1], qpos[2], qpos[3]
        r = b1 * np.sin(theta1) + b2 * np.cos(theta1) + c1 * np.cos(theta1 + theta2) + d1 * np.cos(theta1 + theta2 + theta3) - d2 * np.sin(theta1 + theta2 + theta3)  
        z = a + b1 * np.cos(theta1) - b2 * np.sin(theta1) - c1 * np.sin(theta1 + theta2) - d1 * np.sin(theta1 + theta2 + theta3) - d2 * np.cos(theta1 + theta2 + theta3)     
        beta = theta1 + theta2 + theta3

        return (alpha, r, z, beta)

    def inverse_kinematics(self, r_target, z_target, beta_target):
        qpos = self._neg(self._int322rad(self._decompensite4origin(self.get_position()[:-1])))
        theta1_guess, theta2_guess, theta3_guess = qpos[1], qpos[2], qpos[3]
        solution = fsolve(self._inverse_kinematics_equations, 
                          (theta1_guess, theta2_guess, theta3_guess), args=(r_target, z_target, beta_target))
        solution = self._neg(self._rad2int32(solution))

        return solution


    def _inverse_kinematics_equations(self, vars, r_target, z_target, beta_target):
        a, b1, b2, c1, d1, d2 = 0.08025, 0.1425, 0.02075, 0.1375, 0.138, 0.0205 # geometry information of the arm
        theta1, theta2, theta3 = vars    
        r = b1 * np.sin(theta1) + b2 * np.cos(theta1) + c1 * np.cos(theta1 + theta2) + d1 * np.cos(theta1 + theta2 + theta3) - d2 * np.sin(theta1 + theta2 + theta3)  
        z = a + b1 * np.cos(theta1) - b2 * np.sin(theta1) - c1 * np.sin(theta1 + theta2) - d1 * np.sin(theta1 + theta2 + theta3) - d2 * np.cos(theta1 + theta2 + theta3) 
        
        beta = theta1 + theta2 + theta3
        return [r - r_target, z - z_target, beta - beta_target]

    def _degree2int32(self, angle):
        return [int(item * 4096 / 360) for item in angle]
    
    def _int322degree(self, angle):
        return [int(item * 360 / 4096) for item in angle]

    def _rad2int32(self, angle):
        return [int(item * 4096 / (2*np.pi)) for item in angle]
    
    def _int322rad(self, angle):
        return [item * (2*np.pi) / 4096 for item in angle]
    
    def _meter2int32(self, distance):
        """
            Just use for gripper
        """
        return [int(20161.29 * distance + 350)]
    
    def _int322meter(self):
        """
            Just use for gripper
        """
        pass
    
    def _compensite4origin(self, angle):
        return [item + 2047 for item in angle]
    
    def _decompensite4origin(self, angle):
        return [item - 2047 for item in angle]
    
    def _neg(self, angle):
        return [-item for item in angle]