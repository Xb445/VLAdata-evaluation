#!/usr/bin/env python3
from piper_sdk import *
import time
import math

PI = math.pi
factor = 1000 * 180 / PI
receive_object_center = False
object_center = []
simulation = True


def control_arm(joints, speed=2):
    position = joints

    joint_0 = int(position[0] * factor)
    joint_1 = int(position[1] * factor)
    joint_2 = int(position[2] * factor)
    joint_3 = int(position[3] * factor)
    joint_4 = int(position[4] * factor)
    joint_5 = int(position[5] * factor)

    if (joint_4 < -70000) :
        joint_4 = -70000

    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

    # if len(joints) > 6:
    #     joint_6 = round(position[6] * 1000 * 1000)
    #     piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    # print(piper.GetArmStatus())
    # print(position)


def enable_fun(piper: C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)

        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if (elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)


if __name__ == "__main__":
    piper = C_PiperInterface_V2("can1")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    piper.GripperCtrl(70000, 1000, 0x01, 0)
    
    # 设置初始位置
    joints = [0, 0, 0, 0, 0, 0, 0]
    control_arm(joints, 50)

 