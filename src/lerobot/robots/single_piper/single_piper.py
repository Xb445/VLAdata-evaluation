#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_single_piper import SinglePiperConfig

import math

# piper sdk
from piper_sdk import *
import inspect
from scipy.spatial.transform import Rotation as R
import numpy as np

logger = logging.getLogger(__name__)

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=True).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

# rotate6D转欧拉角
def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz',degrees=True)

def process_position(position):
    """
    position: 长度为10的list或np.ndarray
    返回新的position: 长度10
    """
    position = np.asarray(position)
    if len(position) != 10:
        raise ValueError("position must have length 10")

    # 前3个数保持不动
    first_three = position[:3]

    # 3:9 的6个数 -> rotate6d_to_xyz -> 3个数   单位换算成0.001°
    rotated_three = rotate6d_to_xyz(position[3:9]) * 1000

    # 第10个数保持不动
    last_one = position[9]

    # 拼回去
    new_position = np.concatenate([first_three, rotated_three, [last_one]])
    return new_position

class SinglePiper(Robot):
    config_class = SinglePiperConfig
    name = "single_piper"

    def __init__(self, config: SinglePiperConfig):
        super().__init__(config)
        self.config = config
        # 创建 robot 连接 (端口从配置读取)
        self.piper = C_PiperInterface(config.port)

        # 根据配置选择 joint、eef(rotate6d) 或 joint_eef 模式
        joint_motors = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
        eef_motors = {
            "x.pos": 0.0,
            "y.pos": 0.0,
            "z.pos": 0.0,
            "r0.pos": 0.0,
            "r1.pos": 0.0,
            "r2.pos": 0.0,
            "r3.pos": 0.0,
            "r4.pos": 0.0,
            "r5.pos": 0.0,
            "gripper.pos": 0.0,
        }
        if config.ctrl_mode == "joint":
            self.motors = joint_motors
        elif config.ctrl_mode == "joint_eef":
            self.motors = {**joint_motors, **eef_motors}
        else:
            self.motors = eef_motors

        # 创建相机
        self.cameras = make_cameras_from_configs(config.cameras)
        # 创建 robot 是否连接的标志位
        self.is_robot_connected = False
        self._enable_flag = False
        # 读取耗时日志计数器: get_observation 在控制循环中每步调用,
        # 避免每步都打 DEBUG 造成刷屏, 仅每 200 次打印一次
        self._read_state_log_count = 0

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {k: float for k in self.motors.keys()}

    @property
    def _motors_eef_ft(self) -> dict[str, type]:
        # return {k: float for k in self.motors_eef.keys()}
        pass

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Joint 或 Rotate6D都使用这个
        return {**self._motors_ft, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.is_robot_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        # 如果 robot 和 cam 都已成功连接, 则报错
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # robot connect
        self.piper.ConnectPort()
        self.is_robot_connected = True

        for cam in self.cameras.values():
            cam.connect()
        logger.info(f"{self} connected.")
        # 使能
        # 回放功能需要将其打开
        # self._enable_single()
        
    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        # 暂时空实现
        pass

    def configure(self):
        # 暂时空实现
        pass


    def enable_fun(self):
        '''
        使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序
        '''
        enable_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        elapsed_time_flag = False

        # 循环检查使能状态
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            # 检查所有电机的使能状态
            enable_flag = self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("使能状态:", enable_flag)
            # 使能机械臂
            self.piper.EnableArm(7)
            # 控制夹爪
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            print("--------------------")
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)

        # 如果超时则退出程序
        if(elapsed_time_flag):
            print("程序自动使能超时,退出程序")
            exit(0)

    def _enable_single(self):
        """使能单个机械臂"""
        self.piper.EnableArm(7)
        self.enable_fun()
        self.piper.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        self.piper.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        self._enable_flag = True

    def control_arm(self,joints, speed=15):

        # joints [rad]
        PI = math.pi
        factor = 1000 * 180 / PI

        position = joints

        joint_0 = int(position[0] * factor)
        joint_1 = int(position[1] * factor)
        joint_2 = int(position[2] * factor)
        joint_3 = int(position[3] * factor)
        joint_4 = int(position[4] * factor)
        joint_5 = int(position[5] * factor)

        if (joint_4 < -70000) :
            joint_4 = -70000

        # piper.MotionCtrl_1()
        self.piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        if len(joints) > 6:
            joint_6 = round(position[6] * 70 * 1000)
            self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

        # print(self.piper.GetArmStatus())
        # print(position)


    def pos_ctrl(self,position):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data (): 
        """
        print("末端控制")
        x = int(position[0]*1000000)  # 模型输出的单位是m，单位转换为0.001mm
        y = int(position[1]*1000000) + 250000
        z = int(position[2]*1000000)
        roll = int(position[3])
        pitch = int(position[4])
        yaw = int(position[5])
        gripper = round(position[6] * 70 * 1000)  # 夹爪控制：这里*70是因为夹爪以中点为0点，两边张开的范围是0~70mm，乘以1000是因为输出的单位是mm，但后面的SDk夹爪控制接口需要的是0.001mm
        
        print(x,y,z,roll,pitch,yaw,gripper)
        self.piper.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        # 发送末端位姿控制命令
        self.piper.EndPoseCtrl(x,y,z,roll,pitch,yaw)
        # robot.piper.EndPoseCtrl(55648, 3080, 195009, 160944, 80708, 163636)
        # # 发送关节控制命令
        # robot.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        # 发送夹爪控制命令
        self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        if self.config.ctrl_mode in ["joint", "joint_eef"]:
            joint_state = self.piper.GetArmJointMsgs()
            self.motors["joint_1.pos"] = round(joint_state.joint_state.joint_1 * 0.001 / 57.3, 8)
            self.motors["joint_2.pos"] = round(joint_state.joint_state.joint_2 * 0.001 / 57.3, 8)
            self.motors["joint_3.pos"] = round(joint_state.joint_state.joint_3 * 0.001 / 57.3, 8)
            self.motors["joint_4.pos"] = round(joint_state.joint_state.joint_4 * 0.001 / 57.3, 8)
            self.motors["joint_5.pos"] = round(joint_state.joint_state.joint_5 * 0.001 / 57.3, 8)
            self.motors["joint_6.pos"] = round(joint_state.joint_state.joint_6 * 0.001 / 57.3, 8)
            gripper_raw = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
            self.motors["gripper.pos"] = round((gripper_raw * 0.001) / 70.0, 8)

        if self.config.ctrl_mode != "joint":
            # rotate6d
            end_pos_state = self.piper.GetArmEndPoseMsgs() # 末端位姿
            self.motors["x.pos"] = round(end_pos_state.end_pose.X_axis * 0.000001,8)     #单位为1000mm
            self.motors["y.pos"] = round(end_pos_state.end_pose.Y_axis * 0.000001,8)
            self.motors["z.pos"] = round(end_pos_state.end_pose.Z_axis * 0.000001,8)
            
            rx = end_pos_state.end_pose.RX_axis * 0.001
            ry = end_pos_state.end_pose.RY_axis * 0.001
            rz = end_pos_state.end_pose.RZ_axis * 0.001
            q = np.array([rx, ry, rz], dtype=np.float64)
            rotate6d = euler_to_rotate6d(q, pattern="xyz")
            for i in range(6):
                self.motors[f"r{i}.pos"] = float(rotate6d[i])
            gripper_raw = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
            self.motors["gripper.pos"] = round((gripper_raw * 0.001) / 70.0, 8)

        dt_ms = (time.perf_counter() - start) * 1e3
        # 抑制刷屏: 仅每 200 次读取打印一次耗时
        self._read_state_log_count += 1
        if self._read_state_log_count % 200 == 0:
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 只是复制joint数据
        obs_dict = self.motors.copy()

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict

    # 老版是将该函数直接作为返回action使用
    # 新版应该是使用该函数向机械臂发送数据
    def send_action(self, action: list) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.ctrl_mode == "joint":
            goal_pos = {
                key.removesuffix(".pos"): val.item() if hasattr(val, "item") else val
                for key, val in action.items() if key.endswith(".pos")
            }
            goal_list = list(goal_pos.values())

            print(goal_list)

            if self._enable_flag:
                self.control_arm(goal_list,60)
        else:
            # rotate6d 模式，回放遥操作数据集
            # goal_pos = {
            #     key.removesuffix(".pos"): val.item() if hasattr(val, "item") else val
            #     for key, val in action.items() if key.endswith(".pos")
            # }
            # goal_list = process_position(list(goal_pos.values()))

            # if self._enable_flag:
            #     self.pos_ctrl(goal_list)
            # rotate6d 模式，回放仿真人手数据集
            goal_pos = {
                key.removesuffix(".epos"): val.item() if hasattr(val, "item") else val
                for key, val in action.items() if key.endswith(".epos")
            }
            goal_list = process_position(list(goal_pos.values()))

            if self._enable_flag:
                self.pos_ctrl(goal_list)



    def get_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.motors is None:
            raise DeviceNotConnectedError(f"self motor value is None.")
        
        action_dict = self.motors.copy()
        return action_dict

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
