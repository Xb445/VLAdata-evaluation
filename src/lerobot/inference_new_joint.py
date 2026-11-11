import numpy as np
import torch
import time

# 导入机器人模型
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    # single_piper,
    # moving_dual_piper,
    piper_follower,
)
# 导入相机相关模块
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.utils.robot_utils import busy_wait
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
# 导入策略模型
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser
# 导入piper SDK
from piper_sdk import *



@dataclass
class InferenceConfig:
    """推理配置类，用于存储机器人配置和策略配置"""
    robot: RobotConfig
    # 是否使用策略控制机器人
    policy: PreTrainedConfig | None = None
    # 模型检查点路径
    ckpt_path: str = None
    # 任务名称
    task: str = None

    def __post_init__(self):
        """初始化后处理，从命令行参数获取预训练模型路径"""
        # 从命令行参数获取策略路径
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            # 从预训练路径加载配置
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
        return ["policy"]


def enable_fun(piper: C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5秒,如果使能超时则退出程序
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
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        # 使能机械臂
        piper.EnableArm(7)
        # 控制夹爪
        piper.GripperCtrl(0, 1000, 0x01, 0)
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


import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """主要的推理函数，执行机器人控制循环"""
    # # 根据配置创建机器人实例
    robot = make_robot_from_config(cfg.robot)
    # 如果机器人未连接则进行连接
    if not robot.is_connected:
        robot.connect()

    # 如果是移动双Piper机器人，初始化ROS节点
    # if cfg.robot.type == "moving_dual_piper":
    #     import rospy
    #     from geometry_msgs.msg import Twist
    #     rospy.init_node("pub_cmd_vel", anonymous=True)
    #     cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    #     print(f"using moving_dual_piper robot, ROS node 'pub_cmd_vel' initialized.")

    # 推理参数设置
    inference_time_s = 10000  # 推理总时间（秒）
    fps = 30  # 控制频率（Hz）
    # 设备选择
    device = "cuda"
    
    # 创建策略模型
    ckpt_path = cfg.ckpt_path
    print(f"ckpt_path : {ckpt_path}")
    
    # 检查任务名称是否提供
    if cfg.task == None:
        raise ValueError("You need to provide a task name.")
    
    # 根据策略类型创建相应的策略模型
    if cfg.policy.type == "act":
        policy = ACTPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(ckpt_path)
    # elif cfg.policy.type == "smolvla":
    #     policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "groot":
        policy = GrootPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "xvla":
        policy = XVLAPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "pi05":
        policy = PI05Policy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "wall_x":
        policy = WallXPolicy.from_pretrained(ckpt_path)
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        # prompt = cfg.task
        # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        # language_tokens = inputs["input_ids"]
        # print("[Info] Model converted to float32 for consistent inference.")

    else:
        raise ValueError("You need to provide a valid policy between act/diffusion/smolvla/xvla/wall_x.")
    

    # dataset_id = "/home/hpc/VLA/data_test_new_v3.0"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    # dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    # print(dataset_metadata)
    # exit(0)
    preprocess, postprocess = make_pre_post_processors(policy.config, ckpt_path)

    # 设置模型为评估模式
    policy.eval()
    # 将模型移动到指定设备
    policy.to(device)
    # 重置策略状态
    policy.reset()
    
    # 关节角度到控制信号的缩放因子
    factor = 57324.840764  # 1000 * 180 / 3.14
    
    # 根据机器人类型进行使能操作
    if cfg.robot.type == "piper_follower":
        # 使能单臂机器人
        # robot.arm.EnableArm(7)
        enable_fun(piper=robot.arm)
        robot.arm.MotionCtrl_2(0x01, 0x01, 5, 0x00)   #这行是关节控制
        robot.arm.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
    elif cfg.robot.type == "single_piper":
        # 使能单臂机器人
        robot.piper.EnableArm(7)
        enable_fun(piper=robot.piper)
        robot.piper.MotionCtrl_2(0x01, 0x01, 5, 0x00)   #这行是关节控制
        robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
    elif cfg.robot.type == "dual_piper" or cfg.robot.type == "moving_dual_piper":
        # 使能左臂
        robot.piper_left.EnableArm(7)
        enable_fun(piper=robot.piper_left)
        robot.piper_left.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        robot.piper_left.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        # 使能右臂
        robot.piper_right.EnableArm(7)
        enable_fun(piper=robot.piper_right)
        robot.piper_right.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        robot.piper_right.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
    else:
        raise ValueError("Enable arm failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

    # 等待使能完成
    time.sleep(2.0)

    # 主控制循环
    for _ in range(inference_time_s * fps):
        start_loop_t = time.perf_counter()  # 记录循环开始时间
        
        # 获取当前观测数据（状态和图像）
        observation = robot.get_observation()

        observation_frame = {}

        # 用于保存状态数值
        state_values = []
        state_values_epos = []

        # 处理观测数据
        for key, value in observation.items():
            # 把所有带.pos的位置信息合并成状态向量
            if key.endswith('.pos'):
                state_values.append(np.float32(value))
            if key.endswith('.epos'):
                state_values_epos.append(np.float32(value))

            # 处理图像数据（HWC格式的ndarray）
            elif isinstance(value, np.ndarray) and value.ndim == 3:
                observation_frame[f'observation.images.{key}'] = value

        # 添加合并后的状态向量
        observation_frame['observation.state'] = np.array(state_values, dtype=np.float32) #这里是获取关节
        # observation_frame['observation.state'] = np.array(state_values_epos, dtype=np.float32)  #这里是获取末端位姿

        # print("state:",observation_frame['observation.state'])

        # 数据预处理
        for name in observation_frame:
            # print(name)
            if "image" in name:
                # 图像预处理：归一化并转换通道顺序

                obs = observation_frame[name].astype(np.float32) / 255.0  # 归一化到[0,1]

                obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW

            else:
                obs = observation_frame[name]  # 状态数据直接使用

            # 增加batch维度并转换为tensor
            obs = np.expand_dims(obs, axis=0)
            observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=device)
            
        # 添加任务名称
        if cfg.task == None:
            raise ValueError("You need to provide a task name.")
        observation_frame["task"] = [cfg.task]
        # if cfg.policy.type == "xvla":
        #     observation_frame["observation.language.tokens"] = language_tokens
        
        observation_frame = preprocess(observation_frame)
        #print(observation_frame)

        # 使用策略模型选择动作（推理模式）
        with torch.inference_mode():

            action = policy.select_action(observation_frame)
            print(f"action_policy: {action}")
        
        action = postprocess(action)
        print(f"action_post: {action}")
        infer_time = time.perf_counter() - start_loop_t  # 计算本次循环耗时
        # print("infer_time",infer_time)

        # print(action)
        # 处理动作输出
        if cfg.policy.type == "xvla":
            numpy_action = action.squeeze(0).cpu().to(torch.float32).numpy()
        else:
            numpy_action = action.squeeze(0).cpu().numpy()  # 去掉batch维度，转到CPU，再转numpy
        # numpy_action = action.squeeze(0).cpu().numpy()  # 去掉batch维度，转到CPU，再转numpy
        position = numpy_action.tolist()  # 转成python list方便后续使用
        # print(f"position: {position}")

        # 设置各关节运动限位（弧度）
        joint_limits = [(-3, 3)] * 6
        joint_limits[0] = (-2.687, 2.687)     # 关节0限位
        joint_limits[1] = (0.0, 3.403)        # 关节1限位
        joint_limits[2] = (-3.0541012, 0.0)   # 关节2限位
        joint_limits[3] = (-1.5499, 1.5499)   # 关节3限位
        joint_limits[4] = (-1.22, 1.22)       # 关节4限位
        joint_limits[5] = (-1.7452, 1.7452)   # 关节5限位
        
        # 定义限位函数
        def clamp(value, min_val, max_val):
            """将值限制在最小最大值之间"""
            return max(min(value, max_val), min_val)

        # 根据机器人类型执行相应的控制命令
        if cfg.robot.type == "piper_follower":
            # 单臂机器人控制
            joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
            joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
            joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
            joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
            joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
            joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
            joint_6 = round(position[6] * 70 * 1000)  # 夹爪控制  这里*70是因为夹爪以中点为0点，两边张开的范围是0~70mm，乘以1000是因为输出的单位是mm，但后面的SDk夹爪控制接口需要的是0.001mm
            
            
            print(f"JointCtrl value: {joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6}")
            
            # 发送关节控制命令
            robot.arm.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            # 发送夹爪控制命令
            robot.arm.GripperCtrl(abs(joint_6), 5000, 0x01, 0)
            '''
            (method) def GripperCtrl(
            gripper_angle: int = 0,
            gripper_effort: int = 0,
            gripper_code: Literal[0, 1, 2, 3] = 0,
            set_zero: Literal[0, 174] = 0
            ) -> None
            夹爪控制
            CAN ID:
            0x159

            Args
            gripper_angle : int
            夹爪范围, 以整数表示, 单位0.001mm
            gripper_effort : int
            夹爪力矩,单位 0.001N/m,范围0-5000,对应0-5N/m  (控制夹爪抓取力度)
            gripper_code : int
            0x00失能; 0x01使能; 0x02失能清除错误; 0x03使能清除错误.
            set_zero
            (int): 设定当前位置为0点, 0x00无效值; 0xAE设置零点
            '''
        elif cfg.robot.type == "single_piper":
            # 单臂机器人控制
            joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
            joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
            joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
            joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
            joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
            joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
            joint_6 = round(position[6] * 70 * 1000)  # 夹爪控制  这里*70是因为夹爪以中点为0点，两边张开的范围是0~70mm，乘以1000是因为输出的单位是mm，但后面的SDk夹爪控制接口需要的是0.001mm
            
            print(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5,joint_6)
            # 发送关节控制命令
            robot.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            # 发送夹爪控制命令
            robot.piper.GripperCtrl(abs(joint_6), 5000, 0x01, 0)
            '''
            (method) def GripperCtrl(
            gripper_angle: int = 0,
            gripper_effort: int = 0,
            gripper_code: Literal[0, 1, 2, 3] = 0,
            set_zero: Literal[0, 174] = 0
            ) -> None
            夹爪控制
            CAN ID:
            0x159

            Args
            gripper_angle : int
            夹爪范围, 以整数表示, 单位0.001mm
            gripper_effort : int
            夹爪力矩,单位 0.001N/m,范围0-5000,对应0-5N/m  (控制夹爪抓取力度)
            gripper_code : int
            0x00失能; 0x01使能; 0x02失能清除错误; 0x03使能清除错误.
            set_zero
            (int): 设定当前位置为0点, 0x00无效值; 0xAE设置零点
            '''

        elif cfg.robot.type == "dual_piper":
            # 双臂机器人控制 - 左臂
            left_joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
            left_joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
            left_joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
            left_joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
            left_joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
            left_joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
            left_joint_6 = round(position[6] * 70 * 1000)
            
            # 双臂机器人控制 - 右臂
            right_joint_0 = round(clamp(position[7], joint_limits[0][0], joint_limits[0][1]) * factor)
            right_joint_1 = round(clamp(position[8], joint_limits[1][0], joint_limits[1][1]) * factor)
            right_joint_2 = round(clamp(position[9], joint_limits[2][0], joint_limits[2][1]) * factor)
            right_joint_3 = round(clamp(position[10], joint_limits[3][0], joint_limits[3][1]) * factor)
            right_joint_4 = round(clamp(position[11], joint_limits[4][0], joint_limits[4][1]) * factor)
            right_joint_5 = round(clamp(position[12], joint_limits[5][0], joint_limits[5][1]) * factor)
            right_joint_6 = round(position[13] * 70 * 1000)
            
            # 发送控制命令
            robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
            robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
            robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
            robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
            
        elif cfg.robot.type == "moving_dual_piper" and not rospy.is_shutdown():
            # 移动双臂机器人控制 - 底盘控制
            twist_msg = Twist()
            twist_msg.linear.x = position[0]   # 线速度控制
            twist_msg.angular.z = position[1]  # 角速度控制
            cmd_vel_pub.publish(twist_msg)
            
            # 左臂控制
            left_joint_0 = round(clamp(position[2], joint_limits[0][0], joint_limits[0][1]) * factor)
            left_joint_1 = round(clamp(position[3], joint_limits[1][0], joint_limits[1][1]) * factor)
            left_joint_2 = round(clamp(position[4], joint_limits[2][0], joint_limits[2][1]) * factor)
            left_joint_3 = round(clamp(position[5], joint_limits[3][0], joint_limits[3][1]) * factor)
            left_joint_4 = round(clamp(position[6], joint_limits[4][0], joint_limits[4][1]) * factor)
            left_joint_5 = round(clamp(position[7], joint_limits[5][0], joint_limits[5][1]) * factor)
            left_joint_6 = round(position[8] * 70 * 1000)
            
            # 右臂控制
            right_joint_0 = round(clamp(position[9], joint_limits[0][0], joint_limits[0][1]) * factor)
            right_joint_1 = round(clamp(position[10], joint_limits[1][0], joint_limits[1][1]) * factor)
            right_joint_2 = round(clamp(position[11], joint_limits[2][0], joint_limits[2][1]) * factor)
            right_joint_3 = round(clamp(position[12], joint_limits[3][0], joint_limits[3][1]) * factor)
            right_joint_4 = round(clamp(position[13], joint_limits[4][0], joint_limits[4][1]) * factor)
            right_joint_5 = round(clamp(position[14], joint_limits[5][0], joint_limits[5][1]) * factor)
            right_joint_6 = round(position[15] * 70 * 1000)
            
            # 发送控制命令
            robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
            robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
            robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
            robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        
        else:
            raise ValueError("Execute action failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

        # 控制循环频率
        dt_s = time.perf_counter() - start_loop_t  # 计算本次循环耗时
        
        # print("dts",dt_s)
        busy_wait(1 / fps - dt_s)  # 等待剩余时间以维持固定频率


if __name__ == "__main__":
    # 程序入口点
    inference()
