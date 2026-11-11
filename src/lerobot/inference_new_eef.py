import numpy as np
import torch
import time

# 导入机器人模型
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    # moving_dual_piper,
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
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser
# 导入piper SDK
from piper_sdk import *
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from lerobot.policies.factory import make_pre_post_processors
from scipy.spatial.transform import Rotation as R

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


# 欧拉角 转 rotate6D 输入:q 角度
def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=True).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

# rotate6D转欧拉角
def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2] # 0 2 4
    a2 = v6[..., 1:6:2] # 1 3 5
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
    rotated_three = rotate6d_to_xyz(position[3:9])

    # 第10个数保持不动
    last_one = position[9]

    # 拼回去
    new_position = np.concatenate([first_three, rotated_three, [last_one]])
    return new_position

@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """主要的推理函数，执行机器人控制循环"""
    # # 根据配置创建机器人实例
    robot = make_robot_from_config(cfg.robot)
    # 如果机器人未连接则进行连接
    if not robot.is_connected:
        robot.connect()

    # 推理参数设置
    inference_time_s = 10000  # 推理总时间（秒）
    fps = 20  # 控制频率（Hz）
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
    elif cfg.policy.type == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "groot":
        policy = GrootPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "xvla":
        policy = XVLAPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "pi05":
        policy = PI05Policy.from_pretrained(ckpt_path)
    else:
        raise ValueError("You need to provide a valid policy between act/diffusion/smolvla/xvla.")

    preprocess, postprocess = make_pre_post_processors(policy.config, ckpt_path)

    policy.eval()
    policy.to(device)
    policy.reset()
    
    # 关节角度到控制信号的缩放因子
    factor = 57324.840764  # 1000 * 180 / 3.14
    
    # 根据机器人类型进行使能操作
    if cfg.robot.type == "single_piper":
        # 使能单臂机器人
        robot.piper.EnableArm(7)
        enable_fun(piper=robot.piper)
        robot.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)   #这行是末端位姿控制
        robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
    else:
        raise ValueError("Enable arm failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

    # 等待使能完成
    time.sleep(2.0)
    #    禁用科学计数法，设置保留小数位数
    np.set_printoptions(suppress=True, precision=6)

    # 主控制循环
    for _ in range(inference_time_s * fps):
        start_loop_t = time.perf_counter()  # 记录循环开始时间
        
        # 获取当前观测数据（状态和图像）
        observation = robot.get_observation()

        observation_frame = {}

        # 用于保存状态数值
        state_values = []

        # 处理观测数据
        for key, value in observation.items():
            # 把所有带.pos的位置信息合并成状态向量
            if key.endswith('.pos'):
                state_values.append(np.float32(value))
            # 处理图像数据（HWC格式的ndarray）
            elif isinstance(value, np.ndarray) and value.ndim == 3:
                observation_frame[f'observation.images.{key}'] = value

        # print("观察值state_values:",state_values)

        # 先对得到的 state_values 进行 Rotate6D 转换
        # print(len(state_values))
        # # 前3个数保持不动
        # first_three = state_values[:3]
        # # 将欧拉角的单位转换为1°
        # rotated_three = euler_to_rotate6d(np.array(state_values[3:6])*1000)
        # # 第7个数保持不动
        # last_one = state_values[6]
        # # 拼回去
        # state_values = np.concatenate([first_three, rotated_three, [last_one]])
        # print("转换成rotate6D-state_values_epos",state_values)

        observation_frame['observation.state'] = np.array(state_values, dtype=np.float32) 
        # print("observation_frame['observation.state']",observation_frame['observation.state'])

        for name in observation_frame:
            if "image" in name:
                obs = observation_frame[name].astype(np.float32) / 255.0  # 归一化到[0,1]
                obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
            else:
                obs = observation_frame[name]  # 状态数据直接使用
            obs = np.expand_dims(obs, axis=0)
            observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=device)
            
        if cfg.task == None:
            raise ValueError("You need to provide a task name.")
        observation_frame["task"] = [cfg.task]
        observation_frame = preprocess(observation_frame)

        # 使用策略模型选择动作（推理模式）
        with torch.inference_mode():
            action = policy.select_action(observation_frame)

        action = postprocess(action)
        # print("预测的动作-action:Rotate6D格式:",action)

        # x y z rx ry rz gripper   7
        # x y z rotate6D gripper   10

        infer_time = time.perf_counter() - start_loop_t  # 计算本次循环耗时
        # print("infer_time",infer_time)

        if cfg.policy.type == "xvla":
            numpy_action = action.squeeze(0).cpu().to(torch.float32).numpy()
        else:
            numpy_action = action.squeeze(0).cpu().numpy()  # 去掉batch维度，转到CPU，再转numpy    

        position = numpy_action.tolist()  # 转成python list方便后续使用
        
        # 保存position到txt文件
        with open("position_output.txt", "a") as f:
            f.write(str(position) + "\n")


        # 此刻得到的 action 是 Rotate6D 形态的需要进行转换
        # 并且转换只是 action[]
        position = process_position(position)

        x = int(position[0]*1000000)  # 模型输出的单位是m，单位转换为0.001mm
        y = int(position[1]*1000000)
        z = int(position[2]*1000000)
        roll = int(position[3] * 1000)  #将单位转换为0.001°
        pitch = int(position[4] * 1000)
        yaw = int(position[5] * 1000)
        gripper = round(position[6] * 70 * 1000)  # 夹爪控制：这里*70是因为夹爪以中点为0点，两边张开的范围是0~70mm，乘以1000是因为输出的单位是mm，但后面的SDk夹爪控制接口需要的是0.001mm
        print([x, y, z, roll, pitch, yaw, gripper])

        #为了让夹爪抓取的时候更加紧，当夹爪小于一定阈值时，将夹爪间距进一步缩小
        # print("==========,ori")
        # print(x,y,z,roll,pitch,yaw,gripper)
        if gripper < 10000 and gripper > 5000:
            gripper = gripper - 5000
            gripper = max(0,gripper)
        # print("=========,after")
        # print(x,y,z,roll,pitch,yaw,gripper)

        robot.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)

        # 发送末端位姿控制命令
        robot.piper.EndPoseCtrl(x,y,z,roll,pitch,yaw)
        robot.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
        # time.sleep(1)

        dt_s = time.perf_counter() - start_loop_t  # 计算本次循环耗时
        busy_wait(1 / fps - dt_s)  # 等待剩余时间以维持固定频率

if __name__ == "__main__":
    # 程序入口点
    inference()

# python lerobot/inference_new_eef.py --robot.type=single_piper --robot.id=single_piper_robot --robot.port=can1 --robot.cameras="{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30}}" --policy.compile_model=false  --robot.ctrl_mode="eef" --policy.type=pi05 --ckpt_path=/home/hpc/VLA/lerobot_models/pi05/2_28_rotate6d_300/24000/pretrained_model  --task="Pick up the express package and put it on the tray."
