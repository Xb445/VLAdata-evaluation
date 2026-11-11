#!/usr/bin/env python
"""
================================================================================
ACT 模型关节模式（Joint Mode）推理脚本 v2.0
================================================================================

【适用场景】
  使用 joint 模式数据集（如 batch_3_4_joint_merged / batch5A_joint_merged）
  训练出的 ACT 模型进行真机推理部署。

【ACT 模型推理机制】
  ACT（Action Chunking Transformer）一次预测整个动作序列（chunk），
  然后通过内部队列逐帧吐出动作，队列耗尽后重新预测下一段。
  - predict_action_chunk(): 返回 (B, chunk_size, action_dim) 完整序列
  - select_action(): 从内部队列逐帧取动作，队列空时自动触发预测

【两种运行模式】
  模式1：纯 ACT 推理（--smooth.mode=none）
    → 直接使用模型内置的 action queue，无额外平滑处理
    → 模型内部的 select_action 已包含跨 chunk 过渡平滑

  模式2：TrajectorySmoother 平滑（--smooth.mode=smooth）
    → 在模型输出后叠加 EMA + 窗口平均 + 限幅 + 死区
    → 适合需要精细调整平滑参数的场景

  模式3：兼容 RTC 调用（--smooth.mode=rtc）
    → 预留 RTC 接口，方便未来集成实时 chunking
    → 当前 ACT 无原生 RTC 支持，降级为 TrajectorySmoother 模式

【用法示例】

  # 纯 ACT 推理（无额外平滑）
  python -m lerobot.inference_new_joint_act \
    --robot.type=single_piper \
    --robot.id=single_piper_robot \
    --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --policy.type=act \
    --ckpt_path=/home/ub/Videos/0407/model/020000/pretrained_model \
    --task="Grab the object and move to target position" \
    --smooth.mode=none

  # 带平滑的 ACT 推理
  python -m lerobot.inference_new_joint_act \
    ... \
    --smooth.mode=smooth \
    --smooth.ema_alpha=0.3 \
    --smooth.pos_max_delta=0.03 \
    --smooth.window_size=3

  # 快速启动（使用默认参数）
  python -m lerobot.inference_new_joint_act \
    --robot.type=single_piper \
    --robot.id=single_piper_robot \
    --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --ckpt_path=/data/zhanghang/Videos/0407/lerobot/outputs/train/2026-07-09/10-01-24_act_bs96_lr1e-4/checkpoints/last/pretrained_model \
    --task="Grab the object and move to target position"
================================================================================
"""

import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

# ── 机器人相关 ──
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
)
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.utils.robot_utils import busy_wait

# ── 策略模型 ──
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser

# ── 机械臂 SDK ──
from piper_sdk import *


# =============================================================================
# 第一部分：轨迹平滑器（TrajectorySmoother）
# =============================================================================

class TrajectorySmoother:
    """
    关节空间轨迹平滑器

    【平滑策略组合】
      1. 窗口平均（Window Average）：滑动窗口内的多帧动作取均值，消除高频噪声
      2. EMA（指数移动平均）：对历史动作做指数加权，降低帧间跳变
      3. 限幅（Clipping）：限制相邻帧之间关节角度/夹爪的最大变化量
      4. 死区（Dead Zone）：忽略小于阈值的微量变化，防止静态抖动

    【ACT 适配说明】
      ACT 模型内部已有跨 chunk 过渡平滑（通过 select_action 中的
      _last_action_of_prev_batch 实现），因此外部平滑参数不应设置过强，
      否则可能导致机械臂反应迟钝。

    【参数调优指南】
      - 机械臂抖动明显 → 增大 ema_alpha（如 0.2→0.1），或减小 pos_max_delta
      - 机械臂反应迟钝 → 减小 ema_alpha（如 0.1→0.3），或增大 pos_max_delta
      - 静止时有微颤 → 增大 pos_dead_zone（如 0.0001→0.001）
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        pos_max_delta: float = 0.03,
        gripper_max_delta: float = 0.15,
        pos_dead_zone: float = 0.001,
        gripper_dead_zone: float = 0.001,
        window_size: int = 3,
        use_window_average: bool = True,
        use_ema: bool = True,
        use_clipping: bool = True,
        use_dead_zone: bool = True,
    ):
        self.ema_alpha = ema_alpha
        self.pos_max_delta = pos_max_delta
        self.gripper_max_delta = gripper_max_delta
        self.pos_dead_zone = pos_dead_zone
        self.gripper_dead_zone = gripper_dead_zone
        self.window_size = window_size
        self.use_window_average = use_window_average
        self.use_ema = use_ema
        self.use_clipping = use_clipping
        self.use_dead_zone = use_dead_zone

        self._ema_action = None
        self._prev_raw_action = None
        self._action_history = deque(maxlen=window_size)
        self._initialized = False

    def reset(self):
        """重置所有内部状态"""
        self._ema_action = None
        self._prev_raw_action = None
        self._action_history.clear()
        self._initialized = False

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        对单个动作进行平滑处理

        Args:
            action: 原始动作 [joint_1, ..., joint_6, gripper]，形状 (7,)

        Returns:
            平滑后的动作，形状 (7,)
        """
        action = np.asarray(action, dtype=np.float64).copy()
        if len(action) != 7:
            raise ValueError(f"Expected action of length 7, got {len(action)}")

        # 1. 窗口平均
        if self.use_window_average:
            self._action_history.append(action)
            if len(self._action_history) >= 2:
                action = np.mean(np.stack(list(self._action_history)), axis=0)

        # 2. EMA 平滑
        if self.use_ema:
            if self._ema_action is None:
                self._ema_action = action
            else:
                self._ema_action = (
                    self.ema_alpha * action +
                    (1 - self.ema_alpha) * self._ema_action
                )
            action = self._ema_action.copy()

        # 3. 限幅
        if self.use_clipping and self._prev_raw_action is not None:
            joint_delta = action[:6] - self._prev_raw_action[:6]
            joint_delta_clipped = np.clip(joint_delta, -self.pos_max_delta, self.pos_max_delta)
            action[:6] = self._prev_raw_action[:6] + joint_delta_clipped

            gripper_delta = action[6] - self._prev_raw_action[6]
            gripper_delta_clipped = np.clip(gripper_delta, -self.gripper_max_delta, self.gripper_max_delta)
            action[6] = self._prev_raw_action[6] + gripper_delta_clipped

        # 4. 死区
        if self.use_dead_zone and self._prev_raw_action is not None:
            joint_change = np.abs(action[:6] - self._prev_raw_action[:6])
            mask_joint = joint_change < self.pos_dead_zone
            action[:6][mask_joint] = self._prev_raw_action[:6][mask_joint]

            if abs(action[6] - self._prev_raw_action[6]) < self.gripper_dead_zone:
                action[6] = self._prev_raw_action[6]

        self._prev_raw_action = action.copy()
        self._initialized = True
        return action


# =============================================================================
# 第二部分：安全监控器（SafetyMonitor）
# =============================================================================

class SafetyMonitor:
    """
    关节跳变安全监控器

    监控相邻两帧之间的关节角度和夹爪变化量，
    当连续多次检测到异常跳变时触发紧急停止。
    """

    def __init__(
        self,
        joint_limit_rad: float = 0.10,
        gripper_limit: float = 0.5,
        consecutive_warnings_limit: int = 15,
    ):
        self.joint_limit_rad = joint_limit_rad
        self.gripper_limit = gripper_limit
        self.consecutive_warnings_limit = consecutive_warnings_limit
        self._prev_action = None
        self._consecutive_warnings = 0
        self._total_warnings = 0

    def check(self, action: np.ndarray) -> tuple:
        """
        检查动作是否安全

        Returns:
            (is_safe: bool, warning_msg: str)
        """
        warnings_list = []
        is_safe = True

        if self._prev_action is not None:
            joint_delta = np.max(np.abs(action[:6] - self._prev_action[:6]))
            gripper_delta = abs(action[6] - self._prev_action[6])

            if joint_delta > self.joint_limit_rad:
                delta_deg = joint_delta * 180.0 / math.pi
                limit_deg = self.joint_limit_rad * 180.0 / math.pi
                warnings_list.append(f"[关节跳变] {delta_deg:.1f}° > {limit_deg:.1f}°")
            if gripper_delta > self.gripper_limit:
                warnings_list.append(f"[夹爪跳变] {gripper_delta:.3f} > {self.gripper_limit:.3f}")

        self._prev_action = action.copy()

        if warnings_list:
            self._consecutive_warnings += 1
            self._total_warnings += 1
        else:
            self._consecutive_warnings = 0

        if self._consecutive_warnings >= self.consecutive_warnings_limit:
            warnings_list.append(f"[紧急] 连续{self._consecutive_warnings}次警告，急停!")
            is_safe = False

        return is_safe, " | ".join(warnings_list) if warnings_list else ""

    def reset(self):
        self._prev_action = None
        self._consecutive_warnings = 0
        self._total_warnings = 0


# =============================================================================
# 第三部分：配置类
# =============================================================================

@dataclass
class SmoothConfig:
    """平滑配置"""
    # 平滑模式: "none"（无平滑） / "smooth"（TrajectorySmoother） / "rtc"（预留，降级为 smooth）
    mode: str = "none"

    # ── EMA 参数 ──
    ema_alpha: float = 0.30

    # ── 限幅参数 ──
    pos_max_delta: float = 0.03       # 关节单步最大变化（rad），~1.72°/步
    gripper_max_delta: float = 0.15   # 夹爪单步最大变化

    # ── 死区参数 ──
    pos_dead_zone: float = 0.001      # 关节死区（rad），~0.057°
    gripper_dead_zone: float = 0.001  # 夹爪死区

    # ── 窗口平均参数 ──
    window_size: int = 3
    use_window_average: bool = True

    # ── 各策略开关 ──
    use_ema: bool = True
    use_clipping: bool = True
    use_dead_zone: bool = True


@dataclass
class InferenceACTConfig:
    """ACT 推理顶层配置"""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # 模型检查点路径
    ckpt_path: str = None

    # 任务描述（文本 prompt，ACT 不使用但为兼容保留）
    task: str = "Perform the manipulation task"

    # 平滑配置
    smooth: SmoothConfig = field(default_factory=SmoothConfig)

    # ── 推理参数 ──
    fps: float = 25.0                 # 控制频率（Hz）
    device: str = "cuda"              # 推理设备
    inference_time_s: int = 10000     # 最大推理时长（秒）

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# =============================================================================
# 第四部分：机械臂使能函数
# =============================================================================

def enable_fun(piper: C_PiperInterface, timeout: float = 5.0):
    """
    使能机械臂并等待使能状态确认

    Args:
        piper: Piper 机械臂 SDK 接口
        timeout: 使能超时时间（秒）
    """
    enable_flag = False
    start_time = time.time()

    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")

        if elapsed_time > timeout:
            print("使能超时，退出程序")
            sys.exit(0)
        time.sleep(1)


def clamp_joint(value: float, joint_limits: list, idx: int) -> float:
    """将关节角度限制在安全范围内"""
    low, high = joint_limits[idx]
    return max(min(value, high), low)


# 关节限位表（弧度）
JOINT_LIMITS = [
    (-2.687, 2.687),       # 关节0
    (0.0, 3.403),          # 关节1
    (-3.0541012, 0.0),     # 关节2
    (-1.5499, 1.5499),     # 关节3
    (-1.22, 1.22),         # 关节4
    (-1.7452, 1.7452),     # 关节5
]

# rad → 0.001° 转换因子
RAD_TO_MILLIDEG = 1000.0 * 180.0 / math.pi  # ≈ 57324.84


# =============================================================================
# 第五部分：主推理函数
# =============================================================================

@parser.wrap()
def inference(cfg: InferenceACTConfig) -> None:
    """ACT 模型关节模式推理主函数"""

    # ========== 打印配置信息 ==========
    print("=" * 60)
    print("  ACT 模型 Joint 模式推理")
    print("=" * 60)
    print(f"  检查点路径:   {cfg.ckpt_path}")
    print(f"  任务描述:     {cfg.task}")
    print(f"  控制频率:     {cfg.fps} Hz")
    print(f"  推理设备:     {cfg.device}")
    print(f"  平滑模式:     {cfg.smooth.mode}")
    print(f"  状态/动作维度: 7 (joint_1~6, gripper)")
    print("=" * 60)

    # ========== 验证 ctrl_mode ==========
    if cfg.robot.ctrl_mode != "joint":
        print("[WARNING] ctrl_mode 不是 'joint'！ACT 模型使用 joint 数据训练。")
        print("[WARNING] 请添加 --robot.ctrl_mode=\"joint\"")
        raise ValueError(
            "This inference script requires ctrl_mode='joint'. "
            "Please add --robot.ctrl_mode=\"joint\""
        )

    # ========== 创建机器人实例 ==========
    robot = make_robot_from_config(cfg.robot)
    if not robot.is_connected:
        robot.connect()
    print(f"[INFO] 机器人已连接: {cfg.robot.type}")

    # ========== 加载 ACT 模型 ==========
    ckpt_path = cfg.ckpt_path
    if not ckpt_path:
        raise ValueError("必须指定 --ckpt_path 参数指向训练好的模型检查点目录")

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"检查点路径不存在: {ckpt_path}")

    print(f"[INFO] 加载 ACT 模型: {ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    print(f"[INFO] ACT 模型配置:")
    print(f"       chunk_size: {policy.config.chunk_size}")
    print(f"       n_action_steps: {policy.config.n_action_steps}")
    print(f"       vision_backbone: {policy.config.vision_backbone}")
    print(f"       use_vae: {policy.config.use_vae}")
    print(f"       latent_dim: {policy.config.latent_dim}")

    # 加载预处理器和后处理器
    preprocess, postprocess = make_pre_post_processors(policy.config, ckpt_path)

    # 设置为评估模式
    policy.eval()
    policy.to(cfg.device)
    policy.reset()
    print(f"[INFO] 模型已加载到 {cfg.device}，处于 eval 模式")

    # ========== 初始化平滑器 ==========
    if cfg.smooth.mode in ("smooth", "rtc"):
        if cfg.smooth.mode == "rtc":
            print("[INFO] RTC 模式已指定，当前 ACT 模型无原生 RTC 支持，降级为 TrajectorySmoother")

        smoother = TrajectorySmoother(
            ema_alpha=cfg.smooth.ema_alpha,
            pos_max_delta=cfg.smooth.pos_max_delta,
            gripper_max_delta=cfg.smooth.gripper_max_delta,
            pos_dead_zone=cfg.smooth.pos_dead_zone,
            gripper_dead_zone=cfg.smooth.gripper_dead_zone,
            window_size=cfg.smooth.window_size,
            use_window_average=cfg.smooth.use_window_average,
            use_ema=cfg.smooth.use_ema,
            use_clipping=cfg.smooth.use_clipping,
            use_dead_zone=cfg.smooth.use_dead_zone,
        )
        print(f"[INFO] TrajectorySmoother 已初始化:")
        print(f"       EMA alpha: {smoother.ema_alpha}")
        print(f"       关节限幅: ±{smoother.pos_max_delta*180/math.pi:.2f}°/步")
        print(f"       关节死区: {smoother.pos_dead_zone*180/math.pi:.3f}°")
        print(f"       滑动窗口: {smoother.window_size} 帧")
    else:
        smoother = None
        print("[INFO] 纯 ACT 推理模式（无外部平滑），使用模型内置 action queue")

    # ========== 初始化安全监控器 ==========
    safety_monitor = SafetyMonitor(
        joint_limit_rad=0.10,
        gripper_limit=0.5,
        consecutive_warnings_limit=15,
    )
    print(f"[INFO] 安全监控器已初始化")

    # ========== 机器人使能 ==========
    if cfg.robot.type == "single_piper":
        robot.piper.EnableArm(7)
        enable_fun(piper=robot.piper)
        robot.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
    else:
        raise ValueError(f"不支持的机器人类型: {cfg.robot.type}，当前仅支持 single_piper")

    time.sleep(2.0)
    np.set_printoptions(suppress=True, precision=6)
    print(f"\n[INFO] 开始推理循环，按 Ctrl+C 停止...\n")

    # ========== 统计变量 ==========
    loop_count = 0
    total_infer_time = 0.0
    total_loop_time = 0.0
    stats_interval = 100

    # ========== 主控制循环 ==========
    try:
        for _ in range(cfg.inference_time_s * int(cfg.fps)):
            start_loop_t = time.perf_counter()

            # ---- 1. 获取观测 ----
            observation = robot.get_observation()
            observation_frame = {}
            state_values = []

            for key, value in observation.items():
                if key.endswith('.pos'):
                    state_values.append(np.float32(value))
                elif isinstance(value, np.ndarray) and value.ndim == 3:
                    observation_frame[f'observation.images.{key}'] = value

            observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)

            # ---- 2. 数据预处理 ----
            for name in observation_frame:
                if "image" in name:
                    obs = observation_frame[name].astype(np.float32) / 255.0
                    obs = np.transpose(obs, (2, 0, 1))  # HWC → CHW
                else:
                    obs = observation_frame[name]
                obs = np.expand_dims(obs, axis=0)
                observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=cfg.device)

            observation_frame["task"] = [cfg.task]
            observation_frame = preprocess(observation_frame)

            # ---- 3. 模型推理 ----
            with torch.inference_mode():
                action = policy.select_action(observation_frame)

            action = postprocess(action)
            infer_time = time.perf_counter() - start_loop_t

            # ---- 4. 提取动作（7 维关节角度）----
            numpy_action = action.squeeze(0).cpu().numpy()
            joint_action = numpy_action  # shape (7,)

            # ---- 5. 轨迹平滑（如果启用）----
            if smoother is not None:
                joint_smoothed = smoother.smooth(joint_action)

                # 调试输出（每 10 步）
                if loop_count % 10 == 0:
                    raw_deg = [j * 180 / math.pi for j in joint_action[:6]]
                    smooth_deg = [j * 180 / math.pi for j in joint_smoothed[:6]]
                    diff_deg = [abs(r - s) for r, s in zip(raw_deg, smooth_deg)]
                    print(f"[DEBUG] 原始(°): {[f'{d:.2f}' for d in raw_deg]}")
                    print(f"[DEBUG] 平滑(°): {[f'{d:.2f}' for d in smooth_deg]}")
                    print(f"[DEBUG] 差异(°): {[f'{d:.3f}' for d in diff_deg]} max={max(diff_deg):.3f}°")
                    print(f"[DEBUG] 夹爪 原始={joint_action[6]:.3f} → 平滑={joint_smoothed[6]:.3f}")
            else:
                joint_smoothed = joint_action

            # ---- 6. 安全检查 ----
            is_safe, warning_msg = safety_monitor.check(joint_smoothed)
            if warning_msg:
                print(f"[WARNING] {warning_msg}")
            if not is_safe:
                print("[CRITICAL] 安全监控触发紧急停止！")
                break

            # ---- 7. 动作转换与发送 ----
            # 关节角度 rad → 0.001°
            joint_cmds = []
            for i in range(6):
                clamped_val = clamp_joint(joint_smoothed[i], JOINT_LIMITS, i)
                joint_cmds.append(int(clamped_val * RAD_TO_MILLIDEG))

            # 关节4 额外限位保护（防止自碰撞）
            if joint_cmds[4] < -70000:
                joint_cmds[4] = -70000

            # 夹爪: 比例值 → 0.001mm
            gripper_raw = joint_smoothed[6] * 70 * 1000
            #if 5000 < gripper_raw < 10000:
                # gripper_raw = max(0, gripper_raw - 5000)
            gripper_cmd = int(round(gripper_raw ))
            if gripper_cmd < 40000:
                gripper_cmd = 0
            #print(f"[DEBUG] 夹爪命令: {gripper_cmd} (原始={joint_smoothed[6]:.3f})")

            # 发送控制指令
            robot.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            robot.piper.JointCtrl(*joint_cmds)
            robot.piper.GripperCtrl(abs(gripper_cmd), 5000, 0x01, 0)  # 夹爪控制 1000代表速度，0x01代表位置模式，0代表不使用力矩限制

            # ---- 8. 统计 ----
            loop_count += 1
            total_infer_time += infer_time
            loop_dt = time.perf_counter() - start_loop_t
            total_loop_time += loop_dt

            if loop_count % stats_interval == 0:
                avg_infer = total_infer_time / loop_count * 1000
                avg_loop = total_loop_time / loop_count * 1000
                actual_freq = loop_count / total_loop_time
                joint_deg = [j * 180 / math.pi for j in joint_smoothed[:6]]
                queue_size = len(policy._action_queue) if hasattr(policy, '_action_queue') else 'N/A'
                print(f"[STATS] 步数:{loop_count:6d} | "
                      f"推理:{avg_infer:5.1f}ms | "
                      f"循环:{avg_loop:5.1f}ms | "
                      f"频率:{actual_freq:4.1f}Hz | "
                      f"队列:{queue_size} | "
                      f"关节(°):[{joint_deg[0]:.1f},{joint_deg[1]:.1f},{joint_deg[2]:.1f},"
                      f"{joint_deg[3]:.1f},{joint_deg[4]:.1f},{joint_deg[5]:.1f}] | "
                      f"夹爪:{gripper_cmd}")

            # ---- 9. 频率控制 ----
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / cfg.fps - dt_s)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断推理循环")

    finally:
        print(f"\n{'='*60}")
        print(f"[INFO] 推理结束")
        print(f"[INFO] 总步数: {loop_count}")
        if loop_count > 0:
            print(f"[INFO] 平均推理时间: {total_infer_time/loop_count*1000:.1f} ms")
            print(f"[INFO] 平均循环时间: {total_loop_time/loop_count*1000:.1f} ms")
            print(f"[INFO] 实际频率: {loop_count/total_loop_time:.1f} Hz")
        print(f"[INFO] 安全警告总数: {safety_monitor._total_warnings}")
        print(f"{'='*60}")


if __name__ == "__main__":
    inference()
