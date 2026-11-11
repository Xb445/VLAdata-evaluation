#!/usr/bin/env python
"""
================================================================================
单个 ACT 模型推理效果测试 — 基于 inference_new_joint_act.py, 对齐编排器参数
================================================================================

在 inference_new_joint_act.py 的基础上简化而来, 用于对大模型任务调度项目
(act_vlm_orchestrator_v2) 中的单个 ACT 模型做独立推理测试。

【对齐编排器的两处关键修改】
  1. 平滑系数: 使用编排器 (config.py) 的逐关节 EMA + 逐关节速率限制 + 夹爪死区
       - EMA alpha:  [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50]
       - 速率限制:   [None, 0.015, 0.015, None, None, None, None] (rad/帧, 仅 J2/J3)
       - 夹爪死区:   0.15
     (直接复用编排器的 TrajectorySmoother 类, 保证完全一致)
  2. 机械臂驱动: 使用编排器的 MIT 高跟随模式
       - MotionCtrl_2(0x01, 0x01, 100, 0xAD)  (100% 运动速度 + MIT 高跟随)
     (旧版为 30% 速度 + 0x00, 机械臂跟不上 ACT chunk 命令会造成观测-动作失同步)

【平滑开关 (后处理优化开关)】
  --smooth.mode=smooth   → 开启编排器风格平滑 (EMA+限速+夹爪死区)  [默认]
  --smooth.mode=none     → 关闭平滑, 直接输出模型原始动作 (对比用)

【退出行为 (v3 修复)】
  1. Ctrl+C 正常退出: 脚本会覆盖编排器模块导入时注册的 SIGINT/SIGTERM 处理器
     (编排器只置标志不退出, 导致 Ctrl+C 失效), 恢复 KeyboardInterrupt 中断;
  2. 退出时释放相机: 释放 OpenCV VideoCapture, 避免 /dev/video* 被占用,
     导致下次启动报 "Failed to open OpenCVCamera";
  3. 退出时机械臂处理 (--exit_mode):
       - hold   (默认): 保持使能并固定在原位 (读取当前关节下发保持)
       - disable      : 失能机械臂 (DisableArm, 依靠伺服抱闸固定)

【用法示例】
  # 观察模式 (不控制机械臂, 只推理验证模型输出)
  python -m lerobot.pi05_vlm_orchestrator.test_single_act.test_single_act_inference \
    --robot.type=single_piper --robot.id=single_piper_robot --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --ckpt_path=/home/ub/Videos/0407/model/ACT_model/model/Task_A/pretrained_model \
    --task="Grab the parcel from the table and place it on the tray" \
    --smooth.mode=smooth --dry_run True

  # 真机控制 (对齐编排器平滑 + MIT 模式)
  python -m lerobot.pi05_vlm_orchestrator.test_single_act.test_single_act_inference \
    --robot.type=single_piper --robot.id=single_piper_robot --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --ckpt_path=/home/ub/Videos/0407/model/ACT_model/model/Task_A/pretrained_model \
    --task="Grab the parcel from the table and place it on the tray"

  # 关闭平滑, 对比模型原始输出
  ... --smooth.mode=none

  # 退出时失能机械臂
  ... --exit_mode=disable

  # 跳过预移动到训练初始位姿 (默认开启: 推理前从当前位姿平滑插值移动到
  # 模型训练初始位姿, 避免机械臂迅速跳变; --preset_pose False 关闭,
  # 或 --preset_target=\"j1,..,j6,gripper\" 手动指定目标, --preset_steps/--preset_delay 调速)
  ... --preset_pose False

【预移动机制 (v4 新增)】
  - 背景: 训练时机械臂从训练初始位姿开始执行任务, 若推理时当前位姿与训练初始
    位姿差异大, 模型首个 chunk 会命令机械臂"迅速跳到"训练初始位姿 (大关节跳变)。
  - 方案: 推理循环开始前, 从当前位姿在关节空间线性插值平滑移动到训练初始位姿
    (默认 300 步 × 20ms ≈ 6s), 消除跳变。
  - 目标位姿: 默认自动从模型 normalizer 的 observation.state.mean 读取;
    可用 --preset_target 手动指定 (7 个值: j1..j6 弧度, gripper 0-1)。
  - 若与目标差异小于 --preset_min_diff_deg (默认 3°) 自动跳过。
  - --preset_pose False 可关闭; 干跑模式 (--dry_run True) 下自动不移动机械臂。

================================================================================
"""

import json
import math
import os
import signal
import sys
import time
import traceback
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

# ── 编排器 (大模型任务调度项目): 复用其 TrajectorySmoother, 平滑参数完全对齐 ──
# 注意: 导入 orchestrator 模块会在其模块级注册 SIGINT/SIGTERM 处理器 (只置标志不退出),
#       导致 Ctrl+C 失效。因此本脚本在 inference() 开头重新注册自己的信号处理器覆盖它。
from lerobot.pi05_vlm_orchestrator.orchestrator import TrajectorySmoother

# ── 机械臂 SDK ──
from piper_sdk import *


# =============================================================================
# 安全监控器 (与 inference_new_joint_act.py 一致, 非编排器状态锁)
# =============================================================================

class SafetyMonitor:
    """关节跳变安全监控器"""

    def __init__(
        self,
        joint_limit_rad: float = 0.15,
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
# 配置类
# =============================================================================

@dataclass
class SmoothConfig:
    """平滑配置 (对齐编排器 config.py 的平滑参数)"""
    # 平滑模式: "none"(关闭, 输出模型原始动作) / "smooth"(开启, 对齐编排器)
    mode: str = "smooth"

    # 夹爪死区 (归一化 [0,1]): 预测值低于此阈值强制置零 (对齐编排器 smooth_gripper_dead_zone)
    gripper_dead_zone: float = 0.15

    # 模型切换过渡平滑时长 (秒), 测试中不涉及模型切换, 保留默认即可
    transition_duration_s: float = 1.0


@dataclass
class InferenceACTConfig:
    """ACT 推理顶层配置"""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # 模型检查点路径
    ckpt_path: str = None

    # 任务描述 (文本 prompt, ACT 不使用但为兼容保留)
    task: str = "Perform the manipulation task"

    # 平滑配置
    smooth: SmoothConfig = field(default_factory=SmoothConfig)

    # ── 推理参数 ──
    fps: float = 25.0                 # 控制频率 (Hz)
    device: str = "cuda"              # 推理设备
    inference_time_s: int = 10000     # 最大推理时长 (秒)

    # 干跑模式: 只推理不发送动作 (安全验证模型输出)
    dry_run: bool = False

    # 日志目录 (终端输出双写保存)
    log_dir: str = "./act_vlm_logs"

    # 每 N 步保存一次相机帧 (诊断用, 0=不保存)
    save_frame: int = 0

    # 退出时机械臂处理: "hold"(保持使能并固定在原位, 默认) / "disable"(失能)
    exit_mode: str = "hold"

    # ── 预移动到训练初始位姿 (避免推理开始时机身迅速跳变) ──
    # 开启后, 推理循环开始前会从当前位姿平滑插值移动到训练初始位姿
    # (从模型 normalizer 的 observation.state.mean 自动读取),
    # 若与当前位姿差异小于 preset_min_diff_deg 则自动跳过。
    preset_pose: bool = True
    # 手动指定目标位姿 (7 个值: j1..j6 弧度, gripper 0-1), 为空时自动从模型读取训练初始位姿
    preset_target: str = ""
    # 插值步数 (300 步 × 0.02s ≈ 6 秒到位)
    preset_steps: int = 300
    # 每步间隔 (秒)
    preset_delay: float = 0.02
    # 与目标位姿最大关节差异小于该角度 (度) 时跳过预移动
    preset_min_diff_deg: float = 3.0

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
# 机械臂使能函数 (MIT 高跟随模式, 对齐编排器 _enable_robot)
# =============================================================================

def enable_fun(piper: C_PiperInterface, timeout: float = 5.0):
    """使能机械臂并等待使能状态确认"""
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


# 关节限位表 (弧度)
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
# 运行日志 (终端 + 文件双写)
# =============================================================================

class _Tee:
    """同时输出到终端与日志文件的流。"""

    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.logfile.write(data.replace("\r\n", "\n"))
        self.logfile.flush()

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr


def setup_file_logging(log_dir: str):
    """将 stdout/stderr 同时输出到终端和按时间戳命名的日志文件。"""
    os.makedirs(log_dir, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"single_act_{ts}.log")
    log_fp = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_ORIG_STDOUT, log_fp)
    sys.stderr = _Tee(_ORIG_STDERR, log_fp)
    print(f"[INFO] 运行日志已保存: {log_path}", flush=True)
    return log_path, log_fp


def close_file_logging(log_fp):
    """关闭日志: 先恢复原始流, 再 flush/close 文件。"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout = _ORIG_STDOUT
    sys.stderr = _ORIG_STDERR
    try:
        log_fp.close()
    except Exception:
        pass


# =============================================================================
# 退出清理辅助函数
# =============================================================================

def _raise_keyboard_interrupt(signum, frame):
    """自定义信号处理器: 恢复 KeyboardInterrupt 中断行为。

    原因: 导入 orchestrator 模块时, 其模块级注册了只置 _EMERGENCY_STOP 标志的
    SIGINT/SIGTERM 处理器, 导致 Ctrl+C 无法触发 KeyboardInterrupt (程序不退出)。
    本函数在 inference() 开头重新注册, 覆盖该行为。
    """
    raise KeyboardInterrupt


def _finalize_arm(robot, exit_mode: str):
    """退出时机械臂处理: 保持使能固定在原位 (默认) 或失能。"""
    if robot is None:
        return
    if exit_mode == "disable":
        try:
            robot.piper.DisableArm(7)
            print("[INFO] 机械臂已失能 (DisableArm, 抱闸固定)")
        except Exception as e:
            print(f"[WARN] 失能机械臂失败: {e}")
    else:
        # hold: 读取当前关节, 下发保持, 机械臂保持使能并固定在原位
        try:
            obs = robot.get_observation()
            joint_cmds = []
            for i in range(1, 7):
                v = float(obs.get(f"joint_{i}.pos", 0.0))
                joint_cmds.append(int(clamp_joint(v, JOINT_LIMITS, i - 1) * RAD_TO_MILLIDEG))
            gripper = float(obs.get("gripper.pos", 0.0))
            gripper_cmd = max(0, int(round(gripper * 70 * 1000)))
            # 保持使能并下发当前位姿 (发送数次确保到位)
            robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
            for _ in range(10):
                robot.piper.JointCtrl(*joint_cmds)
                robot.piper.GripperCtrl(abs(gripper_cmd), 5000, 0x01, 0)
                time.sleep(0.05)
            print("[INFO] 机械臂已保持使能并固定在原位")
        except Exception as e:
            print(f"[WARN] 保持机械臂原位失败: {e}")


def _release_cameras(robot):
    """释放所有相机 (避免 /dev/video* 被占用导致下次启动失败)。"""
    if robot is None:
        return
    cams = getattr(robot, "cameras", None)
    if not cams:
        return
    for name, cam in cams.items():
        try:
            if getattr(cam, "is_connected", False):
                cam.disconnect()
                print(f"[INFO] 相机已释放: {name}")
        except Exception as e:
            print(f"[WARN] 相机 {name} 断开失败: {e}")


def _disconnect_piper(robot):
    """断开 Piper CAN 连接 (关闭 CAN 端口与接收线程)。"""
    if robot is None:
        return
    piper = getattr(robot, "piper", None)
    if piper is None:
        return
    try:
        piper.DisconnectPort()
        print("[INFO] Piper CAN 已断开")
    except Exception as e:
        print(f"[WARN] Piper 断开失败: {e}")


# =============================================================================
# 预移动到训练初始位姿 (避免推理开始时机身迅速跳变)
# =============================================================================

def _get_training_pose(preprocess) -> np.ndarray | None:
    """从模型预处理管线提取训练初始位姿 (observation.state 的均值, 7 维)。

    训练数据中观测均值 ≈ 任务起始的工作位姿。若提取失败返回 None。
    """
    try:
        from lerobot.processor import NormalizerProcessorStep

        for step in getattr(preprocess, "steps", []):
            if isinstance(step, NormalizerProcessorStep):
                stats = getattr(step, "stats", None)
                if stats and "observation.state" in stats:
                    m = stats["observation.state"].get("mean")
                    if m is not None:
                        if hasattr(m, "detach"):
                            m = m.detach().cpu().numpy()
                        arr = np.asarray(m, dtype=np.float64).ravel()
                        if arr.shape[0] >= 7:
                            return arr[:7]
    except Exception as e:
        print(f"[WARN] 提取训练初始位姿失败: {e}")
    return None


def _move_to_target(robot, target: np.ndarray, steps: int = 300, delay: float = 0.02):
    """从当前位姿平滑插值移动到目标位姿 (避免迅速跳变)。

    在关节空间线性插值, 每步下发插值位姿 (MIT 高跟随模式, 对齐编排器)。
    Ctrl+C 可随时中断 (由外层 try 捕获进入清理流程)。

    Args:
        robot: SinglePiper 机器人实例
        target: 目标 7 维位姿 [j1..j6 rad, gripper 0-1]
        steps: 插值步数
        delay: 每步间隔 (秒)
    """
    target = np.asarray(target, dtype=np.float64).ravel()[:7]

    # 读取当前位姿 [j1..j6, gripper]
    obs = robot.get_observation()
    current = np.array(
        [float(obs.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)]
        + [float(obs.get("gripper.pos", 0.0))],
        dtype=np.float64,
    )

    diff_deg = np.abs((target[:6] - current[:6]) * 180.0 / np.pi)
    print(f"[INFO] 当前位姿:       {[f'{v*180/np.pi:.1f}°' for v in current[:6]]} 夹爪:{current[6]:.3f}")
    print(f"[INFO] 训练初始位姿:   {[f'{v*180/np.pi:.1f}°' for v in target[:6]]} 夹爪:{target[6]:.3f}")
    print(f"[INFO] 关节差异(°):    {[f'{d:.1f}' for d in diff_deg]} max={max(diff_deg):.1f}°")
    print(f"[INFO] 平滑移动到训练初始位姿 ({steps} 步 × {delay*1000:.0f}ms ≈ {steps*delay:.1f}s) ...")

    step_interval = max(1, steps // 10)
    for i in range(1, steps + 1):
        t = i / steps
        interp = current + (target - current) * t
        joint_cmds = []
        for j in range(6):
            v = clamp_joint(interp[j], JOINT_LIMITS, j)
            joint_cmds.append(int(v * RAD_TO_MILLIDEG))
        g_cmd = int(round(float(np.clip(interp[6], 0.0, 1.0)) * 70 * 1000))
        robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
        robot.piper.JointCtrl(*joint_cmds)
        robot.piper.GripperCtrl(abs(g_cmd), 5000, 0x01, 0)
        if delay > 0:
            time.sleep(delay)
        if i % step_interval == 0:
            print(f"[INFO] 移动进度: {t*100:.0f}%")

    obs_end = robot.get_observation()
    end = np.array(
        [float(obs_end.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)],
        dtype=np.float64,
    )
    print(f"[INFO] ✅ 已到达训练初始位姿: {[f'{v*180/np.pi:.1f}°' for v in end]}")


# =============================================================================
# 主推理函数
# =============================================================================

@parser.wrap()
def inference(cfg: InferenceACTConfig) -> None:
    """ACT 模型关节模式单模型推理测试"""

    # ========== 日志 (终端 + 文件双写) ==========
    log_path, log_fp = setup_file_logging(cfg.log_dir)

    # ========== 覆盖编排器导入时注册的信号处理器, 恢复 Ctrl+C 中断 ==========
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    # 状态变量 (供 finally 汇总)
    robot = None
    actions_fp = None
    actions_log_path = ""
    loop_count = 0
    total_infer_time = 0.0
    total_loop_time = 0.0
    wall_start = time.time()
    safety_warnings = 0
    stats_interval = 100

    try:
        # ========== 打印配置信息 ==========
        print("=" * 60)
        print("  单个 ACT 模型推理效果测试 (对齐编排器)")
        print("=" * 60)
        print(f"  检查点路径:   {cfg.ckpt_path}")
        print(f"  任务描述:     {cfg.task}")
        print(f"  控制频率:     {cfg.fps} Hz")
        print(f"  推理设备:     {cfg.device}")
        print(f"  平滑模式:     {cfg.smooth.mode} "
              f"({'对齐编排器: EMA[0.30,0.12,0.18,0.30,0.30,0.30,0.50] + 限速J2/J3 + 夹爪死区' if cfg.smooth.mode == 'smooth' else '(模型原始输出)'})")
        print(f"  干跑模式:     {'✅ (只推理不发送动作)' if cfg.dry_run else '否 (控制机械臂)'}")
        print(f"  退出行为:     {'保持使能固定在原位' if cfg.exit_mode == 'hold' else '失能机械臂'}")
        print(f"  预移到位姿:   {'✅ 平滑移动到训练初始位姿' if cfg.preset_pose and not cfg.dry_run else '关闭 (推理前不移动机械臂)'}")
        print(f"  日志目录:     {cfg.log_dir}")
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
        print(f"[INFO] ACT 模型配置: chunk_size={policy.config.chunk_size}, "
              f"n_action_steps={policy.config.n_action_steps}")

        # 加载预处理器和后处理器
        preprocess, postprocess = make_pre_post_processors(policy.config, ckpt_path)

        # 设置为评估模式
        policy.eval()
        policy.to(cfg.device)
        policy.reset()
        print(f"[INFO] 模型已加载到 {cfg.device}，处于 eval 模式")

        # ========== 初始化平滑器 (对齐编排器 TrajectorySmoother) ==========
        if cfg.smooth.mode == "smooth":
            smoother = TrajectorySmoother(
                ema_alphas=[0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50],  # 对齐编排器
                rate_limits=[None, 0.015, 0.015, None, None, None, None],  # 对齐编排器 (仅 J2/J3)
                gripper_dead_zone=cfg.smooth.gripper_dead_zone,          # 对齐编排器 0.15
                transition_duration_s=cfg.smooth.transition_duration_s,
                fps=cfg.fps,
            )
            print(f"[INFO] 平滑器已初始化 (对齐编排器):")
            print(f"       EMA alpha:   {smoother.ema_alphas}")
            print(f"       速率限制:    {smoother.rate_limits}")
            print(f"       夹爪死区:    {smoother.gripper_dead_zone}")
        else:
            smoother = None
            print("[INFO] 纯 ACT 推理模式 (无外部平滑)，直接输出模型原始动作")

        # ========== 初始化安全监控器 ==========
        safety_monitor = SafetyMonitor(
            joint_limit_rad=0.15,
            gripper_limit=0.5,
            consecutive_warnings_limit=15,
        )
        print(f"[INFO] 安全监控器已初始化 (joint_limit={safety_monitor.joint_limit_rad}rad)")

        # ========== 机器人使能 (MIT 高跟随模式, 对齐编排器) ==========
        if cfg.robot.type == "single_piper":
            robot.piper.EnableArm(7)
            enable_fun(piper=robot.piper)
            # v9 对齐编排器: 100% 运动速度 + MIT 高跟随模式 (0xAD)
            robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
            robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        else:
            raise ValueError(f"不支持的机器人类型: {cfg.robot.type}，当前仅支持 single_piper")

        time.sleep(2.0)
        np.set_printoptions(suppress=True, precision=6)

        # ========== 预移动到训练初始位姿 (避免推理开始时机身迅速跳变) ==========
        # 训练时机械臂从训练初始位姿开始, 若当前位姿与之差异大, 模型首个 chunk
        # 会命令机械臂迅速跳到该位姿。先在推理前平滑插值移动过去, 消除跳变。
        if cfg.preset_pose and not cfg.dry_run:
            # 确定目标位姿: 手动指定 (--preset_target) 或从模型 normalizer 自动读取
            target = None
            if cfg.preset_target.strip():
                try:
                    vals = [float(x) for x in cfg.preset_target.replace('，', ',').split(',')]
                    if len(vals) != 7:
                        raise ValueError(f"preset_target 需要 7 个值, 实际 {len(vals)}")
                    target = np.array(vals, dtype=np.float64)
                except Exception as e:
                    print(f"[WARN] 解析 --preset_target 失败 ({e}), 改用模型训练初始位姿")
            if target is None:
                target = _get_training_pose(preprocess)
                if target is None:
                    print("[WARN] 无法获取训练初始位姿, 跳过预移动")

            if target is not None:
                # 差异较小则跳过 (机械臂已接近训练位姿)
                obs_now = robot.get_observation()
                cur = np.array(
                    [float(obs_now.get(f"joint_{i}.pos", 0.0)) for i in range(1, 7)],
                    dtype=np.float64,
                )
                diff_deg = float(np.max(np.abs((target[:6] - cur) * 180.0 / np.pi)))
                if diff_deg < cfg.preset_min_diff_deg:
                    print(f"[INFO] 当前位姿已接近训练初始位姿 (max差异 {diff_deg:.1f}° < "
                          f"{cfg.preset_min_diff_deg:.1f}°), 跳过预移动")
                else:
                    _move_to_target(robot, target, steps=cfg.preset_steps, delay=cfg.preset_delay)

        print(f"\n[INFO] 开始推理循环，按 Ctrl+C 停止...\n")

        # ========== 逐帧动作日志 (JSONL) ==========
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        actions_log_path = os.path.join(cfg.log_dir, f"actions_{ts}.jsonl")
        actions_fp = open(actions_log_path, "w", encoding="utf-8")
        print(f"[INFO] 逐帧动作日志: {actions_log_path}")

        # 帧保存目录
        frames_dir = None
        if cfg.save_frame > 0:
            frames_dir = os.path.join(cfg.log_dir, f"frames_{ts}")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"[INFO] 相机帧保存目录: {frames_dir} (每 {cfg.save_frame} 步)")

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
                infer_start = time.perf_counter()
                with torch.inference_mode():
                    action = policy.select_action(observation_frame)
                action = postprocess(action)
                infer_time = time.perf_counter() - infer_start

                # ---- 4. 提取动作（7 维关节角度）----
                numpy_action = action.squeeze(0).cpu().numpy()
                joint_action = numpy_action  # shape (7,)

                # ---- 5. 轨迹平滑 (对齐编排器; 关闭时输出原始动作) ----
                if smoother is not None:
                    joint_smoothed = smoother.smooth(joint_action)
                else:
                    joint_smoothed = joint_action

                # 调试输出 (每 25 步)
                if loop_count % 25 == 0:
                    raw_deg = [j * 180 / math.pi for j in joint_action[:6]]
                    out_deg = [j * 180 / math.pi for j in joint_smoothed[:6]]
                    diff_deg = [abs(r - s) for r, s in zip(raw_deg, out_deg)]
                    print(f"[DEBUG] 平滑={'开' if smoother else '关'} | "
                          f"原始(°): {[f'{d:.2f}' for d in raw_deg]} 夹爪:{joint_action[6]:.3f}")
                    if smoother is not None:
                        print(f"[DEBUG]        平滑(°): {[f'{d:.2f}' for d in out_deg]} 夹爪:{joint_smoothed[6]:.3f} "
                              f"差异max={max(diff_deg):.3f}°")

                # ---- 6. 安全检查 ----
                is_safe, warning_msg = safety_monitor.check(joint_smoothed)
                if warning_msg:
                    safety_warnings += 1
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
                gripper_cmd = int(round(gripper_raw))
                if gripper_cmd < 40000:
                    gripper_cmd = 0

                # ---- 8. 记录逐帧动作 (JSONL) ----
                record = {
                    "step": loop_count,
                    "timestamp": time.time(),
                    "smooth_mode": cfg.smooth.mode,
                    "infer_ms": round(infer_time * 1000, 3),
                    "raw_action": [float(x) for x in joint_action],
                    "final_action": [float(x) for x in joint_smoothed],
                    "joint_cmds": joint_cmds,
                    "gripper_cmd": gripper_cmd,
                    "safety_warning": warning_msg,
                }
                actions_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                actions_fp.flush()

                # ---- 9. 发送控制指令 (MIT 高跟随, 对齐编排器) ----
                if not cfg.dry_run:
                    robot.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
                    robot.piper.JointCtrl(*joint_cmds)
                    robot.piper.GripperCtrl(abs(gripper_cmd), 5000, 0x01, 0)
                elif loop_count % 25 == 0:
                    out_deg = [j * 180 / math.pi for j in joint_smoothed[:6]]
                    print(f"[DRYRUN] 动作(°): {[f'{d:.1f}' for d in out_deg]} 夹爪cmd: {gripper_cmd}")

                # ---- 10. 相机帧保存 (诊断) ----
                if frames_dir is not None and loop_count % cfg.save_frame == 0:
                    try:
                        import cv2
                        for cam_name, img in observation.items():
                            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                                cv2.imwrite(
                                    os.path.join(frames_dir, f"step_{loop_count:06d}_{cam_name}.jpg"),
                                    img,
                                )
                    except Exception as e:
                        print(f"[WARNING] 保存帧失败: {e}")

                # ---- 11. 统计 ----
                loop_count += 1
                total_infer_time += infer_time
                loop_dt = time.perf_counter() - start_loop_t
                total_loop_time += loop_dt

                if loop_count % stats_interval == 0:
                    avg_infer = total_infer_time / loop_count * 1000
                    avg_loop = total_loop_time / loop_count * 1000
                    # 用墙钟计算真实控制频率 (total_loop_time 不含 busy_wait 等待)
                    wall_elapsed = time.time() - wall_start
                    actual_freq = loop_count / max(wall_elapsed, 0.001)
                    joint_deg = [j * 180 / math.pi for j in joint_smoothed[:6]]
                    queue_size = len(policy._action_queue) if hasattr(policy, '_action_queue') else 'N/A'
                    print(f"[STATS] 步数:{loop_count:6d} | "
                          f"推理:{avg_infer:5.1f}ms | "
                          f"循环:{avg_loop:5.1f}ms | "
                          f"频率:{actual_freq:4.1f}Hz | "
                          f"队列:{queue_size} | "
                          f"平滑:{cfg.smooth.mode} | "
                          f"关节(°):[{joint_deg[0]:.1f},{joint_deg[1]:.1f},{joint_deg[2]:.1f},"
                          f"{joint_deg[3]:.1f},{joint_deg[4]:.1f},{joint_deg[5]:.1f}] | "
                          f"夹爪:{gripper_cmd}")

                # ---- 12. 频率控制 ----
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

        except KeyboardInterrupt:
            print("\n[INFO] 用户中断推理循环")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断 (初始化阶段)")

    except Exception as e:
        print(f"\n[ERROR] 运行异常: {e}")
        traceback.print_exc()

    finally:
        # ========== 退出清理 ==========
        print("\n" + "=" * 60)
        print("[INFO] 正在清理并退出 ...")

        # 1. 机械臂: 保持使能固定在原位 / 失能
        _finalize_arm(robot, cfg.exit_mode)

        # 2. 释放相机 (避免 /dev/video* 被占用)
        _release_cameras(robot)

        # 3. 断开 Piper CAN
        _disconnect_piper(robot)

        # 4. 关闭逐帧动作日志
        if actions_fp is not None:
            try:
                actions_fp.close()
            except Exception:
                pass

        # 5. 最终统计 (此时日志流仍开启, 同时写入终端与文件)
        wall_elapsed = time.time() - wall_start
        print(f"\n{'='*60}")
        print(f"[INFO] 推理结束")
        print(f"[INFO] 总步数: {loop_count}")
        if loop_count > 0:
            print(f"[INFO] 平均推理时间: {total_infer_time/loop_count*1000:.1f} ms")
            print(f"[INFO] 平均循环时间: {total_loop_time/loop_count*1000:.1f} ms")
            print(f"[INFO] 实际运行时长: {wall_elapsed:.1f} s")
        print(f"[INFO] 安全警告总数: {safety_warnings}")
        if actions_log_path:
            print(f"[INFO] 动作日志: {actions_log_path}")
        print(f"[INFO] 运行日志: {log_path}")
        print(f"{'='*60}")

        # 6. 关闭日志 (最后一步, 恢复原始 stdout)
        close_file_logging(log_fp)


if __name__ == "__main__":
    inference()
