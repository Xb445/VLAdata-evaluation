"""
PI0.5 + VLM 多任务联调编排器 — 主控制循环
=========================================

核心编排逻辑:
    1. VLM 周期性观察场景 → 分类当前包裹状态
    2. 窗口投票 (5 帧中 ≥3 帧一致) → 决定执行哪个子任务
    3. 根据投票结果切换 PI0.5 模型 (Task_A/B/C/standby) + 对应训练任务文本
    4. PI0.5 后台推理 (RTC) → 机械臂执行动作
    5. 静止检测: 机械臂接近静止 3s → 触发 VLM 重评估 (保险机制)
    6. 安全监控: 关节跳变/连续异常 → 紧急停止

模型切换策略:
    - 切换时停止推理线程 + 清除 GPU 缓存 + Python GC → 消除累积误差
    - 每个 PI 模型绑定独立的训练任务文本 (切换时同步更新 task prompt)
    - standby 模式: 清空模型, 机械臂待机, 周期性 VLM 检查场景是否复原
"""

import math
import os
import sys
import time
import json
import logging
import threading
import signal
from collections import deque
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

from lerobot.pi05_vlm_orchestrator.config import (
    OrchestratorConfig,
    SUBTASK_TO_MODEL,
    SUBTASK_DESCRIPTIONS,
    TASK_PROMPTS,
    DEFAULT_TASK_PROMPT,
    PRESET_POSES,
)

from lerobot.pi05_vlm_orchestrator.vlm_client import VLMClient, VLMDecision
from lerobot.pi05_vlm_orchestrator.window_voter import WindowVoter
from lerobot.pi05_vlm_orchestrator.pi_model_manager import PIModelManager
from lerobot.pi05_vlm_orchestrator.stationary_detector import StationaryDetector
from lerobot.pi05_vlm_orchestrator.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# ============================================================================
# 复用 inference_pi05_adarms_fix.py / inference_new_joint_act_v3.py 的核心组件
# ============================================================================

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


def clamp_joint(value: float, joint_limits: list, idx: int) -> float:
    """将关节角度限制在安全范围内"""
    low, high = joint_limits[idx]
    return max(min(value, high), low)


def motion_ctrl(piper, speed: int = 100, high_follow: bool = True):
    """发送运动控制模式 (默认与 HIL 采集一致: 速度100 + MIT/high_follow 0xAD)。

    对齐 inference_pi05_adarms_fix.py: 采集时从臂 piper_follower 为
    speed_ratio=100, high_follow=True(mit_mode=0xAD)。原 (30, 0x00) 机械臂
    响应慢/模式不同 → 位置滞后 → RTC 的 prev_chunk_left_over 与实际位置错位
    → chunk 切换抖动。
    """
    mit_mode = 0xAD if high_follow else 0x00
    piper.MotionCtrl_2(0x01, 0x01, speed, mit_mode)


def move_to_preset_pose(
    piper,
    target_joints_rad,
    gripper_open: float = 1.0,
    steps: int = 100,
    step_dt: float = 0.02,
    motion_speed: int = 100,
    motion_high_follow: bool = True,
):
    """缓慢将机械臂移动到目标位姿 (任务平均初始位姿), 避免跳变。

    对齐 inference_pi05_adarms_fix.py 的 move_to_preset_pose:
        - 读取当前关节 (Piper 毫度 → 弧度, 必须用 GetArmJointMsgs())
        - 线性插值分 steps 步发送, 每步 step_dt 秒 → 平滑移动
        - 到位后打开夹爪

    Args:
        piper: C_PiperInterface 实例 (self._piper)
        target_joints_rad: 目标 6 关节 (rad)
        gripper_open: 到位后夹爪开度 (0~1)
        steps: 插值步数
        step_dt: 每步间隔 (秒)
        motion_speed: 运动速度百分比 (默认 100, 对齐 HIL 采集)
        motion_high_follow: MIT/高跟随模式 (默认 True, mit_mode=0xAD)
    """
    # 读取当前关节: 必须用 GetArmJointMsgs() (GetArmLowSpdInfoMsgs() 无 joint_state)
    joint_state = piper.GetArmJointMsgs().joint_state
    current = np.array(
        [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
         joint_state.joint_4, joint_state.joint_5, joint_state.joint_6],
        dtype=np.float64,
    ) * (0.001 * math.pi / 180.0)   # 毫度 → 弧度
    target = np.asarray(target_joints_rad, dtype=np.float64)

    if np.max(np.abs(target - current)) < 0.02:
        logger.info(f"[Preset] 🎯 已在目标位姿附近, 跳过预设移动")
    else:
        logger.info(
            f"[Preset] 🎯 缓慢移动到预设位姿 ({steps} 步 × {step_dt*1000:.0f}ms)..."
        )
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = current + (target - current) * alpha
            joints = [int(j * RAD_TO_MILLIDEG) for j in interp]
            motion_ctrl(piper, motion_speed, motion_high_follow)
            piper.JointCtrl(
                joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]
            )
            time.sleep(step_dt)
        logger.info(f"[Preset] ✅ 已到位: {np.round(target, 3)} rad")

    piper.GripperCtrl(round(gripper_open * 70 * 1000), 2000, 0x01, 0)
    logger.info(f"[Preset] 🦾 夹爪开度: {gripper_open:.1f}")


# ============================================================================
# 全局紧急停止标志
# ============================================================================
_EMERGENCY_STOP = False


def _signal_handler(signum, frame):
    global _EMERGENCY_STOP
    logger.warning(f"\n⚠️  收到信号 {signum}, 紧急停止!")
    _EMERGENCY_STOP = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 轨迹平滑器 (匹配 S0 人类在环策略 — 逐关节 EMA + 选择性速率限制)
# ============================================================================

class TrajectorySmoother:
    """
    关节空间轨迹平滑器 — 匹配 S0 人类在环实验的平滑策略。

    与 S0 完全一致的平滑流水线:
        1. 逐关节 EMA 低通滤波 (J2=0.12, J3=0.18, 其余关节=0.30, 夹爪=0.50)
        2. 选择性速率限制 (仅 J2/J3, 0.015 rad/帧)
        3. 夹爪死区 (预测值 < 0.5 → 强制置零)

    与旧版 (v1) 的关键区别:
        - ❌ 移除窗口平均 (S0 不使用)
        - ❌ 移除全局增量限幅 (S0 仅对 J2/J3 做速率限制)
        - ❌ 移除关节死区 (S0 不使用)
        - ✅ 夹爪纳入 EMA 平滑 (S0 使用 alpha=0.50)
        - ✅ 夹爪增设死区 (S0 使用 threshold=0.5 防微颤)
    """

    def __init__(
        self,
        ema_alphas: list = None,
        rate_limits: list = None,
        gripper_dead_zone: float = 0.5,
        enabled: bool = True,
        transition_duration_s: float = 2.0,
        fps: float = 30.0,
    ):
        """
        Args:
            ema_alphas: 逐维度 EMA alpha 列表 [J1, J2, J3, J4, J5, J6, gripper]
                        默认匹配 S0: [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50]
            rate_limits: 逐维度速率限制 (rad/帧), None 表示不限制
                         匹配 HIL 采集: J2/J3 0.015, 其余关节 0.05
                         (Task 模型全关节都有 chunk 首动作跳变, 全关节限幅抑制)
            gripper_dead_zone: 夹爪死区阈值 (归一化 [0,1])
            enabled: 是否启用平滑 (为 False 时直通)
            transition_duration_s: 过渡平滑时长 (秒), 模型切换后短时间内
                                   增强平滑强度, 抑制新模型起始 chunk 的冲击
            fps: 控制频率 (Hz)
        """
        # 默认 EMA alphas — 匹配 S0
        if ema_alphas is None:
            self.ema_alphas = [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50]
        else:
            self.ema_alphas = list(ema_alphas)

        # 默认速率限制 — 匹配 S0: 仅 J2/J3
        if rate_limits is None:
            self.rate_limits = [None, 0.015, 0.015, None, None, None, None]
        else:
            self.rate_limits = list(rate_limits)

        self.gripper_dead_zone = gripper_dead_zone
        self.enabled = enabled
        self.transition_duration_s = transition_duration_s
        self.fps = fps

        # 过渡平滑状态: 切换模型后短时间内增强平滑, 随时间自动恢复
        self._transition_active: bool = False
        self._transition_end_time: float = 0.0
        self._transition_duration_current: float = 0.0  # 当前生效的过渡时长

        # 内部状态: 7 维 [J1..J6, gripper]
        self._ema_action: np.ndarray | None = None   # EMA 历史值
        self._prev_action: np.ndarray | None = None   # 上一帧输出 (用于速率限制参考)
        self._initialized: bool = False

    # ========================================================================
    # 生命周期
    # ========================================================================

    def reset(self):
        """完全重置平滑器内部状态"""
        self._ema_action = None
        self._prev_action = None
        self._initialized = False
        self._transition_active = False
        self._transition_end_time = 0.0
        self._transition_duration_current = 0.0
    def begin_transition(self, duration_s: float = None):
        """
        开启过渡平滑: 短时间内增强平滑强度
        (EMA alpha 与速率限制同步收紧, 随时间线性恢复至正常值),
        抑制模型切换 / 新 chunk 起始的关节冲击, 让过渡更流畅。

        Args:
            duration_s: 过渡时长 (秒), 默认使用 switch_transition_duration_s
        """
        if duration_s is None:
            duration_s = self.transition_duration_s
        if duration_s > 0.0:
            self._transition_end_time = time.time() + duration_s
            self._transition_duration_current = duration_s
            self._transition_active = True
            logger.debug(f"[Smoother] 过渡平滑已开启 ({duration_s:.1f}s)")

    def seed(self, current_joints: np.ndarray, current_gripper: float = 0.0):
        """
        用当前关节状态初始化平滑器基线。

        用于模型切换后: 避免因内部状态为空导致第一个动作不受限幅保护
        而产生关节跳变。

        Args:
            current_joints: 当前 6 个关节角度 (rad), shape (6,)
            current_gripper: 当前夹爪值 (归一化 [0,1])
        """
        joints = np.asarray(current_joints, dtype=np.float64).ravel()[:6]
        seed_action = np.empty(7, dtype=np.float64)
        seed_action[:6] = joints
        seed_action[6] = float(current_gripper)
        self._ema_action = seed_action.copy()
        self._prev_action = seed_action.copy()
        self._initialized = True
        logger.debug(
            f"[Smoother] 已用当前关节状态初始化基线 "
            f"(joints={[f'{j*180/math.pi:.1f}°' for j in joints]})"
        )

    # ========================================================================
    # 核心平滑逻辑
    # ========================================================================

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        对单个动作进行平滑处理 — 匹配 S0 策略。

        平滑流水线 (逐维度):
            1. EMA 低通滤波 (per-joint alpha)
            2. 速率限制 (仅配置了 rate_limit 的维度)
            3. 夹爪死区 (gripper < dead_zone → 0.0)

        Args:
            action: 原始动作 [joint_1, ..., joint_6, gripper]，形状 (7,)

        Returns:
            平滑后的动作，形状 (7,)
        """
        # 禁用时直通
        if not self.enabled:
            return np.asarray(action, dtype=np.float64).copy()

        action = np.asarray(action, dtype=np.float64).copy()
        if len(action) != 7:
            raise ValueError(f"Expected action of length 7, got {len(action)}")

        # 首次调用: 直接使用原始动作初始化状态
        if not self._initialized:
            self._ema_action = action.copy()
            self._prev_action = action.copy()
            self._initialized = True
            return action.copy()

        # 过渡平滑系数: 模型切换后的短时间内增强平滑 (alpha 与速率限制
        # 同步收紧), 随时间线性恢复至正常值, 抑制新模型起始 chunk 的冲击。
        transition_scale = 1.0
        if self._transition_active:
            remaining = self._transition_end_time - time.time()
            if remaining <= 0.0:
                self._transition_active = False
            else:
                dur = self._transition_duration_current or self.transition_duration_s
                progress = 1.0 - remaining / max(dur, 1e-3)  # 0 → 1
                transition_scale = 0.5 + 0.5 * progress  # 0.5 → 1.0

        result = np.empty(7, dtype=np.float64)

        for i in range(7):
            raw_val = action[i]
            ema_alpha = self.ema_alphas[i] if i < len(self.ema_alphas) else 0.30

            # ── Step 1: EMA 低通滤波 (过渡期增强) ──
            eff_alpha = ema_alpha * transition_scale
            ema_val = eff_alpha * raw_val + (1.0 - eff_alpha) * self._ema_action[i]

            # ── Step 2: 速率限制 (仅对配置了限制的维度, 过渡期收紧) ──
            rate_limit = self.rate_limits[i] if i < len(self.rate_limits) else None
            if rate_limit is not None:
                eff_limit = rate_limit * transition_scale
                delta = ema_val - self._prev_action[i]
                if abs(delta) > eff_limit:
                    ema_val = self._prev_action[i] + (
                        eff_limit if delta > 0 else -eff_limit
                    )

            result[i] = ema_val

        # ── Step 3: 夹爪死区 ──
        if self.gripper_dead_zone > 0.0 and result[6] < self.gripper_dead_zone:
            result[6] = 0.0

        # 更新内部状态
        self._ema_action = result.copy()
        self._prev_action = result.copy()

        return result


# ============================================================================
# 夹爪过滤器 (从 inference_new_joint_act_v3.py 移植)
# ============================================================================

class GripperFilter:
    """夹爪动作后处理过滤器"""

    def __init__(
        self,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        max_open_mm: float = 70.0,
        force_close_threshold: float = 50000,
        force_close_value: float = 0,
        boost_exponent: float = 0.5,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_open_mm = max_open_mm
        self.force_close_threshold = force_close_threshold
        self.force_close_value = force_close_value
        # 开爪信号非线性放大指数 (0~1): 增强模型微弱的开爪命令 (raw 0.1→0.32),
        # 全开 1.0→1.0 不变。设为 1.0 禁用放大 (线性映射)。
        self.boost_exponent = boost_exponent

        self._clip_count = 0
        self._force_close_count = 0
        self._total_count = 0

    def process(self, gripper_raw: float) -> Tuple[int, dict]:
        self._total_count += 1
        debug = {}

        original = gripper_raw
        if gripper_raw < self.clip_min or gripper_raw > self.clip_max:
            self._clip_count += 1
            gripper_raw = np.clip(gripper_raw, self.clip_min, self.clip_max)
        debug['original'] = original
        debug['clipped'] = gripper_raw

        # 强制闭合判定: 基于未放大的原始值 (保持对噪声的抑制)。
        # mapped = raw × max_open_mm × 1000, 低于 force_close_threshold 视为噪声
        # 微颤 → 强制闭合 (0)。
        mapped_linear = gripper_raw * self.max_open_mm * 1000
        if mapped_linear < self.force_close_threshold:
            self._force_close_count += 1
            debug['force_closed'] = True
            debug['mapped'] = mapped_linear
            debug['boosted'] = 0.0
            gripper_cmd = int(round(self.force_close_value))
            debug['final'] = gripper_cmd
            return gripper_cmd, debug

        # 开爪信号非线性放大: 增强模型微弱的开爪命令 (释放包裹时 raw≈0.1~0.14),
        # 使夹爪能够可靠打开; 全开 (1.0) 不受影响。设为 1.0 时线性映射。
        if 0.0 < self.boost_exponent < 1.0 and gripper_raw > 0.0:
            gripper_raw = gripper_raw ** self.boost_exponent
        debug['boosted'] = gripper_raw

        mapped_value = gripper_raw * self.max_open_mm * 1000
        debug['mapped'] = mapped_value

        gripper_cmd = int(round(mapped_value))
        debug['final'] = gripper_cmd
        return gripper_cmd, debug

    def reset_stats(self):
        self._clip_count = 0
        self._force_close_count = 0
        self._total_count = 0

    def get_stats(self) -> dict:
        return {
            'total': self._total_count,
            'clip_count': self._clip_count,
            'force_close_count': self._force_close_count,
            'clip_ratio': self._clip_count / max(self._total_count, 1),
            'force_close_ratio': self._force_close_count / max(self._total_count, 1),
        }


# ============================================================================
# 安全监控器 (从 inference_new_joint_act_v3.py 移植)
# ============================================================================

class SafetyMonitor:
    """关节跳变安全监控器"""

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

    def check(self, action: np.ndarray) -> Tuple[bool, str]:
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


# ============================================================================
# 编排器主类
# ============================================================================

class PIVLMOrchestrator:
    """
    PI0.5 + VLM 多任务联调编排器

    整体循环:
        ┌──────────────────────────────────────────────────┐
        │  1. 获取观测 (关节状态 + 相机图像)                  │
        │  2. 静止检测: 关节变化 < 阈值 持续 N 秒             │
        │     └─ 触发 → VLM 连拍投票 (5 帧连续判断)           │
        │  3. 投票窗口满 → 检查共识 (≥3/5 票)                 │
        │  4. 共识达成 → 切换 PI0.5 模型 + 任务文本 + 重置状态  │
        │  5. PI 后台推理 (RTC) → 机械臂执行                 │
        │  6. 安全监控 → 异常急停                             │
        └──────────────────────────────────────────────────┘

    【关键设计】
      - VLM 仅在机械臂静止 ≥ N 秒后触发连拍, 运动期间零干预
      - 连拍 5 帧快速投票, 不依赖多次静止触发
      - 切换模型时: 停止推理线程 → 清 GPU 缓存 → 切换模型+任务文本 → 重启线程
      - PI 推理为后台线程 + ActionQueue + RTC 对齐 (对齐 inference_pi05_adarms_fix.py)

    模型映射:
        subtask1 (桌面)      → Task_A (抓取到条码扫描台)
        subtask2 (托盘,条码×) → Task_B (翻转包裹)
        subtask3 (托盘,条码✓) → Task_C (放入盒子)
        subtask4 (盒子)      → standby (待机等待复原)
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._running = False
        self._paused = False

        # ── 子系统 ──
        self.vlm_client = VLMClient(config)
        self.voter = WindowVoter(config)
        self.model_manager = PIModelManager(config)
        self.stationary_detector = StationaryDetector(config)
        self.camera_manager = CameraManager(config)

        # ── 机器人 ──
        self._robot = None
        self._piper = None

        # ── 平滑 & 过滤 ──
        self._smoother: Optional[TrajectorySmoother] = None
        self._gripper_filter: Optional[GripperFilter] = None
        self._safety_monitor: Optional[SafetyMonitor] = None

        # ── 诊断日志 (对齐 inference_pi05_adarms_fix.py 的 DiagLogger) ──
        self._diag = None

        # ── 流匹配 EMA / 相邻跳变 (发送动作后处理) ──
        self._last_flow_action: Optional[np.ndarray] = None  # 流匹配 EMA 上一帧
        self._last_sent_action: Optional[np.ndarray] = None  # 相邻步跳变检测
        self._last_gripper_ratio: float = 1.0  # 夹爪上一帧发送比例 (预设开1.0)
        self._gripper_state: str = "open"      # 夹爪二值滞回状态
        self._gripper_dwell: int = 0           # 当前夹爪状态已持续步数
        self._t_start_run: float = 0.0                       # 诊断: 运行时刻

        # ── 状态追踪 ──
        self._loop_count: int = 0
        self._current_subtask: str = ""
        self._current_model: str = ""
        self._last_vlm_check_step: int = 0
        self._last_switch_step: int = 0
        self._standby_start_time: float = 0.0
        self._last_standby_log_time: float = 0.0  # 待机日志输出控制
        self._subtask_start_time: float = 0.0   # 当前子任务开始时间 (用于超时判定)
        self._subtask_timeout_triggered: bool = False  # 当前子任务是否已触发过超时
        self._switch_protection_until: float = 0.0  # 模型切换保护截止时间 (禁止 VLM 连拍)
        self._total_infer_time: float = 0.0
        self._total_loop_time: float = 0.0
        # 墙钟起点: 用于计算真实控制频率。
        # 注意: _total_loop_time 只累计循环内计算耗时 (不含 busy_wait 等待),
        # 用它算频率会得到 ~150Hz 的假象 (实际控制频率只有 ~30Hz)。
        self._wall_start_time: float = time.time()

        # 模型输出显著性诊断: 记录连续"输出≈保持姿态"的步数,
        # 用于定位"机械臂无明显动作"是模型问题还是链路问题。
        self._near_hold_count: int = 0

        # 保持锁定 (抑制任务完成/复位后的微振荡):
        # 当模型输出连续多步都在当前关节的小范围内 (处于"保持"状态) 时,
        # 将关节锁定到参考位置, 抑制 ACT chunk 重预测带来的小幅振荡。
        # 模型输出超出保持带时自动解除锁定 (不影响真实运动)。
        self._hold_lock_active: bool = False
        self._hold_lock_target: Optional[np.ndarray] = None  # (6,) 锁定目标 (rad)
        self._hold_lock_count: int = 0
        # 保持窗口原始参考 (6,): 进入保持窗口时的模型原始目标, 用于判别
        # "模型在保持" (目标稳定) 还是 "模型在运动" (目标快速漂移)。
        self._hold_raw_ref: Optional[np.ndarray] = None

        # 最近一次 VLM 连拍是否触发了模型切换。
        # 用于快速重评估: 连拍后未切换 → 启用 fast_recheck 加速下一轮重评估;
        # 切换成功 → 恢复正常阈值。
        self._last_burst_switched: bool = False

        # ── VLM 连拍投票状态 ──
        self._burst_active: bool = False       # 是否正在连拍
        self._burst_count: int = 0             # 已连拍次数
        self._burst_next_time: float = 0.0     # 下次连拍时间
        self._burst_last_result: str = ""      # 连拍中上一次 VLM 结果
        self._burst_start_time: float = 0.0    # 连拍开始时间 (用于整体超时保护)
        self._burst_abort_on_motion: bool = True  # 连拍期间检测到运动是否中止 (子任务超时连拍为 False)

        # ── 日志 ──
        self._log_dir = config.log_dir
        os.makedirs(self._log_dir, exist_ok=True)
        self._decision_log: list = []

        # ── 性能统计 ──
        self._vlm_call_times: deque = deque(maxlen=100)
        self._act_infer_times: deque = deque(maxlen=1000)

    # ========================================================================
    # 初始化
    # ========================================================================

    def initialize(self) -> bool:
        """
        初始化所有子系统:
            1. 连接相机
            2. 连接机械臂
            3. 预加载 ACT 模型
            4. 初始化平滑/过滤器

        Returns:
            True 如果初始化成功
        """
        logger.info("=" * 70)
        logger.info("  ACT + VLM 多任务联调编排器 — 初始化")
        logger.info("=" * 70)

        # ── 1. 连接相机 ──
        if not self.camera_manager.connect():
            logger.error("❌ 相机连接失败, 无法继续")
            return False
        self.camera_manager.start_async()
        time.sleep(1.0)  # 等待异步线程产出第一批帧

        # ── 2. 连接机械臂 (不含相机, 相机由 CameraManager 管理) ──
        try:
            from lerobot.robots import make_robot_from_config, single_piper  # noqa: F401
            from lerobot.robots.single_piper.config_single_piper import SinglePiperConfig

            # 不在 SinglePiperConfig 中配置 cameras,
            # 因为 CameraManager 已经管理了相机生命周期,
            # 避免同一物理相机被两个模块同时打开导致冲突。
            robot_cfg = SinglePiperConfig(
                port=self.config.robot_port,
                ctrl_mode=self.config.robot_ctrl_mode,
                cameras={},  # 空字典: robot 不管理相机
            )

            self._robot = make_robot_from_config(robot_cfg)
            if not self._robot.is_connected:
                self._robot.connect()
            self._piper = self._robot.piper
            logger.info(f"[Robot] ✅ 机械臂已连接: {self.config.robot_type}")

        except Exception as e:
            logger.error(f"❌ 机械臂连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # ── 3. 使能机械臂 ──
        self._enable_robot()

        # ── 3.5 移到默认任务的平均初始位姿 (训练均值, 让初始状态落在训练分布内) ──
        self._move_to_task_preset(self.config.default_model)

        # ── 3.6 注入机器人/相机引用给 PI 模型管理器 (后台推理线程使用) ──
        self.model_manager.set_robot_refs(self._robot, self.camera_manager)

        # ── 3.7 创建诊断日志并注入模型管理器 ──
        from lerobot.pi05_vlm_orchestrator.diag_logger import DiagLogger
        self._diag = DiagLogger(log_dir=self.config.log_dir, enabled=self.config.diag_enabled)
        self.model_manager.set_diag(self._diag)

        # ── 4. 预加载 PI0.5 模型 ──
        logger.info("[Init] 预加载 PI0.5 模型...")
        preload_results = self.model_manager.preload_all()
        for model_name, success in preload_results.items():
            if not success:
                logger.warning(f"[Init] ⚠️ 模型 {model_name} 预加载失败, 将在需要时重试")

        # ── 5. 初始化平滑器 (匹配 S0 人类在环策略) ──
        if self.config.smooth_enabled:
            self._smoother = TrajectorySmoother(
                ema_alphas=self.config.smooth_ema_alphas,
                rate_limits=self.config.smooth_rate_limits,
                gripper_dead_zone=self.config.smooth_gripper_dead_zone,
                transition_duration_s=self.config.switch_transition_duration_s,
                fps=self.config.fps,
            )
            logger.info(
                "[Init] TrajectorySmoother 已初始化 (HIL/S0 纯 EMA 策略: "
                f"EMA={self.config.smooth_ema_alphas}, "
                f"rate_limits={self.config.smooth_rate_limits}, "
                f"gripper_dead_zone={self.config.smooth_gripper_dead_zone}, "
                f"transition_duration={self.config.switch_transition_duration_s}s)"
            )

        # ── 6. 初始化夹爪过滤器 ──
        self._gripper_filter = GripperFilter(
            clip_min=self.config.gripper_clip_min,
            clip_max=self.config.gripper_clip_max,
            max_open_mm=self.config.gripper_max_open_mm,
            force_close_threshold=self.config.gripper_force_close_threshold,
            force_close_value=self.config.gripper_force_close_value,
            boost_exponent=self.config.gripper_boost_exponent,
        )
        logger.info("[Init] GripperFilter 已初始化 "
                    f"(force_close_threshold={self.config.gripper_force_close_threshold}, "
                    f"boost_exponent={self.config.gripper_boost_exponent})")

        # ── 7. 初始化安全监控器 ──
        self._safety_monitor = SafetyMonitor(
            joint_limit_rad=self.config.joint_limit_rad,
            gripper_limit=self.config.safety_gripper_limit,
            consecutive_warnings_limit=self.config.consecutive_warnings_limit,
        )
        logger.info("[Init] SafetyMonitor 已初始化")

        logger.info("=" * 70)
        logger.info("  ✅ 初始化完成, 准备开始联调循环")
        logger.info(f"  默认模型: {self.config.default_model}")
        logger.info(f"  RTC:      {'启用' if self.config.rtc_enabled else '禁用'} "
                     f"(horizon={self.config.rtc_execution_horizon}, "
                     f"guidance={self.config.rtc_max_guidance_weight})")
        logger.info(f"  确定性采样: {'启用(零噪声)' if self.config.pi_deterministic_noise else '禁用(随机流匹配)'}")
        logger.info(f"  投票窗口: {self.config.vote_window_size} 帧, "
                     f"阈值: {self.config.vote_consensus_threshold}")
        logger.info(f"  静止检测: {self.config.stationary_time_threshold}s @ "
                     f"{self.config.stationary_joint_threshold} rad")
        timeout_str = (f"{self.config.subtask_timeout}s"
                       if self.config.subtask_timeout_enabled and self.config.subtask_timeout > 0
                       else "禁用")
        logger.info(f"  子任务超时: {timeout_str}")
        logger.info(f"  切换保护:   {self.config.switch_cooldown}s")
        logger.info(f"  VLM 服务: {self.config.vlm_base_url}")
        logger.info(f"  PI 模型: {self.config.model_path}")
        logger.info("=" * 70)

        return True

    def _enable_robot(self):
        """使能机械臂"""
        from piper_sdk import C_PiperInterface

        logger.info("[Robot] 使能机械臂...")
        self._piper.EnableArm(7)

        # 等待使能确认
        enable_flag = False
        start_time = time.time()
        timeout = 5.0

        while not enable_flag:
            elapsed = time.time() - start_time
            try:
                info = self._piper.GetArmLowSpdInfoMsgs()
                enable_flag = (
                    info.motor_1.foc_status.driver_enable_status and
                    info.motor_2.foc_status.driver_enable_status and
                    info.motor_3.foc_status.driver_enable_status and
                    info.motor_4.foc_status.driver_enable_status and
                    info.motor_5.foc_status.driver_enable_status and
                    info.motor_6.foc_status.driver_enable_status
                )
            except Exception:
                pass

            logger.info(f"[Robot] 使能状态: {enable_flag}")
            self._piper.EnableArm(7)
            self._piper.GripperCtrl(0, 1000, 0x01, 0)

            if elapsed > timeout:
                logger.error("[Robot] ❌ 使能超时!")
                sys.exit(1)
            time.sleep(1)

        # 设置控制模式
        # 对齐 inference_pi05_adarms_fix.py: 速度100 + MIT/高跟随 (0xAD),
        # 与 HIL 采集一致, 避免机械臂响应慢导致 RTC 对齐错位。
        motion_ctrl(self._piper, self.config.robot_motion_speed,
                    self.config.robot_motion_high_follow)
        self._piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        time.sleep(2.0)
        logger.info("[Robot] ✅ 机械臂使能完成")

    # ========================================================================
    # 主控制循环
    # ========================================================================

    def run(self):
        """启动主控制循环"""
        if not self._running:
            self._running = True
            self._paused = False
            self._main_loop()

    def _main_loop(self):
        """主编排循环"""
        global _EMERGENCY_STOP

        logger.info("\n" + "=" * 70)
        logger.info("  🚀 PI0.5 + VLM 联调循环开始")
        logger.info("  按 Ctrl+C 停止")
        logger.info("=" * 70 + "\n")

        total_steps = self.config.inference_time_s * int(self.config.fps)
        np.set_printoptions(suppress=True, precision=6)

        # 墙钟计时起点: 放在主循环开始处, 排除模型加载/相机连接等初始化耗时
        # (若在 __init__ 设置, 初始化耗时会被计入, 导致早期频率统计偏低)
        self._wall_start_time = time.time()
        self._t_start_run = time.time()   # 诊断: 运行时刻
        self._last_flow_action = None     # 流匹配 EMA 上一帧
        self._last_sent_action = None     # 相邻步跳变检测

        # ── 首次场景评估: 启动 VLM 连拍投票, 确认正确模型后再执行 ACT ──
        # 旧版只做 1 次 VLM 判断 (无法填满 5 票窗口达成共识), 机械臂会立即用
        # 默认模型执行, 若场景需要其它子任务则产生错误动作 (无动作/轻微抬升又放下)。
        # 改为连拍投票: 共识确认前 ACT 推理被暂停 (阶段5跳过), 机械臂保持静止,
        # 确认正确的 ACT 模型后才开始执行。
        logger.info("[Orchestrator] 执行首次场景评估 (VLM 连拍投票确认模型)...")
        self._start_vlm_burst()

        try:
            for step in range(total_steps):
                if _EMERGENCY_STOP:
                    logger.warning("[Orchestrator] ⚠️ 紧急停止触发, 退出循环")
                    break

                start_loop_t = time.perf_counter()

                # ================================================
                # 阶段1: 获取观测 (关节状态 + 相机图像 — 统一从帧缓存获取)
                # ================================================
                try:
                    joint_obs = self._robot.get_observation()
                except Exception as e:
                    logger.error(f"[Orchestrator] 获取关节观测失败: {e}, 跳过本步")
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                state_values = []
                for key, value in joint_obs.items():
                    if key.endswith('.pos'):
                        state_values.append(np.float32(value))

                if len(state_values) < 6:
                    logger.warning(
                        f"[Orchestrator] 关节数据不完整: 期望≥6, "
                        f"实际={len(state_values)}, 跳过本步"
                    )
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                current_joints = np.array(state_values, dtype=np.float64)

                # 统一从帧缓存获取图像 (ACT 和 VLM 共用同一帧源, 避免竞争)
                top_img, wrist_img, frame_ts = self.camera_manager.get_latest_frames()

                # 检查帧新鲜度
                if not self.camera_manager.is_frame_fresh():
                    logger.warning(
                        f"[Orchestrator] 相机帧过期 "
                        f"(top_age={time.time() - frame_ts:.2f}s), 跳过本步"
                    )
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                if top_img is None or wrist_img is None:
                    logger.warning("[Orchestrator] 相机帧缺失, 跳过本步")
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                # 构建 ACT 观测帧 (关节 + 图像, 同一时刻采集)
                observation_frame = {
                    'observation.state': np.array(state_values, dtype=np.float32),
                    'observation.images.top': top_img,
                    'observation.images.wrist': wrist_img,
                }

                # ================================================
                # 阶段2: 静止检测 + VLM 连拍投票
                #   切换保护: 模型刚切换后禁止触发 VLM 连拍,
                #   给新模型足够的执行时间, 避免"一静止就切"
                # ================================================
                if self._burst_active:
                    # 连拍模式: 持续监测机械臂运动, 若恢复运动则中止连拍。
                    # 修复"机械臂还在运动却触发状态切换"——连拍仅应在机械臂
                    # 真正静止时进行, 运动中采集的图像无法可靠判断场景。
                    if (self.config.burst_abort_on_motion
                            and self._burst_abort_on_motion
                            and self.stationary_detector.is_moving(current_joints)):
                        logger.warning(
                            "[Orchestrator] ⚠️ 连拍期间检测到机械臂运动, 中止连拍投票 "
                            "(避免在机械臂运动中切换任务)"
                        )
                        self._finish_vlm_burst(aborted=True)
                    else:
                        self._vlm_burst_tick()
                elif time.time() < self._switch_protection_until:
                    # 处于切换保护期, 完全跳过静止检测
                    # 保护期结束时由下方逻辑自动 full_reset
                    pass
                else:
                    # 保护期刚结束 → 重置静止检测器基准 (避免旧数据误触发)
                    if self._switch_protection_until > 0:
                        self._switch_protection_until = 0.0
                        self.stationary_detector.full_reset()
                        logger.debug("[Orchestrator] 🛡️ 切换保护期结束, 静止检测器已重置")

                    # 正常模式: 更新静止检测
                    if self.stationary_detector.update(current_joints):
                        logger.info(
                            f"[Orchestrator] 🔔 机械臂静止 {self.config.stationary_time_threshold:.0f}s, "
                            f"启动 VLM 连拍投票"
                        )
                        self._start_vlm_burst()

                # ================================================
                # 阶段3: 检查连拍结果, 有共识则切换
                #   注意: 待机模式下跳过此阶段, 由 _handle_standby() 统一处理
                #   避免共识被消费后 _handle_standby 无法感知场景复原
                # ================================================
                if not self.model_manager.is_standby:
                    if self._burst_active and self.voter.is_window_full():
                        consensus = self.voter.get_consensus()
                        if consensus is not None:
                            subtask, target_model = consensus
                            self._handle_consensus(subtask, target_model)
                            self._finish_vlm_burst()
                        else:
                            # 窗口满但无共识 → 清空窗口重来
                            logger.info("[Orchestrator] 连拍窗口满但无共识, 清空窗口等待下次触发")
                            self.voter.reset()
                            self._finish_vlm_burst()

                # ================================================
                # 阶段4: 待机模式处理
                # ================================================
                if self.model_manager.is_standby:
                    self._handle_standby()
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                # ================================================
                # 阶段4.5: 子任务超时检测
                # ================================================
                if (self.config.subtask_timeout_enabled
                        and self.config.subtask_timeout > 0
                        and self._subtask_start_time > 0
                        and self._current_subtask
                        and not self._burst_active
                        and not self._subtask_timeout_triggered):
                    subtask_elapsed = time.time() - self._subtask_start_time
                    if subtask_elapsed >= self.config.subtask_timeout:
                        logger.warning(
                            f"[Orchestrator] ⏰ 子任务超时! "
                            f"当前子任务={self._current_subtask} "
                            f"已执行 {subtask_elapsed:.1f}s (阈值={self.config.subtask_timeout}s), "
                            f"触发 VLM 重评估"
                        )
                        self._subtask_timeout_triggered = True
                        # 子任务超时连拍: 机械臂可能处于循环运动中,
                        # 不启用运动中止 (需要强制重评估), 并先静置 1s 再取帧。
                        self._start_vlm_burst(abort_on_motion=False, settle_time_s=1.0)

                # ================================================
                # 阶段5: PI 模型推理
                #   注意: VLM 连拍期间暂停取动作, 因为:
                #     1) 连拍由静止检测触发, 手臂已停止运动, PI 输出无意义
                #     2) VLM 需要清晰、无机械臂遮挡的场景图像
                #   PI 动作为后台线程 + ActionQueue (RTC) 模式:
                #     - 队列非空 → 取一个动作执行
                #     - 队列为空 (后台正在预测新 chunk) → 保持当前姿态等待
                # ================================================
                if self._burst_active:
                    # VLM 连拍进行中, 跳过 PI 动作, 仅做频率控制
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                # ── chunk 边界检测: 队列耗尽 → 本次推理会重新预测新 chunk ──
                boundary_detected = self.model_manager.is_new_chunk()

                try:
                    infer_start = time.perf_counter()
                    numpy_action = self.model_manager.infer_step(
                        observation_frame,
                        task_prompt=self._get_task_prompt(),
                    )
                    infer_time = time.perf_counter() - infer_start
                    self._total_infer_time += infer_time
                    self._act_infer_times.append(infer_time)
                except Exception as e:
                    logger.error(f"[Orchestrator] PI 推理失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                # 队列为空 (后台推理进行中) → 保持当前姿态, 等待下一帧
                if numpy_action is None:
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / self.config.fps - dt_s)
                    continue

                # ── chunk 边界诊断: 新 chunk 首动作相对上一动作的跳变 ──
                # 用于定位"执行一段时间机械臂抖动一下"的来源 (chunk 边界 vs
                # 连拍暂停)。DEBUG 级别, 频率 = 每 chunk 一次。
                if boundary_detected and getattr(self, "_prev_raw_action", None) is not None:
                    raw_jump = numpy_action[:6] - self._prev_raw_action[:6]
                    logger.debug(
                        "[ChunkBnd] 新 chunk 边界: 原始动作跳变(°)="
                        f"{[f'{j*180/math.pi:.1f}' for j in raw_jump]}"
                    )

                # ── 模型输出显著性诊断 ──
                # 若模型输出与当前关节几乎一致 (保持姿态), 说明模型在当前
                # 场景下没有产生任务动作 (场景不匹配训练分布 / 模型问题),
                # 而非 orchestrator 调度或控制链路问题。
                # 每累计 200 步 (约 7s) 提示一次, 避免刷屏。
                if len(current_joints) >= 6:
                    joint_delta = float(np.max(np.abs(numpy_action[:6] - current_joints[:6])))
                    if joint_delta < 0.015:  # < ~0.86°
                        self._near_hold_count += 1
                        # 仅持续保持 ~100 步 (约 3~4s) 才提示一次, 避免
                        # "模型短暂保持→又运动→再保持" 的交替造成刷屏
                        if self._near_hold_count == 100:
                            logger.warning(
                                f"[Diagnose] ⚠️ 模型连续 {self._near_hold_count} 步输出接近保持姿态 "
                                f"(最大关节差异 {joint_delta*180/math.pi:.2f}°)。"
                                f"模型={self.model_manager.active_model}, "
                                f"subtask={self._current_subtask}。"
                                f"可能原因: ①模型本身在当前场景输出保持 (模型质量问题); "
                                f"②场景/包裹位置与训练分布不匹配。"
                            )
                            self._save_scene_snapshot(joint_delta)
                    else:
                        if self._near_hold_count >= 100:
                            logger.info(
                                f"[Diagnose] 模型恢复有效动作 (此前保持姿态 {self._near_hold_count} 步)"
                            )
                        self._near_hold_count = 0
                        # 模型恢复有效动作 (机械臂将运动) → 恢复正常重评估阈值
                        self.stationary_detector.set_fast_recheck(False)

                # 原始动作 (平滑前) 调试日志: 每 50 步一条, 便于对比
                # "模型输出小" 还是 "平滑过度"。
                if self._loop_count % 50 == 0 and self.config.debug_verbose:
                    raw_deg = [j * 180 / math.pi for j in numpy_action[:6]]
                    logger.debug(
                        f"[Action] 原始动作(°): {[f'{d:.2f}' for d in raw_deg]}, "
                        f"夹爪: {numpy_action[6]:.3f}"
                    )

                # ================================================
                # 阶段6: 轨迹平滑
                # ================================================
                if self._smoother is not None:
                    joint_smoothed = self._smoother.smooth(numpy_action)
                else:
                    joint_smoothed = numpy_action

                # ── chunk 边界诊断: 平滑后 sent 相对上一发送的单步变化 ──
                if boundary_detected and getattr(self, "_prev_sent", None) is not None:
                    sent_delta = joint_smoothed[:6] - self._prev_sent[:6]
                    logger.debug(
                        "[ChunkBnd] 新 chunk 边界: sent 单步变化(°)="
                        f"{[f'{j*180/math.pi:.2f}' for j in sent_delta]}"
                    )

                # ── 保持锁定 (抑制任务完成/复位后的微振荡) ──
                # ACT 在"保持位姿"时, 每次 chunk 重预测的目标略有偏移 (实测
                # ~0.5~2°), 机械臂跟随产生小幅振荡。当模型输出连续多步都在当前
                # 关节的保持带内 (即处于"保持"状态) 时, 将关节锁定到参考位置;
                # 模型输出超出保持带时自动解除锁定 (不影响真实运动)。
                if len(current_joints) >= 6:
                    joint_smoothed = self._apply_hold_lock(
                        current_joints, numpy_action, joint_smoothed)

                # ================================================
                # 阶段7: 安全检查
                # ================================================
                is_safe, warning_msg = self._safety_monitor.check(joint_smoothed)
                if warning_msg:
                    logger.warning(f"[Safety] {warning_msg}")
                if not is_safe:
                    logger.critical("[Safety] ❌ 安全监控触发紧急停止!")
                    break

                # ================================================
                # 阶段7.5: 流匹配抖动抑制 (仅流匹配模式)
                #   流匹配每次采样噪声随机, chunk 间输出不连续; 对发送动作
                #   做轻量 EMA (alpha 高、滞后小), 抑制高频抖动。
                #   确定性采样下不启用 (保持原输出)。
                # ================================================
                if (self.config.flow_ema_alpha > 0
                        and not self.config.pi_deterministic_noise):
                    # ★ 同步 flowmatch 调优: 仅对关节做 EMA, 夹爪交给
                    #   _send_action 的二值滞回控制器 (避免 EMA 拖慢闭合)。
                    if self._last_flow_action is not None:
                        joint_smoothed[:6] = (
                            self.config.flow_ema_alpha * joint_smoothed[:6]
                            + (1.0 - self.config.flow_ema_alpha) * self._last_flow_action[:6]
                        )
                    self._last_flow_action = joint_smoothed.copy()

                # ================================================
                # 阶段8: 动作转换与发送
                # ================================================
                _sent_gripper_cmd = self._send_action(joint_smoothed)

                # 记录上一步原始/发送动作 (供 chunk 边界诊断对比)
                self._prev_raw_action = numpy_action.copy()
                self._prev_sent = joint_smoothed.copy()

                # ── 8.5 相邻步跳变检测 (定位 chunk 衔接跳变) ──
                # ★ 关节与夹爪分开计算 (同步 flowmatch 调优): 夹爪是 0~1 比例,
                #   不能混进"度"的关节指标 (否则夹爪开合把 [JUMP] 刷屏)。
                #   关节: chunk 内相邻步应 <2°; 若某步突然 >阈值, 大概率是
                #   chunk 边界 (拼接点对齐不准)。夹爪单独用 [GRIP] 报警。
                step_jump_deg = 0.0     # 关节跳变 (度)
                gripper_jump = 0.0      # 夹爪跳变 (0~1 比例, 基于实际发送值)
                if self._last_sent_action is not None:
                    step_jump_deg = float(
                        np.max(np.abs(joint_smoothed[:6] - self._last_sent_action[:6]))
                        * 180.0 / math.pi
                    )
                    gripper_jump = float(
                        abs(_sent_gripper_cmd
                            / (self.config.gripper_max_open_mm * 1000)
                            - self._last_gripper_ratio)
                    )
                self._last_sent_action = joint_smoothed.copy()
                self._last_gripper_ratio = (
                    _sent_gripper_cmd / (self.config.gripper_max_open_mm * 1000)
                )
                if step_jump_deg > self.config.jump_alert_deg:
                    logger.warning(
                        f"[JUMP] ⚠️ 步{self._loop_count + 1} 关节跳变 {step_jump_deg:.1f}° "
                        f"(qsize={self.model_manager.queue_size})"
                    )
                if gripper_jump > self.config.gripper_jump_alert:
                    logger.warning(
                        f"[GRIP] ⚠️ 步{self._loop_count + 1} 夹爪跳变 {gripper_jump:.3f} "
                        f"(qsize={self.model_manager.queue_size})"
                    )

                # 诊断日志: 间隔 10 步写入 exec CSV
                if self._diag is not None and self._loop_count % 10 == 0:
                    self._diag.log_exec(
                        self._loop_count,
                        time.time() - self._t_start_run,
                        joint_smoothed[:6] * 180.0 / math.pi,
                        int(_sent_gripper_cmd),
                        step_jump_deg,
                    )

                # ================================================
                # 阶段9: 统计与日志
                # ================================================
                self._loop_count += 1
                loop_dt = time.perf_counter() - start_loop_t
                self._total_loop_time += loop_dt

                if self._loop_count % 100 == 0:
                    self._print_stats()

                # ================================================
                # 阶段10: 频率控制
                # ================================================
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / self.config.fps - dt_s)

        except KeyboardInterrupt:
            logger.info("\n[Orchestrator] 用户中断")

        except Exception as e:
            logger.error(f"[Orchestrator] ❌ 未处理的异常: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._shutdown()

    # ========================================================================
    # VLM 检查与投票 (单次)
    # ========================================================================

    def _vlm_check_and_vote(self, top_img=None, wrist_img=None):
        """
        执行一次 VLM 场景分类并加入投票窗口。

        调用时机:
          - 启动时: 初始场景评估 (传入 None, 从缓存获取)
          - VLM 连拍中: 静止确认后连续多帧判断
          - 待机模式: 周期性检查场景复原

        Args:
            top_img: 顶部图像 (可选, 为 None 时从帧缓存获取)
            wrist_img: 腕部图像 (可选, 为 None 时从帧缓存获取)
        """
        self._last_vlm_check_step = self._loop_count

        # 获取相机图像: 优先使用传入的帧, 否则从缓存获取
        if top_img is None or wrist_img is None:
            top_img, wrist_img, _ = self.camera_manager.get_latest_frames()

        # 帧新鲜度检查: 过期帧不用于 VLM 分类。
        # 应对相机流延迟/卡顿, 避免用陈旧图像判断场景导致误切换。
        if not self.camera_manager.is_frame_fresh(max_age=self.config.vlm_frame_max_age):
            logger.warning(
                f"[Orchestrator] VLM 检查: 相机帧过期 (>{self.config.vlm_frame_max_age:.1f}s), 跳过本次"
            )
            return

        if top_img is None or wrist_img is None:
            logger.warning("[Orchestrator] VLM 检查: 相机图像获取失败, 跳过")
            return

        # 调用 VLM
        decision = self.vlm_client.classify(top_img, wrist_img)

        # 记录决策
        self._decision_log.append({
            "step": self._loop_count,
            "timestamp": time.time(),
            "subtask": decision.subtask,
            "parcel_location": decision.parcel_location,
            "label_visible": decision.label_visible,
            "target_model": decision.target_model,
            "success": decision.success,
        })

        # 添加到投票窗口
        consensus_reached = self.voter.add_decision(decision)

        if decision.success:
            desc = SUBTASK_DESCRIPTIONS.get(decision.subtask, "未知")
            logger.info(
                f"[VLM] 场景分类: {desc} "
                f"(location={decision.parcel_location}, "
                f"label_visible={decision.label_visible})"
            )
        else:
            logger.warning(f"[VLM] 分类失败: {decision.error_message}")

        # 打印窗口状态
        ws = self.voter.window_status
        logger.debug(f"[Voter] 窗口: {ws}")

    # ========================================================================
    # 保持锁定 (抑制任务完成/复位后的微振荡)
    # ========================================================================

    def _apply_hold_lock(
        self,
        current_joints: np.ndarray,
        raw_action: np.ndarray,
        joint_smoothed: np.ndarray,
    ) -> np.ndarray:
        """保持锁定: 模型输出连续多步处于"保持"状态时, 冻结关节。

        背景: ACT 在"保持位姿"时, 命令目标稳定但机械臂存在 ±1° 量级的
        物理伺服抖动/慢漂移 (2026-08-06 实测), 静止检测因此永不触发, 模型
        切换流程卡死。保持锁定在"模型确实在保持"时把关节冻结到目标位置,
        让机械臂真正静止, 恢复"静止→VLM 重评估→切换"流程。

        判别 (双条件, 缺一不可):
          - raw_drift: 原始目标相对保持窗口参考的位移 < smooth_hold_raw_band_rad
            → 模型目标稳定 (真·保持); 目标快速漂移 = 模型在运动, 不锁定。
          - hold_delta: 原始目标与当前关节差异 < smooth_hold_band_rad
            → 机械臂已接近目标 (容忍伺服抖动/慢漂移的物理滞后)。

        机制:
          - 双条件同时满足 → 计数 +1; 否则重置计数并重开保持窗口;
          - 计数 ≥ smooth_hold_steps → 锁定到当前平滑位置 (夹爪不锁);
          - 锁定期: 模型目标重新移动 (raw_drift ≥ 2×带) 或目标脱离机械臂
            (hold_delta ≥ 2×带) → 解除锁定, 跟随真实运动。

        返回: 可能被覆盖的 joint_smoothed (锁定期关节被冻结)。
        """
        hold_band = self.config.smooth_hold_band_rad
        if hold_band <= 0:
            # 保持锁定禁用
            return joint_smoothed

        raw_band = getattr(self.config, "smooth_hold_raw_band_rad", 0.02)

        # 保持窗口参考: 记录进入窗口时的原始目标 (模型"保持"的基准)
        if self._hold_raw_ref is None:
            self._hold_raw_ref = raw_action[:6].copy()
        raw_drift = float(np.max(np.abs(raw_action[:6] - self._hold_raw_ref[:6])))

        # 物理滞后: 原始目标与当前关节差异 (含伺服抖动/慢漂移的滞后)
        hold_delta = float(np.max(np.abs(raw_action[:6] - current_joints[:6])))

        if raw_drift < raw_band and hold_delta < hold_band:
            self._hold_lock_count += 1
        else:
            # 模型目标在漂移 (运动) 或机械臂未跟上 → 重置计数, 以当前目标
            # 为新参考重开保持窗口 (避免慢速任务运动时误锁定)
            self._hold_lock_count = 0
            self._hold_lock_active = False
            self._hold_lock_target = None
            self._hold_raw_ref = raw_action[:6].copy()

        # 连续保持达到阈值 → 建立锁定
        if (not self._hold_lock_active
                and self._hold_lock_count >= self.config.smooth_hold_steps):
            self._hold_lock_active = True
            self._hold_lock_target = joint_smoothed[:6].copy()
            logger.info(
                f"[HoldLock] 🔒 保持锁定激活 (目标 "
                f"{[f'{j*180/math.pi:.1f}°' for j in self._hold_lock_target]}, "
                f"抑制微振荡)"
            )

        # 锁定期: 模型目标仍在保持带内 → 冻结关节到锁定目标
        if self._hold_lock_active and self._hold_lock_target is not None:
            if raw_drift >= raw_band * 2.0 or hold_delta >= hold_band * 2.0:
                # 模型重新要运动 (目标明显漂移) 或目标脱离机械臂 → 解除
                self._hold_lock_active = False
                self._hold_lock_target = None
                self._hold_raw_ref = raw_action[:6].copy()
                logger.info("[HoldLock] 🔓 保持锁定解除 (模型开始运动)")
            else:
                # 冻结关节 (夹爪通道不锁定, 允许开合)
                joint_smoothed[:6] = self._hold_lock_target

        return joint_smoothed

    # ========================================================================
    # VLM 连拍投票 (静止确认后连续多帧判断)
    # ========================================================================

    def _start_vlm_burst(self, abort_on_motion: bool = True, settle_time_s: float = 0.0):
        """
        启动 VLM 连拍投票。

        Args:
            abort_on_motion: 连拍期间检测到机械臂运动时是否中止连拍。
                             静止检测触发的连拍为 True; 子任务超时触发的连拍
                             为 False (此时机械臂可能处于循环运动中, 需要强制重评估)。
            settle_time_s: 连拍开始前的静置时间 (秒)。子任务超时连拍时,
                           机械臂刚被暂停可能有残余运动, 先静置再取帧更可靠。
        """
        self._burst_active = True
        self._burst_count = 0
        self._burst_abort_on_motion = abort_on_motion
        # settle_time_s > 0 时延迟首次 VLM 调用, 让机械臂先静置
        self._burst_next_time = time.time() + settle_time_s
        self._burst_start_time = time.time()  # 记录开始时间用于整体超时
        self.voter.reset()           # 清空旧窗口, 重新收集
        logger.info(
            f"[Orchestrator] 📸 启动 VLM 连拍投票 "
            f"(目标 {self.config.vlm_burst_count} 帧, 间隔 {self.config.vlm_burst_interval}s, "
            f"总超时 {self.config.vlm_burst_total_timeout}s, "
            f"运动中止={'开' if abort_on_motion else '关'})"
        )

    def _vlm_burst_tick(self):
        """连拍时钟: 按间隔执行 VLM 调用"""
        if not self._burst_active:
            return

        now = time.time()

        # 整体超时保护: 如果连拍总时间超过阈值, 强制结束
        burst_elapsed = now - self._burst_start_time
        if burst_elapsed > self.config.vlm_burst_total_timeout:
            logger.warning(
                f"[Orchestrator] ⚠️ VLM 连拍整体超时 "
                f"({burst_elapsed:.1f}s > {self.config.vlm_burst_total_timeout}s), "
                f"已收集 {self._burst_count}/{self.config.vlm_burst_count} 帧, "
                f"强制结束连拍"
            )
            self._finish_vlm_burst()
            return

        # 已达到目标帧数 → 不再调用, 等待主循环消费共识结果
        if self._burst_count >= self.config.vlm_burst_count:
            return

        # 未到下次调用时间 → 等待
        if now < self._burst_next_time:
            return

        # 执行一次 VLM 调用
        logger.info(
            f"[Orchestrator] 📸 连拍 {self._burst_count + 1}/{self.config.vlm_burst_count}"
        )
        self._vlm_check_and_vote()
        self._burst_count += 1

        # 安排下次调用
        self._burst_next_time = now + self.config.vlm_burst_interval

    def _finish_vlm_burst(self, aborted: bool = False):
        """
        结束 VLM 连拍投票。

        Args:
            aborted: True 表示连拍因机械臂运动而中止 (丢弃未消费的投票)。
        """
        if aborted:
            # 中止连拍: 丢弃未消费的投票, 避免半途共识被使用
            self.voter.reset()
            logger.info("[Orchestrator] 📸 VLM 连拍投票中止 (机械臂运动)")
        else:
            logger.info("[Orchestrator] 📸 VLM 连拍投票结束")

        self._burst_active = False
        self._burst_count = 0
        self._burst_next_time = 0.0
        # 重置静止检测器, 准备下一轮检测。
        # keep_cooldown=True: 保留静止冷却 (stationary_recheck_cooldown),
        # 避免连拍结束后立刻再次触发 → 每 1-2s 一次连拍暂停机械臂的抖动
        # (2026-08-07 实测)。full_reset() 会清冷却, 不能用于此处。
        self.stationary_detector.full_reset(keep_cooldown=True)

        # ── 快速重评估: 若本次连拍未触发模型切换 ──
        # 说明当前模型可能已失效/任务已结束 (机械臂静止), 启用 fast_recheck
        # 让下一轮重评估用更短的静止阈值 (1s), 加快任务切换。
        # 若切换成功 (_handle_consensus 已 set_fast_recheck(False)), 保持正常阈值。
        if not self._last_burst_switched:
            self.stationary_detector.set_fast_recheck(True)

        # 重置激活模型的内部状态 (清空动作队列/时序集成器),
        # 确保连拍结束后 ACT 从当前观测重新预测 chunk, 不回放陈旧动作
        # (修复"抓取前先抬升再恢复原状"的抖动)。
        self.model_manager.reset_active_policy()

    # ========================================================================
    # 共识处理 → 模型切换
    # ========================================================================

    def _handle_consensus(self, subtask: str, target_model: str):
        """
        处理投票共识结果, 执行模型切换。

        注意: 此方法仅在 VLM 连拍完成后被调用,
        此时机械臂已确认静止 ≥ stationary_time_threshold 秒,
        无需额外的运动保护。

        Args:
            subtask: VLM 判断的 subtask (subtask1/2/3/4)
            target_model: 对应的任务标签 (Task_A/B/C/standby)
        """
        logger.info(
            f"[Orchestrator] 🎯 投票共识: subtask={subtask} → model={target_model}"
        )

        # 默认认为本次连拍未切换; 若下方实际切换成功则置 True。
        # _finish_vlm_burst 据此决定是否启用快速重评估。
        self._last_burst_switched = False

        current_model = self.model_manager.active_model

        # 场景未变化 → 无需切换
        if target_model == "standby" and self.model_manager.is_standby:
            logger.debug("[Orchestrator] 已在待机状态, 无需切换")
            self._current_subtask = subtask
            return

        if target_model == current_model and target_model != "standby":
            logger.debug(f"[Orchestrator] 模型未变化 ({target_model}), 无需切换")
            self._current_subtask = subtask
            return

        # ── 执行任务切换 ──
        # 先不重启推理线程 (restart_thread=False): 切换到任务平均初始位姿后
        # 再重启, 让模型基于新位姿 + 新任务文本重新预测。
        logger.info(f"[Orchestrator] 🔄 切换任务: {current_model or 'standby'} → {target_model}")
        success = self.model_manager.switch_model(target_model, restart_thread=False)

        if success:
            self._current_subtask = subtask
            self._current_model = target_model
            self._last_switch_step = self._loop_count

            # 重置子任务计时器
            self._subtask_start_time = time.time()
            self._subtask_timeout_triggered = False

            # 设置切换保护: 保护期内禁止 VLM 连拍, 给新任务执行时间
            self._switch_protection_until = time.time() + self.config.switch_cooldown
            logger.info(
                f"[Orchestrator] 🛡️ 切换保护激活: "
                f"{self.config.switch_cooldown:.0f}s 内禁止模型切换"
            )

            # ── 切换到任务对应的平均初始位姿 (缓慢移动, 训练均值) ──
            # standby (包裹在盒子) 时在 standby_preset_task 的初始位姿待机。
            self._move_to_task_preset(target_model)

            # 非待机任务: 移动完成后重启后台推理线程
            if target_model != "standby":
                self.model_manager.start_inference_thread()

            # 切换后重置平滑器, 并用当前关节+夹爪状态初始化基线避免关节跳变
            if self._smoother is not None:
                try:
                    joint_obs = self._robot.get_observation()
                    state_vals = []
                    for key, value in joint_obs.items():
                        if key.endswith('.pos'):
                            state_vals.append(np.float32(value))
                    if len(state_vals) >= 6:
                        # 若观测含第 7 维 (夹爪), 用真实夹爪值初始化基线,
                        # 避免旧版固定用 0.0 导致切换后夹爪被 EMA 拉向闭合。
                        gripper_val = float(state_vals[6]) if len(state_vals) >= 7 else 0.0
                        self._smoother.seed(np.array(state_vals[:6]), current_gripper=gripper_val)
                    else:
                        self._smoother.reset()
                    # 开启过渡平滑, 抑制新任务起始 chunk 的关节冲击,
                    # 让任务间切换更流畅
                    self._smoother.begin_transition()
                except Exception:
                    self._smoother.reset()

            # 重置静止检测器 (避免旧状态干扰新任务运行)
            self.stationary_detector.full_reset()
            # 切换成功 → 恢复正常重评估阈值
            self.stationary_detector.set_fast_recheck(False)
            self._last_burst_switched = True

            # 切换任务 → 解除保持锁定 (新任务将产生新动作)
            self._hold_lock_active = False
            self._hold_lock_target = None
            self._hold_lock_count = 0
            self._hold_raw_ref = None
            # 清空 chunk 边界诊断基准 (避免跨任务对比误导)
            self._prev_raw_action = None
            self._prev_sent = None
            # 夹爪二值滞回状态复位 (预设位姿已打开夹爪, 见 _move_to_task_preset)
            self._gripper_state = "open"
            self._gripper_dwell = 0
            self._last_gripper_ratio = 1.0

            logger.info(f"[Orchestrator] ✅ 任务切换完成: {target_model}")
        else:
            logger.error(f"[Orchestrator] ❌ 任务切换失败: → {target_model}")

    # ========================================================================
    # 任务预设位姿移动
    # ========================================================================

    def _move_to_task_preset(self, task_label: str) -> bool:
        """切换到任务对应的平均初始位姿 (缓慢移动, 训练 state 均值)。

        任务切换后先缓慢移到该任务的训练均值位姿 (工作区), 让模型初始状态
        落在训练分布内, 避免"初始位姿偏离训练分布导致无动作"
        (对齐 inference_pi05_adarms_fix.py 的 preset_pose 机制)。

        standby (包裹在盒子) 时在 standby_preset_task (默认 Task_A) 的
        初始位姿待机, 等待场景复原。

        Args:
            task_label: "Task_A"/"Task_B"/"Task_C"/"standby"

        Returns:
            True 如果移动成功 (或已禁用/无需移动)
        """
        if not self.config.preset_pose_enabled:
            return False

        # standby → 用待机任务的位姿
        preset_task = task_label
        if preset_task == "standby":
            preset_task = self.config.standby_preset_task

        joints = self.config.preset_poses.get(preset_task)
        if joints is None:
            joints = self.config.preset_pose_fallback
            if joints is None:
                logger.warning(f"[Preset] 任务 {preset_task} 无预设位姿, 跳过")
                return False
            logger.info(f"[Preset] 任务 {preset_task} 无专属位姿, 使用兜底位姿")

        logger.info(
            f"[Preset] 🎯 切换到 {preset_task} 平均初始位姿: "
            f"{[f'{j*180/math.pi:.1f}°' for j in joints]}"
        )
        try:
            move_to_preset_pose(
                self._piper,
                np.asarray(joints, dtype=np.float64),
                gripper_open=self.config.preset_pose_gripper,
                steps=self.config.preset_pose_steps,
                step_dt=self.config.preset_pose_step_dt,
                motion_speed=self.config.robot_motion_speed,
                motion_high_follow=self.config.robot_motion_high_follow,
            )
            return True
        except Exception as e:
            logger.error(f"[Preset] ❌ 预设位姿移动失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ========================================================================
    # 待机模式
    # ========================================================================

    def _handle_standby(self):
        """待机模式处理: 利用静止检测触发 VLM 连拍检查场景复原"""
        if self._standby_start_time == 0.0:
            self._standby_start_time = time.time()
            # 进入待机时清除子任务超时计时（待机不是子任务）
            self._subtask_start_time = 0.0
            self._subtask_timeout_triggered = False
            logger.info("[Orchestrator] ⏸️ 进入待机模式, 等待场景复原...")
            self._last_standby_log_time = 0.0  # 用于控制日志输出频率

        standby_duration = time.time() - self._standby_start_time

        # 每 30 秒打印一次待机状态 (使用状态变量避免同一秒内重复打印)
        if standby_duration - self._last_standby_log_time >= 30.0:
            self._last_standby_log_time = standby_duration
            logger.info(f"[Orchestrator] ⏸️ 待机中... ({standby_duration:.0f}s)")

        # 待机时静止检测照常工作, 会触发 VLM 连拍
        # 检查连拍是否有新共识 (场景复原 → 自动切换出待机)
        if self._burst_active and self.voter.is_window_full():
            consensus = self.voter.get_consensus()
            if consensus is not None:
                subtask, target_model = consensus
                if target_model != "standby":
                    logger.info(f"[Orchestrator] 🟢 场景复原检测: {subtask} → {target_model}")
                    self._standby_start_time = 0.0
                    self._handle_consensus(subtask, target_model)
            self._finish_vlm_burst()

    # ========================================================================
    # 动作发送
    # ========================================================================

    def _send_action(self, joint_action: np.ndarray):
        """将关节动作转换为机械臂指令并发送。

        Returns:
            实际发送的夹爪命令 (0.001mm), 供诊断日志使用。
        """
        # 验证输入
        if joint_action is None:
            logger.error("[Action] joint_action 为 None, 跳过发送")
            return 0
        if len(joint_action) < 7:
            logger.error(f"[Action] joint_action 长度不足: {len(joint_action)}, 期望≥7, 跳过发送")
            return 0

        # 关节角度 rad → 0.001°
        joint_cmds = []
        for i in range(6):
            clamped_val = clamp_joint(joint_action[i], JOINT_LIMITS, i)
            joint_cmds.append(int(clamped_val * RAD_TO_MILLIDEG))

        # 关节4 额外限位保护
        if joint_cmds[4] < -70000:
            joint_cmds[4] = -70000

        # ── 夹爪控制: 二值滞回 (默认) 或连续处理 ──
        # 模型输出的夹爪意图常在 chunk 内全开↔全闭往返 (夹爪极差>0.8),
        # 导致夹爪快速扇动、抓取时刻无法稳定闭合。二值滞回 + 最小驻留时间
        # 把意图转化为稳定"开/闭"状态, 闭合时直接发送 0 (完全闭合)。
        # (同步 inference_pi05_flowmatch.py 调优)
        gripper_ratio = float(joint_action[6])
        if self.config.gripper_binary:
            self._gripper_dwell += 1
            if gripper_ratio > self.config.gripper_open_above and self._gripper_state != "open":
                if self._gripper_dwell >= self.config.gripper_min_dwell:
                    self._gripper_state = "open"
                    self._gripper_dwell = 0
            elif gripper_ratio < self.config.gripper_close_below and self._gripper_state != "closed":
                if self._gripper_dwell >= self.config.gripper_min_dwell:
                    self._gripper_state = "closed"
                    self._gripper_dwell = 0
            gripper_cmd = (
                0 if self._gripper_state == "closed"
                else int(round(self.config.gripper_max_open_mm * 1000))
            )
            gripper_debug = {'binary': True, 'state': self._gripper_state}
        else:
            # ── 夹爪闭合增强 (对齐 inference_pi05_adarms_fix.py) ──
            # 模型输出 gripper 在闭合阶段可能停在 0.05~0.25 (夹爪有间隙)。
            # 分段线性压缩把低值映射到更小值 (夹得更紧); 映射连续无阶跃,
            # 保留夹爪状态平滑变化, 不破坏模型闭环。
            if self.config.gripper_gain > 0 and gripper_ratio < self.config.gripper_gain_above:
                if gripper_ratio <= self.config.gripper_gain_below:
                    gripper_ratio = gripper_ratio * self.config.gripper_gain
                else:
                    span = self.config.gripper_gain_above - self.config.gripper_gain_below
                    t = (gripper_ratio - self.config.gripper_gain_below) / span
                    gripper_ratio = (
                        self.config.gripper_gain_below * self.config.gripper_gain
                        + t * (self.config.gripper_gain_above
                               - self.config.gripper_gain_below * self.config.gripper_gain)
                    )

            # 夹爪处理
            gripper_cmd, gripper_debug = self._gripper_filter.process(gripper_ratio)

        # ── 干跑模式: 只打印动作, 不发送给机械臂 (安全测试状态机) ──
        if self.config.dry_run:
            if self._loop_count % 25 == 0 or self.config.debug_verbose:
                joint_deg = [j * 180 / math.pi for j in joint_action[:6]]
                logger.info(
                    f"[DryRun] 动作(°): {[f'{d:.1f}' for d in joint_deg]}, "
                    f"夹爪cmd: {gripper_cmd}"
                )
            return gripper_cmd

        # 发送控制指令 (带异常保护)
        try:
            # 运动参数与 HIL 采集对齐: 速度100 + MIT/高跟随 (0xAD)
            motion_ctrl(self._piper, self.config.robot_motion_speed,
                        self.config.robot_motion_high_follow)
            self._piper.JointCtrl(*joint_cmds)
            self._piper.GripperCtrl(abs(gripper_cmd), 1000, 0x01, 0)
        except Exception as e:
            # Piper SDK 通信异常通常是瞬时的，记录但不中断循环
            logger.warning(f"[Action] 机械臂通信异常 (单次): {e}")

        # 调试输出
        if self._loop_count % 50 == 0 and self.config.debug_verbose:
            joint_deg = [j * 180 / math.pi for j in joint_action[:6]]
            logger.debug(
                f"[Action] 关节(°): {[f'{d:.1f}' for d in joint_deg]}, "
                f"夹爪: {gripper_cmd}"
            )

        return gripper_cmd

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_task_prompt(self) -> str:
        """根据当前激活的 PI 模型返回对应的训练任务文本。

        必须与训练数据集的 task 文本逐字一致 (PI 模型对 task token 敏感)。
        任务文本映射见 config.TASK_PROMPTS:
            Task_A ← batch5A: "Grab the package and place it on the barcode scanning table."
            Task_B ← batch5B: "Flip the package if the barcode is not facing up."
            Task_C ← batch5C: "Grab the scanned package and place it into the box."
        """
        active = self.model_manager.active_model
        return TASK_PROMPTS.get(active, DEFAULT_TASK_PROMPT)

    def _print_stats(self):
        """打印运行统计"""
        avg_infer = self._total_infer_time / max(self._loop_count, 1) * 1000
        avg_loop = self._total_loop_time / max(self._loop_count, 1) * 1000
        # 真实控制频率 = 步数 / 墙钟经过时间 (busy_wait 等待也计入)
        wall_elapsed = time.time() - self._wall_start_time
        actual_freq = self._loop_count / max(wall_elapsed, 0.001)

        vlm_stats = self.vlm_client.stats
        model_stats = self.model_manager.stats
        voter_stats = self.voter.stats
        stationary_stats = self.stationary_detector.stats

        logger.info(
            f"[Stats] 步数:{self._loop_count:6d} | "
            f"推理:{avg_infer:5.1f}ms | "
            f"循环:{avg_loop:5.1f}ms | "
            f"频率:{actual_freq:4.1f}Hz | "
            f"模型:{model_stats['active_model']} | "
            f"切换:{model_stats['total_switches']} | "
            f"VLM:{vlm_stats['success_calls']}/{vlm_stats['total_calls']} "
            f"({vlm_stats['success_rate']:.0%}) | "
            f"共识:{voter_stats['total_consensuses']} | "
            f"静止事件:{stationary_stats['total_stationary_events']}"
        )

    # ========================================================================
    # 关闭
    # ========================================================================

    def _shutdown(self):
        """优雅关闭"""
        logger.info("\n" + "=" * 70)
        logger.info("  正在关闭...")
        logger.info("=" * 70)

        self._running = False

        # 清理模型 (先停后台推理线程, 避免线程还在用相机/机器人)
        self.model_manager.cleanup()

        # 停止相机
        self.camera_manager.disconnect()

        # 保存决策日志
        self._save_decision_log()

        # 打印最终统计
        logger.info(f"\n[Final Stats]")
        logger.info(f"  总步数: {self._loop_count}")
        if self._loop_count > 0:
            wall_elapsed = time.time() - self._wall_start_time
            logger.info(f"  平均推理时间: {self._total_infer_time/self._loop_count*1000:.1f}ms")
            logger.info(f"  平均循环时间: {self._total_loop_time/self._loop_count*1000:.1f}ms")
            logger.info(f"  实际频率: {self._loop_count/max(wall_elapsed,0.001):.1f}Hz "
                        f"(墙钟 {wall_elapsed:.1f}s)")
        logger.info(f"  模型切换次数: {self.model_manager.stats['total_switches']}")
        logger.info(f"  VLM 调用: {self.vlm_client.stats}")
        logger.info(f"  投票共识次数: {self.voter.stats['total_consensuses']}")
        logger.info(f"  静止事件: {self.stationary_detector.stats['total_stationary_events']}")
        logger.info(f"  日志目录: {self._log_dir}")
        logger.info("=" * 70)

    def _save_decision_log(self):
        """保存 VLM 决策日志到 JSON 文件"""
        if not self._decision_log:
            return

        log_path = os.path.join(self._log_dir, f"vlm_decisions_{int(time.time())}.json")
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self._decision_log, f, ensure_ascii=False, indent=2)
            logger.info(f"[Orchestrator] 决策日志已保存: {log_path} "
                        f"({len(self._decision_log)} 条)")
        except Exception as e:
            logger.error(f"[Orchestrator] 保存决策日志失败: {e}")

    def _save_scene_snapshot(self, joint_delta: float):
        """
        保存"模型保持姿态"时的场景快照 (相机图像 + 关节状态 + 模型信息),
        便于复盘是模型问题还是场景不匹配。

        节流: 每次运行最多保存 3 份快照 (按时间戳命名)。
        """
        try:
            import cv2

            # 节流: 最多保存 3 份
            snap_count = len([f for f in os.listdir(self._log_dir)
                              if f.startswith("snapshot_")])
            if snap_count >= 3:
                return

            ts = int(time.time())
            top_img, wrist_img, _ = self.camera_manager.get_latest_frames()

            if top_img is not None:
                cv2.imwrite(os.path.join(self._log_dir, f"snapshot_{ts}_top.jpg"), top_img)
            if wrist_img is not None:
                cv2.imwrite(os.path.join(self._log_dir, f"snapshot_{ts}_wrist.jpg"), wrist_img)

            # 关节状态 + 模型信息
            info = {
                "timestamp": ts,
                "model": self.model_manager.active_model,
                "subtask": self._current_subtask,
                "joint_delta_rad": joint_delta,
                "loop_count": self._loop_count,
            }
            # 尽量读取当前关节, 失败则不记录
            try:
                joint_obs = self._robot.get_observation()
                info["joints"] = {
                    k: float(v) for k, v in joint_obs.items() if k.endswith(".pos")
                }
            except Exception:
                pass

            with open(os.path.join(self._log_dir, f"snapshot_{ts}.json"),
                      'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            logger.info(f"[Diagnose] 📸 场景快照已保存到 {self._log_dir} (snapshot_{ts}_*)")
        except Exception as e:
            logger.debug(f"[Diagnose] 保存场景快照失败: {e}")


# ============================================================================
# 工具函数
# ============================================================================

def busy_wait(seconds: float):
    """高精度忙等待"""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass
