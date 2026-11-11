"""
ACT + VLM 多任务联调编排器 — 主控制循环
=========================================

核心编排逻辑:
    1. VLM 周期性观察场景 → 分类当前包裹状态
    2. 窗口投票 (5 帧中 ≥3 帧一致) → 决定执行哪个子任务
    3. 根据投票结果切换 ACT 模型 (Task_A/B/C/standby)
    4. ACT 模型执行动作推理 → 控制机械臂
    5. 静止检测: 机械臂接近静止 3s → 触发 VLM 重评估 (保险机制)
    6. 安全监控: 关节跳变/连续异常 → 紧急停止

模型切换策略:
    - 切换时清除 GPU 缓存 + Python GC → 消除累积误差
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

from lerobot.act_vlm_orchestrator_v2.config import (
    OrchestratorConfig,
    SUBTASK_TO_MODEL,
    SUBTASK_DESCRIPTIONS,
)

from lerobot.act_vlm_orchestrator_v2.vlm_client import VLMClient, VLMDecision
from lerobot.act_vlm_orchestrator_v2.window_voter import WindowVoter
from lerobot.act_vlm_orchestrator_v2.act_model_manager import ACTModelManager
from lerobot.act_vlm_orchestrator_v2.stationary_detector import StationaryDetector
from lerobot.act_vlm_orchestrator_v2.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# ============================================================================
# 复用 inference_new_joint_act_v3.py 的核心组件
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

class ACTVLMOrchestrator:
    """
    ACT + VLM 多任务联调编排器

    整体循环:
        ┌──────────────────────────────────────────────────┐
        │  1. 获取观测 (关节状态 + 相机图像)                  │
        │  2. 静止检测: 关节变化 < 阈值 持续 N 秒             │
        │     └─ 触发 → VLM 连拍投票 (5 帧连续判断)           │
        │  3. 投票窗口满 → 检查共识 (≥3/5 票)                 │
        │  4. 共识达成 → 切换 ACT 模型 + 重置状态              │
        │  5. ACT 推理 → 机械臂执行                           │
        │  6. 安全监控 → 异常急停                             │
        └──────────────────────────────────────────────────┘

    【关键设计】
      - VLM 仅在机械臂静止 ≥ N 秒后触发连拍, 运动期间零干预
      - 连拍 5 帧快速投票, 不依赖多次静止触发
      - 切换模型时清 GPU 缓存 + 重置状态, 消除累积误差

    模型映射:
        subtask1 (桌面)      → Task_A (抓取到托盘)
        subtask2 (托盘,标签×) → Task_B (翻转包裹)
        subtask3 (托盘,标签✓) → Task_C (放入盒子)
        subtask4 (盒子)      → standby (待机等待复原)
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._running = False
        self._paused = False

        # ── 子系统 ──
        self.vlm_client = VLMClient(config)
        self.voter = WindowVoter(config)
        self.model_manager = ACTModelManager(config)
        self.stationary_detector = StationaryDetector(config)
        self.camera_manager = CameraManager(config)

        # ── 机器人 ──
        self._robot = None
        self._piper = None

        # ── 平滑 & 过滤 ──
        self._smoother: Optional[TrajectorySmoother] = None
        self._gripper_filter: Optional[GripperFilter] = None
        self._safety_monitor: Optional[SafetyMonitor] = None

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

        # ── 试次统计 (成功率 / 吞吐率) ──
        # 每个试次 = 一轮完整的 A→B→C 循环:
        #   起点: 首次 VLM 场景判断 (_main_loop 首次 _start_vlm_burst)
        #   终点: C 任务完成机械臂回位 (VLM 共识切换到 subtask4/standby)
        # 成功判定: 阈值 (默认 90s) 内执行完 A/B/C 三个任务
        self._trial_active: bool = False          # 是否有进行中的试次
        self._trial_start_time: float = 0.0       # 当前试次开始时间 (首次 VLM 场景判断)
        self._trial_deadline: float = 0.0         # 试次时间预算截止点 (start + threshold)
        self._trial_timeout_warned: bool = False  # 是否已发出超时警告 (只警告一次)
        self._executed_tasks: set = set()         # 本试次已执行过的 ACT 模型集合
        self._task_start_times: Dict[str, float] = {}  # 各任务开始时间
        self._task_end_times: Dict[str, float] = {}    # 各任务结束时间
        self._trial_results: list = []            # 本次进程内的试次结果
        self._trial_finished: bool = False        # 试次是否已结束 (完成/超时 → 退出进程)
        self._current_trial_id: int = 0           # 当前试次全局编号 (基于持久化历史, 跨进程递增)

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

        # ── 4. 预加载 ACT 模型 ──
        logger.info("[Init] 预加载 ACT 模型...")
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
            consecutive_warnings_limit=self.config.consecutive_warnings_limit,
        )
        logger.info("[Init] SafetyMonitor 已初始化")

        logger.info("=" * 70)
        logger.info("  ✅ 初始化完成, 准备开始联调循环")
        logger.info(f"  默认模型: {self.config.default_model}")
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
        logger.info(f"  模型路径: {self.config.model_paths}")
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
        # v9 (2026-08-10): 对齐 HIL (piper_follower) 驱动参数 — 100% 运动速度
        # + MIT 高跟随模式 (0xAD)。旧值 (30, 0x00) 使机械臂跟不上 ACT chunk
        # 命令, 造成观测-动作失同步与 chunk 边界抖动。
        self._piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
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
        logger.info("  🚀 ACT + VLM 联调循环开始")
        logger.info("  按 Ctrl+C 停止")
        logger.info("=" * 70 + "\n")

        total_steps = self.config.inference_time_s * int(self.config.fps)
        np.set_printoptions(suppress=True, precision=6)

        # 墙钟计时起点: 放在主循环开始处, 排除模型加载/相机连接等初始化耗时
        # (若在 __init__ 设置, 初始化耗时会被计入, 导致早期频率统计偏低)
        self._wall_start_time = time.time()

        # ── 首次场景评估: 启动 VLM 连拍投票, 确认正确模型后再执行 ACT ──
        # 旧版只做 1 次 VLM 判断 (无法填满 5 票窗口达成共识), 机械臂会立即用
        # 默认模型执行, 若场景需要其它子任务则产生错误动作 (无动作/轻微抬升又放下)。
        # 改为连拍投票: 共识确认前 ACT 推理被暂停 (阶段5跳过), 机械臂保持静止,
        # 确认正确的 ACT 模型后才开始执行。
        logger.info("[Orchestrator] 执行首次场景评估 (VLM 连拍投票确认模型)...")
        # 试次开始: 首次 VLM 场景判断 = 单试次总时间计时起点
        self._start_trial()
        self._start_vlm_burst()

        try:
            for step in range(total_steps):
                if _EMERGENCY_STOP:
                    logger.warning("[Orchestrator] ⚠️ 紧急停止触发, 退出循环")
                    break

                # ── 试次结束 (完成或超时) → 退出进程 ──
                if self._trial_finished and self.config.trial_auto_exit:
                    logger.info(
                        f"[Trial] 试次已结束 (总耗时 "
                        f"{time.time() - self._trial_start_time:.1f}s), 退出进程"
                    )
                    break

                start_loop_t = time.perf_counter()

                # ================================================
                # 阶段0: 试次时间预算监控
                #   达到阈值 → 判定失败并退出进程 (用户要求: 运行 90s 即退出)
                # ================================================
                if self._trial_active and not self._trial_timeout_warned:
                    now_wall = time.time()
                    if now_wall > self._trial_deadline:
                        self._trial_timeout_warned = True
                        logger.warning(
                            f"[Trial] ⏰ 已达到 {self.config.trial_time_threshold:.0f}s "
                            f"时间预算 (已运行 {now_wall - self._trial_start_time:.1f}s), "
                            f"试次判定为失败, 退出进程"
                        )
                        self._finish_trial(reason="timeout")
                        # 超时退出前主动保持机械臂当前位置, 避免失稳
                        self._hold_arm_at_current()
                        break

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
                    # 连拍模式: 主动保持机械臂静止 (发送当前关节位置),
                    # 抑制伺服保持抖动, 让 VLM 采集到清晰稳定的场景图像。
                    self._hold_arm_at_current()
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
                # 阶段5: ACT 模型推理
                #   注意: VLM 连拍期间暂停 ACT 推理, 因为:
                #     1) 连拍由静止检测触发, 手臂已停止运动, ACT 输出无意义
                #     2) 避免 ACT 和 VLM 竞争相机帧 (帧缓存已解决, 但推理暂停减少无效计算)
                #     3) VLM 需要清晰、无机械臂遮挡的场景图像
                # ================================================
                if self._burst_active:
                    # VLM 连拍进行中, 跳过 ACT 推理, 仅做频率控制
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
                    logger.error(f"[Orchestrator] ACT 推理失败: {e}")
                    import traceback
                    traceback.print_exc()
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
                # 阶段8: 动作转换与发送
                # ================================================
                self._send_action(joint_smoothed)

                # 记录上一步原始/发送动作 (供 chunk 边界诊断对比)
                self._prev_raw_action = numpy_action.copy()
                self._prev_sent = joint_smoothed.copy()

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
            target_model: 对应的 ACT 模型名 (Task_A/B/C/standby)
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
            # 任务确认 (无需切换): 记录任务开始/已执行。
            # 关键修复: 启动时默认模型即 Task_A, 首次共识 subtask1 → Task_A
            # 走此路径 (不触发模型切换), 若不在此记录则 Task_A 不会被计入
            # 已执行任务, 导致 A/B/C 全部完成仍被误判为失败。
            self._record_task_start(target_model)
            return

        # ── 执行模型切换 ──
        logger.info(f"[Orchestrator] 🔄 切换模型: {current_model or 'standby'} → {target_model}")
        success = self.model_manager.switch_model(target_model)

        if success:
            self._current_subtask = subtask
            self._current_model = target_model
            self._last_switch_step = self._loop_count

            # ── 试次统计: 任务执行追踪 ──
            # 上一个任务结束 (current_model 为切换前的模型)
            if current_model in ("Task_A", "Task_B", "Task_C"):
                self._task_end_times[current_model] = time.time()
            # 进入新任务 → 记录开始时间 + 标记已执行
            self._record_task_start(target_model)
            # 进入待机 (subtask4, 包裹已在盒子) = C 任务完成机械臂回位 → 试次结束
            if target_model == "standby":
                self._finish_trial()

            # 重置子任务计时器
            self._subtask_start_time = time.time()
            self._subtask_timeout_triggered = False

            # 设置切换保护: 保护期内禁止 VLM 连拍, 给新模型执行时间
            self._switch_protection_until = time.time() + self.config.switch_cooldown
            logger.info(
                f"[Orchestrator] 🛡️ 切换保护激活: "
                f"{self.config.switch_cooldown:.0f}s 内禁止模型切换"
            )

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
                    # 开启过渡平滑, 抑制新模型起始 chunk 的关节冲击,
                    # 让任务间切换更流畅
                    self._smoother.begin_transition()
                except Exception:
                    self._smoother.reset()

            # 重置静止检测器 (避免旧状态干扰新模型运行)
            self.stationary_detector.full_reset()
            # 切换成功 → 恢复正常重评估阈值
            self.stationary_detector.set_fast_recheck(False)
            self._last_burst_switched = True

            # 切换模型 → 解除保持锁定 (新模型将产生新动作)
            self._hold_lock_active = False
            self._hold_lock_target = None
            self._hold_lock_count = 0
            self._hold_raw_ref = None
            # 清空 chunk 边界诊断基准 (避免跨模型对比误导)
            self._prev_raw_action = None
            self._prev_sent = None

            logger.info(f"[Orchestrator] ✅ 模型切换完成: {target_model}")
        else:
            logger.error(f"[Orchestrator] ❌ 模型切换失败: → {target_model}")

    # ========================================================================
    # 待机模式
    # ========================================================================

    def _handle_standby(self):
        """待机模式处理: 利用静止检测触发 VLM 连拍检查场景复原"""
        # 主动保持机械臂静止 (发送当前关节位置), 抑制任务完成回位后的
        # 伺服保持抖动。
        self._hold_arm_at_current()

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
                    # 场景复原到 subtask1 (新包裹在桌面) → 开启新一轮试次
                    if target_model == "Task_A":
                        self._start_trial()
                    self._handle_consensus(subtask, target_model)
            self._finish_vlm_burst()

    # ========================================================================
    # 动作发送
    # ========================================================================

    def _send_action(self, joint_action: np.ndarray):
        """将关节动作转换为机械臂指令并发送"""
        # 验证输入
        if joint_action is None:
            logger.error("[Action] joint_action 为 None, 跳过发送")
            return
        if len(joint_action) < 7:
            logger.error(f"[Action] joint_action 长度不足: {len(joint_action)}, 期望≥7, 跳过发送")
            return

        # 关节角度 rad → 0.001°
        joint_cmds = []
        for i in range(6):
            clamped_val = clamp_joint(joint_action[i], JOINT_LIMITS, i)
            joint_cmds.append(int(clamped_val * RAD_TO_MILLIDEG))

        # 关节4 额外限位保护
        if joint_cmds[4] < -70000:
            joint_cmds[4] = -70000

        # 夹爪处理
        gripper_cmd, gripper_debug = self._gripper_filter.process(joint_action[6])

        # ── 干跑模式: 只打印动作, 不发送给机械臂 (安全测试状态机) ──
        if self.config.dry_run:
            if self._loop_count % 25 == 0 or self.config.debug_verbose:
                joint_deg = [j * 180 / math.pi for j in joint_action[:6]]
                logger.info(
                    f"[DryRun] 动作(°): {[f'{d:.1f}' for d in joint_deg]}, "
                    f"夹爪cmd: {gripper_cmd}"
                )
            return

        # 发送控制指令 (带异常保护)
        try:
            # v9 (2026-08-10): 对齐 HIL (piper_follower) 驱动参数 — 100% 运动
            # 速度 + MIT 高跟随模式 (0xAD), 使机械臂能紧跟 ACT chunk 命令,
            # 避免观测-动作失同步导致的 chunk 边界抖动。
            self._piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
            self._piper.JointCtrl(*joint_cmds)
            self._piper.GripperCtrl(abs(gripper_cmd), 5000, 0x01, 0)
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

    def _hold_arm_at_current(self):
        """
        主动保持: 把当前关节位置作为目标持续发送, 抑制伺服保持抖动。

        在 VLM 连拍 / 待机期间调用 (此时无 ACT 命令下发, 机械臂自由保持
        会产生 ±1° 量级伺服抖动), 让机械臂稳定静止, 同时使 VLM 采集到
        更清晰的场景图像。
        """
        if self._robot is None or self._piper is None or self.config.dry_run:
            return
        try:
            joint_obs = self._robot.get_observation()
            state_vals = [np.float32(v) for k, v in joint_obs.items()
                          if k.endswith('.pos')]
            if len(state_vals) >= 6:
                joint_cmds = [
                    int(clamp_joint(state_vals[i], JOINT_LIMITS, i) * RAD_TO_MILLIDEG)
                    for i in range(6)
                ]
                self._piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
                self._piper.JointCtrl(*joint_cmds)
        except Exception:
            # 通信异常为瞬时性, 忽略 (下一帧重试)
            pass

    # ========================================================================
    # 辅助方法
    # ========================================================================

    def _get_task_prompt(self) -> str:
        """根据当前 subask 生成任务提示"""
        prompts = {
            "subtask1": "Grab the parcel from the table and place it on the tray",
            "subtask2": "Flip the parcel on the tray to make the label visible",
            "subtask3": "Pick up the parcel from the tray and place it into the box",
        }
        return prompts.get(self._current_subtask, "Perform the manipulation task")

    # ========================================================================
    # 试次统计 (成功率 / 吞吐率)
    # ========================================================================

    def _start_trial(self):
        """
        开始新试次: 以首次 VLM 场景判断为计时起点。

        调用时机:
          - 程序启动, _main_loop 首次场景评估前
          - 待机中检测到场景复原 (subtask1, 新包裹在桌面)
        """
        if self._trial_active:
            return
        self._trial_active = True
        self._trial_finished = False
        self._trial_start_time = time.time()
        self._trial_deadline = self._trial_start_time + self.config.trial_time_threshold
        self._trial_timeout_warned = False
        self._executed_tasks = set()
        self._task_start_times = {}
        self._task_end_times = {}
        # 全局唯一试次编号: 基于持久化历史的最大编号 + 1 (跨进程递增不重复,
        # 避免每次进程都从 1 开始导致 trial_id 重复)
        history = self._load_trial_history()
        self._current_trial_id = max(
            [t.get("trial_id", 0) for t in history], default=0) + 1
        logger.info(
            f"[Trial] ▶️ 试次#{self._current_trial_id} 开始 "
            f"(VLM 场景判断, 时间预算 {self.config.trial_time_threshold:.0f}s)"
        )

    def _record_task_start(self, task_name: str):
        """
        记录任务开始时间并标记为已执行 (模型切换与共识确认共用)。

        只有当该任务首次被记录时才写入开始时间 (避免任务中途被 VLM 重复
        确认时重置计时); 已执行集合始终更新。

        Args:
            task_name: "Task_A" / "Task_B" / "Task_C" / "standby"
        """
        if task_name not in ("Task_A", "Task_B", "Task_C"):
            return
        if task_name not in self._task_start_times:
            self._task_start_times[task_name] = time.time()
            logger.info(
                f"[Trial] 🎯 任务 {task_name} 开始 (试次已运行 "
                f"{time.time() - self._trial_start_time:.1f}s)"
            )
        self._executed_tasks.add(task_name)

    def _finish_trial(self, reason: str = "standby"):
        """
        结束当前试次: 计算总耗时 / 各任务耗时 / 成功判定, 写入持久化日志。

        Args:
            reason: "standby"  = 正常进入待机 (C 任务完成, 包裹在盒子);
                    "timeout"   = 达到时间预算 (90s) 仍未完成;
                    "shutdown"  = 程序中断 (试次未完成)。
        """
        if not self._trial_active:
            return
        self._trial_active = False
        self._trial_finished = True

        now = time.time()
        total_time = now - self._trial_start_time
        timeout = total_time > self.config.trial_time_threshold

        # 各任务耗时: 若某任务未记录结束时间 (试次中断), 用试次结束时间兜底
        task_times = {}
        for task in ("Task_A", "Task_B", "Task_C"):
            ts = self._task_start_times.get(task)
            if ts is not None:
                te = self._task_end_times.get(task, now)
                task_times[task] = te - ts

        # 成功判定: 阈值内执行完 A/B/C 三个任务
        tasks_done = self._executed_tasks >= {"Task_A", "Task_B", "Task_C"}
        success = tasks_done and not timeout

        result = {
            "trial_id": self._current_trial_id,
            "reason": reason,
            "start_time": self._trial_start_time,
            "end_time": now,
            "start_local": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._trial_start_time)),
            "end_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "total_time_s": round(total_time, 2),
            "threshold_s": self.config.trial_time_threshold,
            "timeout": timeout,
            "executed_tasks": sorted(self._executed_tasks),
            "tasks_done": tasks_done,
            "task_times_s": {k: round(v, 2) for k, v in task_times.items()},
            "success": success,
        }
        self._trial_results.append(result)
        self._log_trial_result(result)

        # 日志输出
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(
            f"[Trial] 🏁 试次#{result['trial_id']} {status} | "
            f"总耗时 {total_time:.1f}s (阈值 {self.config.trial_time_threshold:.0f}s) | "
            f"执行任务: {', '.join(sorted(self._executed_tasks)) or '无'}"
        )
        for task in ("Task_A", "Task_B", "Task_C"):
            if task in task_times:
                logger.info(f"[Trial]    {task}: {task_times[task]:.1f}s")
        if timeout:
            logger.warning(f"[Trial] ⚠️ 超出 {self.config.trial_time_threshold:.0f}s 时间预算")

    def _log_trial_result(self, result: dict):
        """追加写入试次结果到持久化文件 (跨程序重启累积) """
        try:
            stats_file = getattr(self.config, "trial_stats_file", None) or \
                os.path.join(self._log_dir, "trial_stats.jsonl")
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"[Trial] 结果已写入: {stats_file}")
        except Exception as e:
            logger.error(f"[Trial] 保存试次结果失败: {e}")

    def _load_trial_history(self) -> list:
        """读取持久化的历史试次结果 (含本次进程已写入的) """
        trials = []
        stats_file = getattr(self.config, "trial_stats_file", None) or \
            os.path.join(self._log_dir, "trial_stats.jsonl")
        try:
            if os.path.exists(stats_file):
                with open(stats_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            trials.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"[Trial] 读取历史统计失败: {e}")
        return trials

    def _print_trial_summary(self):
        """
        打印成功率与吞吐率统计。

        分两部分:
          - 本次运行: 当前进程内已结束的试次 (默认每次运行即一个试次);
          - 累计:     从持久化文件读取的历史试次汇总 (跨程序重启累积)。

        说明: 各任务平均耗时按"执行过该任务的试次"独立统计, 不同任务可能
        来自不同试次样本, 因此各任务平均之和 ≠ 单试次总耗时。
        """
        # ── 本次运行 ──
        for r in self._trial_results:
            task_str = ", ".join(
                f"{k}({v:.1f}s)" for k, v in (r.get("task_times_s") or {}).items()
            ) or "无"
            logger.info(
                f"[Trial 统计] 本次运行: 试次#{r.get('trial_id')} "
                f"{'✅ 成功' if r.get('success') else '❌ 失败'} | "
                f"总耗时 {r.get('total_time_s', 0):.1f}s | 任务: {task_str}"
            )

        # ── 累计 (跨进程持久化) ──
        trials = self._load_trial_history()
        if not trials:
            return
        n = len(trials)
        successes = [t for t in trials if t.get("success")]
        succ_n = len(successes)
        success_rate = succ_n / n

        total_times = [t["total_time_s"] for t in trials
                       if t.get("total_time_s") is not None]
        avg_total = sum(total_times) / len(total_times) if total_times else 0.0

        logger.info(
            f"[Trial 统计] 累计 ({n} 试次): 成功 {succ_n} ({success_rate:.0%}) | "
            f"失败 {n - succ_n} | 平均总耗时 {avg_total:.1f}s"
        )
        if succ_n > 0:
            succ_times = [t["total_time_s"] for t in successes]
            avg_succ = sum(succ_times) / len(succ_times)
            logger.info(
                f"[Trial 统计] 整循环吞吐率: {3600 / avg_succ:.1f} 循环/小时 "
                f"(成功试次平均 {avg_succ:.1f}s)"
            )
        else:
            logger.info("[Trial 统计] 整循环吞吐率: 无成功试次, 暂不统计")

        # 各任务平均耗时 → 单任务吞吐率 (仅统计执行过该任务的试次)
        for task in ("Task_A", "Task_B", "Task_C"):
            task_entries = [
                t.get("task_times_s", {}).get(task)
                for t in trials if t.get("task_times_s")
            ]
            task_entries = [x for x in task_entries if x]
            if not task_entries:
                continue
            avg_task = sum(task_entries) / len(task_entries)
            logger.info(
                f"[Trial 统计]   {task}: 平均 {avg_task:.1f}s "
                f"({len(task_entries)}/{n} 试次执行) "
                f"→ 吞吐率 ~{3600 / avg_task:.0f} 次/小时"
            )
        logger.info(
            "[Trial 统计]   (注: 各任务平均按执行过该任务的试次独立统计, "
            "合计 ≠ 单试次总耗时)"
        )

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

        # 停止相机
        self.camera_manager.disconnect()

        # 清理模型
        self.model_manager.cleanup()

        # 保存决策日志
        self._save_decision_log()

        # ── 试次统计收尾 ──
        # 程序中断时若仍有进行中的试次, 记为失败 (未完成 A/B/C)
        if self._trial_active:
            logger.warning("[Trial] 程序中断, 当前试次未完成 (记为失败)")
            self._finish_trial(reason="shutdown")
        # 打印累计成功率/吞吐率 (含本次运行的试次)
        self._print_trial_summary()

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
