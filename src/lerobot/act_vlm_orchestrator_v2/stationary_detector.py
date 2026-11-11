"""
静止检测器 — 机械臂接近静止状态检测 (v3: 速度阈值进入 + 保持带维持)

用途:
    当机械臂接近静止超过阈值时间 (默认 3s), 触发场景重评估。
    这作为保险机制, 防止:
      - 子任务完成但 VLM 未及时检测到
      - 机械臂卡住/遇到障碍物
      - ACT 模型输出退化导致机械臂停止运动

v2 改进 (修复"机械臂还在运动却触发切换"):
    旧版用"单步 delta < 阈值"判定, 25Hz 下等效速度阈值约 0.44 rad/s,
    过于宽松, 低速运动会误判为静止。

v3 改进 (修复"阈值过严导致永不触发"):
    子任务完成后机械臂通常不会绝对静止: 伺服保持存在微小抖动/缓慢收敛
    (典型 0.001~0.005 rad 的来回摆动, 折算速度可达 0.05 rad/s)。
    纯速度阈值过严会被抖动反复打断计时。v3 采用两段式判定:
      1) 进入判定: 相邻采样速度 < stationary_velocity_threshold → 锁定参考, 开始计时
      2) 维持判定: 静止期间位置相对参考的总漂移 < stationary_hold_band → 持续计时
         抖动是来回摆动(总位移≈0)天然被容忍; 缓慢单调漂移会累计超出保持带 → 判为运动

    另提供 is_moving(): 供 VLM 连拍期间的运动监测, 若机械臂恢复运动
    立即中止连拍, 避免在机械臂运动中切换任务。
"""

import time
import logging
from typing import Optional

import numpy as np

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class StationaryDetector:
    """
    机械臂静止状态检测器 (v3: 速度阈值进入 + 保持带维持)

    使用示例:
        detector = StationaryDetector(config)

        # 每个控制循环步:
        detector.update(joint_positions)
        if detector.is_stationary:
            # 触发 VLM 重评估
            detector.reset()
    """

    def __init__(self, config: OrchestratorConfig):
        # 速度阈值 (rad/s): 进入静止的判据 (容忍伺服保持时的微小抖动)
        self._velocity_threshold = config.stationary_velocity_threshold
        # 静止保持带 (rad): 静止期间相对参考位置的总漂移容忍
        self._stationary_hold_band = config.stationary_hold_band
        # 单步 delta 安全上限 (rad): 兜底, 防止采样异常误判
        self._joint_delta_safety = config.stationary_joint_threshold
        # 静止持续时间阈值 (秒)
        self._time_threshold = config.stationary_time_threshold

        # 静止检测基准 (位置 + 时间戳)
        self._prev_joints: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None

        # 运动检测基准 (用于连拍中止, 独立于静止计时)
        self._motion_prev_joints: Optional[np.ndarray] = None
        self._motion_prev_time: Optional[float] = None

        # 静止状态
        self._stationary_start_time: Optional[float] = None
        self._stationary_reference: Optional[np.ndarray] = None  # 进入静止时锁定的参考位置
        self._stationary_elapsed: float = 0.0
        self._is_stationary: bool = False
        self._stationary_triggered: bool = False

        # 静止参考重锁间隔 (秒): 静止期间周期性重锁参考, 容忍伺服保持时的
        # 物理抖动/慢漂移 (总漂移退化为窗口内漂移), 防止计时被缓慢漂移
        # 反复重置而卡死模型切换 (2026-08-06 实测)。
        self._renew_interval = getattr(
            config, "stationary_reference_renew_interval", 0.5)
        self._last_renew_time: float = 0.0

        # 统计
        self._total_checks: int = 0
        self._total_stationary_events: int = 0
        self._last_max_delta: float = 0.0
        self._last_velocity: float = 0.0
        self._last_drift: float = 0.0

        # 冷却时间 (避免频繁触发)
        self._last_trigger_time: float = 0.0
        self._cooldown = config.stationary_recheck_cooldown

        # ── 快速重评估模式 ──
        # 一次 VLM 连拍后若共识未变 (未切换模型), 用更短的静止阈值加速下一轮重评估,
        # 避免"静止3s+连拍1s"的 4s 死循环, 加快任务切换。
        self._fast_recheck_enabled = config.stationary_fast_recheck_enabled
        self._fast_recheck_time = config.stationary_fast_recheck_time

    def set_fast_recheck(self, enabled: bool):
        """
        启用/禁用快速重评估模式。

        启用时, 静止触发阈值临时切换为 fast_recheck_time (更短),
        使机械臂静止后更快触发下一轮 VLM 重评估。
        模型切换成功或恢复有效动作后应调用 set_fast_recheck(False) 恢复正常阈值。
        """
        self._fast_recheck_enabled = bool(enabled)
        if enabled:
            logger.debug(
                f"[Stationary] ⚡ 快速重评估模式开启 "
                f"(静止阈值 {self._time_threshold}s → {self._fast_recheck_time}s)"
            )

    # ========================================================================
    # 核心接口
    # ========================================================================

    def update(self, joint_positions: np.ndarray) -> bool:
        """
        采样一次关节位置并更新静止检测状态 (v3: 速度阈值进入 + 保持带维持)。

        两段式判定:
          阶段A (进入): 相邻采样速度 < velocity_threshold → 锁定参考位置, 开始计时
          阶段B (维持): 位置相对参考的总漂移 < hold_band → 持续累计静止时间;
                        总漂移 > hold_band → 判定为运动, 重置计时

        Args:
            joint_positions: 当前 6 个关节角度 (rad), shape (6,) 或 (7,)

        Returns:
            True 如果检测到"接近静止超过阈值时间" (触发重评估)
        """
        self._total_checks += 1

        # 确保是 (6,) 的关节数组
        joints = np.asarray(joint_positions, dtype=np.float64).ravel()[:6]

        now = time.time()

        # 首次调用: 初始化基准 (位置 + 时间戳)
        if self._prev_joints is None:
            self._prev_joints = joints.copy()
            self._prev_time = now
            return False

        # 计算真实采样间隔与最大关节速度 (与循环频率解耦)
        dt = max(now - self._prev_time, 1e-4)
        max_delta = np.max(np.abs(joints - self._prev_joints))
        max_velocity = max_delta / dt
        self._prev_joints = joints.copy()
        self._prev_time = now
        self._last_max_delta = max_delta
        self._last_velocity = max_velocity

        # ── 阶段A: 尚未进入静止状态 ──
        if self._stationary_start_time is None:
            # 速度低于阈值 (且单步 delta 在安全范围内) → 锁定参考, 开始计时。
            # 阈值放宽到 0.05 rad/s 以容忍伺服保持时的微小抖动。
            if (max_velocity < self._velocity_threshold) and \
                    (max_delta < self._joint_delta_safety):
                self._stationary_start_time = now
                self._stationary_reference = joints.copy()
            return False

        # ── 阶段B: 已进入静止, 用"总漂移"维持判定 ──
        # 抖动是来回摆动 (总位移≈0) 天然被容忍; 缓慢单调漂移会累计超出保持带。
        total_drift = np.max(np.abs(joints - self._stationary_reference))
        self._last_drift = total_drift

        # 阶段B维持判定仅用"窗口内漂移" (配合周期重锁):
        #   - 真·运动由阶段A的速度阈值 (进入判定) 与漂移带兜底: 运动速度
        #     > 2.9°/s 无法进入静止; 进入后若每窗口漂移 > 带 (≈5°/s) 也会重置;
        #   - 不再叠加瞬时速度检查: 伺服保持的慢振荡 (~1-2°/s) 会被瞬时速度
        #     偶发超阈值而反复重置, 导致静止永不触发 (2026-08-06 复位卡死根因)。
        if total_drift > self._stationary_hold_band:
            # 漂移超出保持带 → 判定为运动, 重置静止计时
            if self._stationary_elapsed > 0.0:
                logger.debug(
                    f"[Stationary] 静止期间漂移超出保持带, 静止计时重置 "
                    f"(曾静止 {self._stationary_elapsed:.1f}s, 总漂移={total_drift:.5f} rad)"
                )
            self._stationary_start_time = None
            self._stationary_reference = None
            self._stationary_elapsed = 0.0
            self._is_stationary = False
            self._stationary_triggered = False
            return False

        # 周期重锁参考: 伺服保持时机械臂存在 ±1° 量级的物理抖动/慢漂移,
        # 若只从进入静止时锁定参考, 累计漂移会缓慢增长并反复超出保持带,
        # 导致静止计时永远达不到阈值 (复位后模型切换卡死, 2026-08-06 实测)。
        # 每 renew_interval 重锁一次, 将"总漂移"退化为"窗口内漂移":
        # 只要速度低于阈值 (且每窗口漂移 < 带), 缓慢漂移也正确判定为静止。
        if now - self._last_renew_time >= self._renew_interval:
            self._stationary_reference = joints.copy()
            self._last_renew_time = now

        # 保持带内 → 持续累计静止时间
        self._stationary_elapsed = now - self._stationary_start_time

        # 检查是否超过阈值 (快速重评估模式使用更短阈值, 加速切换)
        effective_threshold = (
            self._fast_recheck_time if self._fast_recheck_enabled else self._time_threshold
        )
        if self._stationary_elapsed >= effective_threshold:
            if not self._stationary_triggered:
                # 检查冷却时间
                if now - self._last_trigger_time >= self._cooldown:
                    self._is_stationary = True
                    self._stationary_triggered = True
                    self._last_trigger_time = now
                    self._total_stationary_events += 1

                    mode = "快速" if self._fast_recheck_enabled else "正常"
                    logger.info(
                        f"[Stationary] 🔔 机械臂静止 {self._stationary_elapsed:.1f}s "
                        f"({mode}阈值={effective_threshold:.1f}s, "
                        f"保持带={self._stationary_hold_band:.4f} rad, "
                        f"总漂移={total_drift:.5f} rad), 触发场景重评估"
                    )
                    return True

        return False

    def is_moving(self, joint_positions: np.ndarray) -> bool:
        """
        检测机械臂当前是否处于运动状态 (独立于静止计时, 不改变静止计时状态)。

        用于 VLM 连拍期间的运动监测: 若连拍过程中机械臂恢复运动,
        立即中止连拍, 避免在机械臂运动中切换任务。

        Args:
            joint_positions: 当前 6 个关节角度 (rad)

        Returns:
            True 如果机械臂当前运动速度超过阈值
        """
        joints = np.asarray(joint_positions, dtype=np.float64).ravel()[:6]
        now = time.time()

        if self._motion_prev_joints is None:
            self._motion_prev_joints = joints.copy()
            self._motion_prev_time = now
            return False

        dt = max(now - self._motion_prev_time, 1e-4)
        max_delta = np.max(np.abs(joints - self._motion_prev_joints))
        max_velocity = max_delta / dt
        self._motion_prev_joints = joints.copy()
        self._motion_prev_time = now

        return (max_velocity >= self._velocity_threshold) or \
            (max_delta >= self._joint_delta_safety)

    # ========================================================================
    # 状态查询
    # ========================================================================

    @property
    def is_stationary(self) -> bool:
        """机械臂当前是否处于静止状态"""
        return self._is_stationary

    @property
    def stationary_elapsed(self) -> float:
        """当前静止持续时间 (秒)"""
        return self._stationary_elapsed

    @property
    def last_velocity(self) -> float:
        """最近一次采样的最大关节速度 (rad/s)"""
        return self._last_velocity

    @property
    def last_drift(self) -> float:
        """静止期间相对参考位置的总漂移 (rad)"""
        return self._last_drift

    @property
    def needs_reevaluation(self) -> bool:
        """
        是否需要触发 VLM 重评估。

        与 is_stationary 的区别:
          - is_stationary: 持续状态, 机械臂是否当前静止
          - needs_reevaluation: 一次性事件, 静止阈值刚被触发
        """
        return self._stationary_triggered

    # ========================================================================
    # 重置
    # ========================================================================

    def reset(self):
        """重置检测器状态 (在触发重评估后调用)"""
        self._stationary_start_time = None
        self._stationary_reference = None
        self._stationary_elapsed = 0.0
        self._is_stationary = False
        self._stationary_triggered = False
        self._last_renew_time = 0.0

    def full_reset(self, keep_cooldown: bool = False):
        """完全重置 (包括基准关节位置/时间戳、运动检测基准)。

        Args:
            keep_cooldown: True 时保留静止冷却计时 (_last_trigger_time 不清零)。
                用于连拍结束等场景: 仍需遵守 stationary_recheck_cooldown,
                避免"静止→连拍→立即再触发"的快速循环 (2026-08-07 实测每
                1-2s 一次连拍, 每次暂停机械臂+重新预测 → 周期性抖动)。
        """
        self.reset()
        self._prev_joints = None
        self._prev_time = None
        self._motion_prev_joints = None
        self._motion_prev_time = None
        if not keep_cooldown:
            self._last_trigger_time = 0.0  # 清除冷却, 允许立即重新触发

    # ========================================================================
    # 统计
    # ========================================================================

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "total_stationary_events": self._total_stationary_events,
            "is_stationary": self._is_stationary,
            "stationary_elapsed": self._stationary_elapsed,
            "last_velocity": self._last_velocity,
            "last_drift": self._last_drift,
            "last_trigger_time": self._last_trigger_time,
        }
