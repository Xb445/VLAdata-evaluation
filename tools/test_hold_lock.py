"""
保持锁定 (HoldLock) 功能测试
=============================
验证 orchestrator._apply_hold_lock 在"任务完成/复位后模型小幅振荡"场景下
能冻结关节, 且模型真正要运动时能正确解除锁定。

运行:
    ~/anaconda3/envs/evo-rl-videos/bin/python tools/test_hold_lock.py
"""
import os
import sys
import numpy as np

# 确保能 import lerobot 包
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC)

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig
from lerobot.act_vlm_orchestrator_v2.orchestrator import (
    ACTVLMOrchestrator,
    TrajectorySmoother,
)

BASE = np.array([0.0, 1.2, -1.8, 0.5, 0.3, 0.0])  # 保持位姿 (rad)


def make_orch() -> ACTVLMOrchestrator:
    """绕过 __init__ 构造最小实例 (不碰硬件)。"""
    orch = object.__new__(ACTVLMOrchestrator)
    orch.config = OrchestratorConfig()
    orch.config.smooth_hold_band_rad = 0.03   # ≈1.72°
    orch.config.smooth_hold_steps = 60        # ≈2s @30Hz
    orch._hold_lock_count = 0
    orch._hold_lock_active = False
    orch._hold_lock_target = None
    return orch


def test_oscillation_suppression():
    """场景1: 模型保持时输出小幅振荡 (每次重预测偏移 ~0.5°), 应被冻结。"""
    orch = make_orch()
    n = 200
    smoothed_out = []
    for i in range(n):
        # 振荡: 每 50 步"重预测"一次, 目标在 BASE 附近 ±0.01 rad 抖动
        phase = (i // 50) % 2
        offset = 0.01 if phase == 0 else -0.01
        raw = np.concatenate([BASE + offset, [0.0]])
        current = np.concatenate([BASE + offset * 0.5, [0.0]])  # 已跟随一部分
        sm = orch._apply_hold_lock(current, raw, raw.copy())
        smoothed_out.append(sm[:6].copy())

    assert orch._hold_lock_active, "振荡场景应在阈值后激活保持锁定"
    locked = np.array(smoothed_out[orch.config.smooth_hold_steps:])
    # 逐关节最大差值 (避免跨关节的固定姿态差干扰: J2=1.2 vs J3=-1.8)
    per_joint_spread = np.max(np.max(locked, axis=0) - np.min(locked, axis=0))
    # 锁定后输出应完全冻结 (全零 spread 或 <1e-6)
    assert per_joint_spread < 1e-6, f"锁定后输出仍有波动 spread={per_joint_spread:.6f} rad"
    print(f"[PASS] 场景1 振荡抑制: 激活于第 {orch.config.smooth_hold_steps} 步, "
          f"锁定后逐关节 spread={per_joint_spread:.2e} rad")


def test_release_on_real_motion():
    """场景2: 模型真正要运动 (目标偏离保持带), 应解除锁定并跟随。"""
    orch = make_orch()
    # 先建立锁定
    for _ in range(orch.config.smooth_hold_steps + 5):
        raw = np.concatenate([BASE + 0.005, [0.0]])
        orch._apply_hold_lock(np.concatenate([BASE, [0.0]]), raw, raw.copy())
    assert orch._hold_lock_active, "前置: 应已激活锁定"

    # 模型目标大幅偏离 → 应立即解除并跟随
    target = BASE + 0.5
    raw = np.concatenate([target, [0.0]])
    sm = orch._apply_hold_lock(np.concatenate([BASE, [0.0]]), raw, raw.copy())
    assert not orch._hold_lock_active, "模型运动时应解除锁定"
    assert np.allclose(sm[:6], raw[:6]), "解除后应输出原始目标"
    print("[PASS] 场景2 解除锁定: 模型运动 → 跟随原始目标")


def test_gripper_not_locked():
    """场景3: 锁定期内夹爪通道 (idx 6) 不被冻结, 允许开合。"""
    orch = make_orch()
    # 建立锁定
    for _ in range(orch.config.smooth_hold_steps + 5):
        raw = np.concatenate([BASE + 0.005, [0.0]])
        orch._apply_hold_lock(np.concatenate([BASE, [0.0]]), raw, raw.copy())
    assert orch._hold_lock_active

    # 夹爪目标从 0 → 1 (开), 关节保持锁定
    raw = np.concatenate([BASE + 0.005, [1.0]])
    sm = orch._apply_hold_lock(np.concatenate([BASE, [0.0]]), raw, raw.copy())
    assert abs(sm[6] - 1.0) < 1e-9, f"夹爪应跟随, 实际 {sm[6]:.3f}"
    assert np.allclose(sm[:6], orch._hold_lock_target), "关节应保持冻结"
    print("[PASS] 场景3 夹爪不锁定: 锁定期内夹爪可开合, 关节冻结")


def test_with_smoother_end_to_end():
    """场景4: 端到端 (TrajectorySmoother + HoldLock):
    任务完成后模型保持, 平滑器产生小振荡 → HoldLock 冻结; 
    模型重新运动 → 平滑器正常追踪。"""
    orch = make_orch()
    smoother = TrajectorySmoother(
        ema_alphas=orch.config.smooth_ema_alphas,
        rate_limits=orch.config.smooth_rate_limits,
        gripper_dead_zone=orch.config.smooth_gripper_dead_zone,
        enabled=True,
        transition_duration_s=orch.config.switch_transition_duration_s,
        fps=orch.config.fps,
    )
    # 冷启动: 当前关节=初始位
    current = BASE.copy()
    for _ in range(20):
        raw = np.concatenate([current, [0.0]])
        sm = smoother.smooth(raw)
        sm = orch._apply_hold_lock(current, raw, sm)
        current = sm[:6]

    # 任务完成: 模型保持, 但每次 chunk 重预测目标偏移 (振荡)
    n = 200
    raw_traces = []
    out_traces = []
    for i in range(n):
        phase = (i // 50) % 2
        offset = 0.012 if phase == 0 else -0.012
        raw = np.concatenate([BASE + offset, [0.0]])
        sm = smoother.smooth(raw)
        sm = orch._apply_hold_lock(current, raw, sm)
        current = sm[:6]
        raw_traces.append(raw[:6].copy())
        out_traces.append(sm[:6].copy())

    raw_arr = np.array(raw_traces)
    out_arr = np.array(out_traces)
    # 逐关节最大差值
    raw_spread = np.max(np.max(raw_arr, axis=0) - np.min(raw_arr, axis=0))
    # 锁定后输出应冻结
    locked = out_arr[orch.config.smooth_hold_steps:]
    out_spread = np.max(np.max(locked, axis=0) - np.min(locked, axis=0))
    print(f"[INFO] 场景4 原始振荡幅度: {raw_spread:.4f} rad, 输出幅度: {out_spread:.2e} rad")
    assert out_spread < 1e-6, f"端到端锁定后仍有波动 {out_spread:.2e} rad"
    assert orch._hold_lock_active, "任务完成后应处于锁定状态"

    # 模型重新运动: 从保持位移动到抓取位
    move_target = BASE + np.array([0.4, -0.2, 0.1, 0.0, 0.0, 0.0])
    n2 = 60
    released_at = None
    for i in range(n2):
        alpha = min(1.0, (i + 1) / 30.0)
        raw = np.concatenate([BASE + (move_target - BASE) * alpha, [0.0]])
        sm = smoother.smooth(raw)
        sm = orch._apply_hold_lock(current, raw, sm)
        current = sm[:6]
        if released_at is None and not orch._hold_lock_active:
            released_at = i
    assert released_at is not None, "模型运动时应解除锁定"
    # 最终应接近目标位 (平滑收敛)
    err = np.max(np.abs(current - move_target))
    print(f"[PASS] 场景4 端到端: 锁定于保持期, 第 {released_at} 步解除, "
          f"最终误差 {err:.4f} rad")
    assert err < 0.05, f"末端误差过大 {err:.4f} rad"


if __name__ == "__main__":
    test_oscillation_suppression()
    test_release_on_real_motion()
    test_gripper_not_locked()
    test_with_smoother_end_to_end()
    print("\n🎉 保持锁定全部测试通过")
