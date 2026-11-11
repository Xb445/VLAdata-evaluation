#!/usr/bin/env python3
"""实时监控示教器→夹爪映射关系的调试工具。

连接主臂（示教器）+ 从臂，实时显示：
- 主臂示教器扳机位置（归一化 0~1, 已 clamp）
- 从臂夹爪当前位置（归一化 0~1）
- 从臂夹爪目标位置
- 夹爪运动方向 + 力矩
- 夹爪映射延迟

Usage:
    python tools/piper_gripper_debug.py

依赖:
    conda activate evo-rl
"""

import sys
import time
import os

# ---- 可配置参数 ----
GRIPPER_MAX_RANGE_MM = 70.0   # 小夹爪 70mm, 大夹爪 100mm
GRIPPER_EFFORT = 5000          # 最大力矩 5 N·m
TEACHING_RANGE_PER = 100       # 示教器行程系数 100%
TEACHING_FRICTION = 1          # 示教器摩擦系数 (1=最小)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lerobot.utils.piper_sdk import get_piper_sdk, guard_piper_ctrl_mode_on_connect
from lerobot.utils.piper_sdk import milli_to_unit, unit_to_milli


def main():
    interface_cls, _ = get_piper_sdk()

    # 连接主臂 (can0, 示教器)
    print("🔌 连接主臂 (can0, 示教器)...")
    leader = interface_cls(can_name="can0")
    leader.ConnectPort()
    guard_piper_ctrl_mode_on_connect(arm=leader, interface_name="can0")
    # Disable master-slave mode → standalone for separate CAN buses
    leader.MasterSlaveConfig(0x00, 0x00, 0x00, 0x00)
    leader.EnablePiper()
    time.sleep(0.5)

    # 配置示教器参数（必须！SDK要求首次使用前配置）
    print("⚙️  配置示教器参数...")
    leader.GripperTeachingPendantParamConfig(
        teaching_range_per=TEACHING_RANGE_PER,
        max_range_config=int(GRIPPER_MAX_RANGE_MM),
        teaching_friction=TEACHING_FRICTION,
    )

    # 进入示教模式 (重力补偿, 可拖动)
    leader.MotionCtrl_2(0x01, 0x02, 100, 0x00)
    time.sleep(0.3)

    # 夹爪被动模式 (编码器活跃, 可自由捏合)
    gripper_msg = leader.GetArmGripperMsgs()
    gripper_state = getattr(gripper_msg, "gripper_state", None)
    init_gripper_raw = 0
    if gripper_state is not None:
        init_gripper_raw = abs(int(getattr(gripper_state, "grippers_angle", 0)))
    leader.GripperCtrl(init_gripper_raw, 50, 0x01, 0x00)
    print(f"   初始夹爪位置: {init_gripper_raw} raw ≈ {init_gripper_raw * 0.001 / 70:.4f} (norm)")

    # 连接从臂 (can1)
    print("🔌 连接从臂 (can1)...")
    follower = interface_cls(can_name="can1")
    follower.ConnectPort()
    guard_piper_ctrl_mode_on_connect(arm=follower, interface_name="can1")
    # Disable master-slave mode → standalone for separate CAN buses
    follower.MasterSlaveConfig(0x00, 0x00, 0x00, 0x00)
    follower.EnablePiper()
    follower.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
    time.sleep(0.5)

    print("\n" + "=" * 80)
    print("  实时示教器→夹爪映射监控 (Ctrl+C 退出)")
    print(f"  力矩={GRIPPER_EFFORT} | 行程={GRIPPER_MAX_RANGE_MM}mm | 独立模式")
    print("=" * 80)
    print(f"{'帧':>5s} {'示教器(raw)':>12s} {'示教器(norm)':>13s} "
          f"{'从臂目标':>10s} {'从臂实际':>10s} {'方向':>4s} {'延迟':>6s}")
    print("-" * 80)

    frame = 0
    prev_target_norm = 0.0
    prev_target_time = time.monotonic()
    prev_obs_norm = 0.0

    try:
        while True:
            frame += 1
            t0 = time.monotonic()

            # 1. 读取主臂示教器扳机位置
            gripper_msg = leader.GetArmGripperMsgs()
            gripper_state = getattr(gripper_msg, "gripper_state", None)
            if gripper_state is not None:
                raw = getattr(gripper_state, "grippers_angle", 0)
                leader_norm = abs(milli_to_unit(raw) / GRIPPER_MAX_RANGE_MM)
            else:
                raw = 0
                leader_norm = 0.0

            # ---- Clamp 到 [0, 1] ----
            # 示教器空闲位置可能为负值 (~ -0.015), 捏合到底可能超程 (~ 1.08)
            leader_norm_clamped = max(0.0, min(1.0, leader_norm))

            # 2. 读取从臂夹爪当前位置
            follower_msg = follower.GetArmGripperMsgs()
            fs = getattr(follower_msg, "gripper_state", None)
            if fs is not None:
                follower_norm = abs(milli_to_unit(getattr(fs, "grippers_angle", 0)) / GRIPPER_MAX_RANGE_MM)
            else:
                follower_norm = 0.0
            follower_norm = max(0.0, min(1.0, follower_norm))

            # 3. 判断运动方向
            direction = "⬇闭" if leader_norm_clamped < follower_norm - 0.01 else ("⬆开" if leader_norm_clamped > follower_norm + 0.01 else "◼止")

            # 4. 发送夹爪命令到从臂 (统一使用最大力矩)
            gripper_raw_cmd = unit_to_milli(leader_norm_clamped * GRIPPER_MAX_RANGE_MM)
            follower.GripperCtrl(gripper_raw_cmd, GRIPPER_EFFORT, 0x01, 0x00)

            # 5. 计算延迟
            if abs(leader_norm_clamped - prev_target_norm) > 0.01:
                prev_target_norm = leader_norm_clamped
                prev_target_time = t0
            latency_ms = (t0 - prev_target_time) * 1000 if abs(follower_norm - leader_norm_clamped) > 0.03 else 0

            # 6. 显示
            bar_len = 30
            bar_pos = int(leader_norm_clamped * bar_len)
            bar = "█" * bar_pos + "░" * (bar_len - bar_pos)

            if frame % 5 == 1:
                print(f"{frame:5d} {raw:12d} {leader_norm_clamped:13.4f} "
                      f"{leader_norm_clamped:10.4f} {follower_norm:10.4f} "
                      f"{direction:>4s} {latency_ms:6.0f}  {bar}")

            prev_obs_norm = follower_norm

            # 保持 ~30Hz 循环速率
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, 1.0 / 30.0 - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n\n🛑 退出监控")

    finally:
        leader.DisableArm(7)
        leader.DisconnectPort()
        follower.DisableArm(7)
        follower.DisconnectPort()
        print("✅ 双臂已断开")


if __name__ == "__main__":
    main()
