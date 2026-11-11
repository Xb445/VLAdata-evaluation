#!/usr/bin/env python3
"""
Piper 机械臂 CAN 通信只读测试 (阶段 0)

安全说明:
    - 仅连接 + 读取状态 (使能状态 / 关节角度 / 夹爪)
    - 不使能、不发送任何运动指令
    - 通过即可进入 orchestrator 的 --dry-run 阶段

用法:
    conda activate evo-rl-videos
    python tools/test_piper_connect.py          # 默认 can1
    python tools/test_piper_connect.py can0     # 指定端口
"""
import sys
import time


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "can1"
    print(f"[Piper] 连接 CAN 端口: {port} ...")

    # 注意: 正确 API 是 C_PiperInterface(port) + ConnectPort(),
    # 不是 Connect() (老版本/其他厂商 SDK 才是 Connect)
    from piper_sdk import C_PiperInterface

    piper = C_PiperInterface(port)
    piper.ConnectPort()

    try:
        # 等待读取线程收到第一批 CAN 数据
        time.sleep(1.0)

        # ── 1. 使能状态 ──
        info = piper.GetArmLowSpdInfoMsgs()
        enables = [
            info.motor_1.foc_status.driver_enable_status,
            info.motor_2.foc_status.driver_enable_status,
            info.motor_3.foc_status.driver_enable_status,
            info.motor_4.foc_status.driver_enable_status,
            info.motor_5.foc_status.driver_enable_status,
            info.motor_6.foc_status.driver_enable_status,
        ]
        print(f"[Piper] 使能状态: {enables}  (全部 True = 已使能)")

        # ── 2. 关节角度 (0.001° → °) ──
        joint = piper.GetArmJointMsgs()
        j = joint.joint_state
        joints_deg = [
            j.joint_1 * 0.001, j.joint_2 * 0.001, j.joint_3 * 0.001,
            j.joint_4 * 0.001, j.joint_5 * 0.001, j.joint_6 * 0.001,
        ]
        print(f"[Piper] 关节角度(°): {[round(x, 2) for x in joints_deg]}")

        # ── 3. 夹爪 ──
        gripper = piper.GetArmGripperMsgs()
        print(f"[Piper] 夹爪: {gripper.gripper_state.grippers_angle * 0.001:.2f} mm")

        print("\n✅ CAN 通信正常, 可以进入 orchestrator 的 --dry-run 阶段测试")
    finally:
        piper.DisconnectPort()
        print("[Piper] 已断开连接")


if __name__ == "__main__":
    main()
