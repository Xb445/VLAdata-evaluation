#!/usr/bin/env python3
"""
ACT 模型自检工具 — 在当前真实场景下检查各 Task 模型是否输出有效动作

用途:
    快速定位"机械臂不动"是模型问题还是场景问题。
    对每个 Task 模型, 抓取当前相机 + 关节状态, 输出其首次 chunk 预测的前几步动作。
    若模型输出 ≈ 当前姿态 (差异 <1°) 且夹爪预测闭合 → 该模型在当前场景输出"保持",
    说明模型/场景不匹配, 而非 orchestrator 调度或平滑/时序集成问题。

用法:
    conda activate evo-rl-videos
    python tools/check_act_models.py                # 检查全部 3 个模型
    python tools/check_act_models.py Task_A         # 只检查 Task_A
"""
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, "/home/ub/Videos/Evo-RL/src")

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.act_vlm_orchestrator_v2.camera_manager import CameraManager
from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig, DEFAULT_MODEL_PATHS

TASK_PROMPTS = {
    "Task_A": "Grab the parcel from the table and place it on the tray",
    "Task_B": "Flip the parcel on the tray to make the label visible",
    "Task_C": "Pick up the parcel from the tray and place it into the box",
}


def main():
    targets = sys.argv[1:] or ["Task_A", "Task_B", "Task_C"]

    # ── 抓取真实观测 ──
    cfg = OrchestratorConfig()
    cfg.top_camera_id, cfg.wrist_camera_id = 10, 4
    cam = CameraManager(cfg)
    cam.connect(); cam.start_async(); time.sleep(1.0)
    top, wrist = cam.capture_vlm_images()
    cam.disconnect()
    if top is None or wrist is None:
        print("❌ 相机图像获取失败")
        sys.exit(1)

    from piper_sdk import C_PiperInterface
    p = C_PiperInterface("can1"); p.ConnectPort(); time.sleep(0.5)
    js = p.GetArmJointMsgs().joint_state
    joints = np.array([js.joint_1, js.joint_2, js.joint_3,
                       js.joint_4, js.joint_5, js.joint_6], dtype=np.float32) * 0.001 / 57.3
    gr = p.GetArmGripperMsgs().gripper_state.grippers_angle * 0.001 / 70.0
    p.DisconnectPort()
    state = np.concatenate([joints, [gr]]).astype(np.float32)

    print(f"当前关节(°): {[round(float(x)*180/3.14159, 1) for x in joints]}")
    print(f"当前夹爪(归一化): {float(gr):.3f}")
    print("=" * 70)

    for task in targets:
        path = DEFAULT_MODEL_PATHS.get(task)
        if not path:
            print(f"❌ {task} 路径未配置")
            continue
        print(f"===== {task}  ({path}) =====")
        try:
            policy = ACTPolicy.from_pretrained(path)
            pre, post = make_pre_post_processors(policy.config, path)
            policy.eval(); policy.to("cuda"); policy.reset()

            batch = {
                "observation.state": torch.tensor(state[None], device="cuda"),
                "observation.images.top": torch.tensor(top[None] / 255.0).permute(0, 3, 1, 2).float().to("cuda"),
                "observation.images.wrist": torch.tensor(wrist[None] / 255.0).permute(0, 3, 1, 2).float().to("cuda"),
                "task": [TASK_PROMPTS.get(task, "Perform the manipulation task")],
            }
            batch = pre(batch)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch)
            chunk = post(chunk).squeeze(0).cpu().numpy()

            print(f"  chunk前3步动作(°): ")
            for k in range(3):
                a = chunk[k]
                diff = np.max(np.abs(a[:6] - joints))
                print(f"    步{k+1}: {[round(float(x)*180/3.14159, 1) for x in a[:6]]}, "
                      f"夹爪={float(a[6]):.3f}, 与当前最大差异={diff*180/3.14159:.2f}°")
            # 判断
            max_diff = np.max(np.abs(chunk[0, :6] - joints))
            if max_diff < 0.017 and chunk[0, 6] < 0.1:
                print(f"  ⚠️ 判定: 模型输出≈当前姿态且夹爪闭合 → 该模型在当前场景输出'保持', "
                      f"无法执行任务 (模型/场景不匹配)")
            else:
                print(f"  ✅ 判定: 模型产生有效动作 (与当前最大差异 {max_diff*180/3.14159:.1f}°)")
        except Exception as e:
            print(f"  ❌ 加载/推理失败: {e}")
        print()


if __name__ == "__main__":
    main()
