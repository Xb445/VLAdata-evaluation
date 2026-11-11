#!/usr/bin/env python3
"""SmolVLA 动作不明显诊断脚本：
1) 保存当前相机图像（确认场景/图像是否正常）
2) 打印真实状态的归一化前后数值（确认状态预处理）
3) 对不同任务文本各推理一次，打印模型原始输出(归一化)与反归一化动作
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ub/Videos/Evo-RL/src")
from smolvla_inference import SmolVLAInference
from smolvla_real_inference import CameraPair, PiperArm, DEFAULT_TASKS

OUT_DIR = Path("/tmp/smolvla_diag")
OUT_DIR.mkdir(exist_ok=True)


def main():
    # 1) 连接相机并保存图像
    cams = CameraPair(top_id=10, wrist_id=4)
    cams.connect()
    top, wrist = cams.read()
    import cv2
    cv2.imwrite(str(OUT_DIR / "top_raw.png"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / "wrist_raw.png"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
    print(f"[1] 相机图像已保存: {OUT_DIR}/top_raw.png, wrist_raw.png")
    print(f"    top 均值={top.mean():.1f} std={top.std():.1f} | wrist 均值={wrist.mean():.1f} std={wrist.std():.1f}")
    cams.close()

    # 2) 加载模型
    model = SmolVLAInference(
        checkpoint_path="/home/ub/Videos/models/smolVLa_5W_bs64/pretrained_model",
        device="cuda", dtype="float32", mode="select_action",
    )
    model.load()

    # 3) 读真实状态并打印归一化前后
    arm = PiperArm(can_name="can1")
    arm.connect()
    state = arm.get_observation()
    arm.stop()
    print(f"\n[3] 真实状态 (弧度+gripper): {np.round(state, 4)}")

    # 归一化前后
    if model._dataset_stats is not None:
        m = np.array(model._dataset_stats["observation.state"]["mean"], dtype=np.float32)
        s = np.array(model._dataset_stats["observation.state"]["std"], dtype=np.float32)
        print(f"    state mean: {np.round(m,4)}")
        print(f"    state std : {np.round(s,4)}")
        norm = (state - m) / (s + 1e-8)
        print(f"    归一化后  : {np.round(norm,4)}  (模型实际输入)")
    else:
        print("    ⚠ 未加载状态统计量！状态未归一化，模型输入异常！")

    # 4) 对每个任务推理一次，对比输出
    print(f"\n[4] 各任务推理对比（同一状态、同一图像）:")
    top, wrist = top, wrist  # 使用保存的图像
    cams = CameraPair(top_id=10, wrist_id=4)
    cams.connect()
    top_img, wrist_img = cams.read()
    cams.close()

    for i, task in enumerate(DEFAULT_TASKS):
        model.reset()
        # select_action 第一次调用会推理完整 chunk
        action = model.select_action(wrist_img, top_img, state, task=task)
        # 打印模型队列里的完整 chunk 前几帧（观察原始规划幅度）
        queue = model.policy._queues.get("action", None)
        print(f"\n  --- 任务[{i+1}]: {task}")
        print(f"      select_action 第1帧(反归一化): {np.round(action,4)}")
        if queue is not None and len(queue) > 0:
            # 队列存的是归一化动作
            chunk = np.stack(list(queue)[:5])  # (5, 7) 归一化
            am = np.array(model._dataset_stats["action"]["mean"], dtype=np.float32)
            as_ = np.array(model._dataset_stats["action"]["std"], dtype=np.float32)
            chunk_unnorm = chunk * (as_ + 1e-8) + am
            print(f"      队列前5帧(归一化):")
            for j in range(min(5, len(chunk))):
                print(f"        [{j}] norm={np.round(chunk[j],3)} -> unnorm={np.round(chunk_unnorm[j],3)}")


if __name__ == "__main__":
    main()
