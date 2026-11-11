#!/usr/bin/env python
"""
================================================================================
Pi0.5 60K checkpoint 离线加载 + 单次推理验证脚本（不连接机械臂）
================================================================================

【用途】
  真机推理前，验证 model/Pimodel/60K/pretrained_model 能否正确：
    1. 适配加载 config.json（PI05Config）
    2. 加载 812 键完整权重（含 74 个 AdaLN .dense 键）
    3. 加载 pre/post processor（本地 tokenizer override，不依赖 /data/zhanghang）
    4. 用真实相机帧或随机图 + 指定状态跑一次 predict_action_chunk
    5. 打印输出动作统计，判断模型行为是否合理（避免白跑一趟真机）

【用法示例】
  # 随机图像 + 训练均值状态（最快验证加载与推理链路）
  python -m lerobot.test_pi05_60k_load --use-random=true

  # 真实相机（wrist=4, top=10）+ 训练均值状态
  python -m lerobot.test_pi05_60k_load

  # 真实相机 + 指定状态 + 指定任务
  python -m lerobot.test_pi05_60k_load \
    --state "0.307 1.304 -1.277 -0.048 0.953 0.116 0.203" \
    --task "Grab the scanned package and place it into the box."

  # 对比确定性采样 vs 随机采样（各推理一次）
  python -m lerobot.test_pi05_60k_load --compare-noise=true

【说明】
  - 模型 dtype=bfloat16（约 4.8GB 显存），RTX 4080 16GB 足够
  - 首次推理含 CUDA 算子编译，可能慢 1-2 秒，属正常
================================================================================
"""

import argparse
import os
import time

import numpy as np
import torch

# ── 复用主推理脚本的加载函数 ──
from lerobot.inference_pi05_adarms_fix import (
    load_0407_config,
    load_pi05_policy_from_0407,
    make_deterministic_noise,
)
from lerobot.policies.factory import make_pre_post_processors

# 本机 60K 模型路径
DEFAULT_CKPT = "/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model"

# 训练 state 均值（batch5_joint_merged，来自 smolvla 排查记录）
DEFAULT_STATE = [0.307, 1.304, -1.277, -0.048, 0.953, 0.116, 0.203]

# 训练数据集三任务
TASKS = [
    "Grab the package and place it on the barcode scanning table.",
    "Flip the package if the barcode is not facing up.",
    "Grab the scanned package and place it into the box.",
]


def build_observation(
    use_random: bool,
    state: list[float],
    device: str,
    task: str,
) -> dict:
    """构造与 prepare_observation 相同格式的观测 dict（不依赖机器人对象）。

    返回 dict 可直接送入 preprocess_fn。
    """
    if use_random:
        # 随机图（0~1 均匀分布），用于快速验证链路
        wrist = np.random.uniform(0.0, 1.0, (480, 640, 3)).astype(np.float32)
        top = np.random.uniform(0.0, 1.0, (480, 640, 3)).astype(np.float32)
        print(f"[INFO] 🎲 使用随机图像（链路验证）")
    else:
        # 真实相机：wrist=/dev/video4, top=/dev/video10
        import cv2

        cams = {}
        for name, idx in (("wrist", 4), ("top", 10)):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开相机 /dev/video{idx}（{name}）")
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"相机 /dev/video{idx}（{name}）读取失败")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cams[name] = frame.astype(np.float32) / 255.0
            print(f"[INFO] 📷 {name} 相机帧: {frame.shape}")
        wrist, top = cams["wrist"], cams["top"]

    # ── 与 inference_pi05_adarms_fix.prepare_observation 相同处理 ──
    frame = {}
    frame["observation.images.wrist"] = torch.tensor(
        np.transpose(wrist, (2, 0, 1))[None],  # HWC→CHW, +batch
        dtype=torch.float32,
        device=device,
    )
    frame["observation.images.top"] = torch.tensor(
        np.transpose(top, (2, 0, 1))[None],
        dtype=torch.float32,
        device=device,
    )
    frame["observation.state"] = torch.tensor(
        np.array(state, dtype=np.float32)[None],
        dtype=torch.float32,
        device=device,
    )
    frame["task"] = [task]
    return frame


def run_inference(policy, preprocess, postprocess, obs, device, chunk_size, max_action_dim, noise):
    """跑一次 predict_action_chunk，返回 (原始动作, 后处理动作)。"""
    processed_obs = preprocess(obs)
    with torch.inference_mode():
        actions = policy.predict_action_chunk(
            processed_obs,
            inference_delay=0,
            execution_horizon=8,
            noise=noise,
        )
    original = actions.squeeze(0).clone()
    post = postprocess(actions).squeeze(0)
    return original, post


def summarize(name: str, actions: torch.Tensor):
    """打印动作 chunk 统计。"""
    a = actions.cpu().numpy()  # (50, 7)
    print(f"\n=== {name} ===")
    print(f"  shape: {a.shape}  (chunk=50, dim=7: [j1..j6, gripper])")
    print(f"  dtype: {a.dtype}")
    for i, jname in enumerate(["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]):
        col = a[:, i]
        print(f"  {jname:7s}: min={col.min():+.4f}  max={col.max():+.4f}  "
              f"mean={col.mean():+.4f}  std={col.std():.4f}  "
              f"首帧={col[0]:+.4f} 末帧={col[-1]:+.4f}")
    # 相邻帧最大跳变（rad / gripper）
    delta = np.max(np.abs(np.diff(a, axis=0)), axis=0)
    print(f"  相邻帧最大跳变(rad): {delta[:6]}")
    print(f"  相邻帧最大跳变(gripper): {delta[6]:.4f}")
    print(f"  总行程(首→末) norm: {np.linalg.norm(a[-1] - a[0]):.4f}")


def main():
    ap = argparse.ArgumentParser(description="Pi0.5 60K 离线加载+推理验证（不接机械臂）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="pretrained_model 目录")
    ap.add_argument("--device", default="cuda", help="推理设备")
    ap.add_argument("--dtype", default="native", choices=["native", "float32", "bfloat16"],
                    help="推理 dtype（默认 native=不统一，保持 checkpoint 设计好的混合 dtype："
                         "action 投影 float32 + paligemma 内部 bfloat16 + layer_norm/patch float32）。"
                         "在 transformers 4.53.3（含 siglip bfloat16 补丁）下 native 即可跑通。")
    ap.add_argument("--use-random", action="store_true", help="使用随机图像（默认用真实相机）")
    ap.add_argument("--task", default=TASKS[2], help="任务文本")
    ap.add_argument("--state", type=str, default=None,
                    help="7 维状态（空格分隔）；默认训练均值")
    ap.add_argument("--compare-noise", action="store_true",
                    help="对比确定性采样 vs 随机采样")
    args = ap.parse_args()

    if args.state:
        state = [float(x) for x in args.state.split()]
        assert len(state) == 7, f"state 需 7 维，收到 {len(state)}"
    else:
        state = DEFAULT_STATE
    print(f"[INFO] 📝 任务: {args.task!r}")
    print(f"[INFO] 📐 状态: {state}")

    # ── 1. 加载配置 ──
    print(f"\n[1/4] 加载 config.json → PI05Config ...")
    config = load_0407_config(args.ckpt)
    print(f"      tokenizer_name = {config.tokenizer_name}")
    print(f"      input_features = {list(config.input_features.keys())}")
    print(f"      chunk_size={config.chunk_size}, max_state_dim={config.max_state_dim}, "
          f"dtype={config.dtype}")

    # ── 2. 加载权重 ──
    print(f"\n[2/4] 加载模型权重（812 键）...")
    t0 = time.time()
    policy = load_pi05_policy_from_0407(args.ckpt)
    print(f"      加载耗时 {time.time()-t0:.1f}s")

    # ── 3. 加载 pre/post processor（tokenizer override 到本地）──
    print(f"\n[3/4] 加载 pre/post processor ...")
    preprocess, postprocess = make_pre_post_processors(
        config,
        args.ckpt,
        preprocessor_overrides={
            "tokenizer_processor": {
                "tokenizer_name": config.tokenizer_name,
            }
        },
    )
    print(f"      ✅ 处理器加载完成")

    policy.eval()
    # ⚠️ 默认 native：不要强制统一 dtype！
    # 60K checkpoint 是设计好的混合 dtype：
    #   - action_in_proj/out_proj/time_mlp_in/out = float32（PI05Pytorch 层）
    #   - paligemma_with_expert 内部 = bfloat16（layer_norm/patch_embedding 保留 float32）
    # 若 policy.to(bfloat16) 会把 action 投影也转 bfloat16，导致 time_emb(float32) ×
    # time_mlp_in(bfloat16) 的 "got Float and BFloat16" 冲突。
    # 4.53.3 的 transformers 自带 siglip/gemma 的 bfloat16 补丁，native 即可跑通。
    if args.dtype in ("float32", "bfloat16"):
        policy.to(torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    policy.to(args.device)
    policy.reset()
    print(f"      模型参数: {sum(p.numel() for p in policy.parameters())/1e9:.2f} B")
    print(f"      推理 dtype: {args.dtype}（native=保持 checkpoint 混合 dtype）")

    # ── 4. 构造观测 + 推理 ──
    print(f"\n[4/4] 构造观测 + 推理 ...")
    obs = build_observation(args.use_random, state, args.device, args.task)

    # CUDA 预热（首次推理编译算子）
    print(f"      🔥 CUDA warmup ...")
    # action_in_proj 是 float32 → noise 用 float32（native 模式）；
    # 若强制 bfloat16 统一，则 noise 也用 bfloat16。
    dtype_t = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    warmup_noise = torch.zeros(1, config.chunk_size, config.max_action_dim,
                               device=args.device, dtype=dtype_t)
    with torch.inference_mode():
        policy.predict_action_chunk(
            preprocess(obs),
            inference_delay=0,
            execution_horizon=8,
            noise=warmup_noise,
        )

    noise_det = torch.zeros(1, config.chunk_size, config.max_action_dim,
                            device=args.device, dtype=dtype_t)
    t0 = time.time()
    orig_det, post_det = run_inference(
        policy, preprocess, postprocess, obs, args.device,
        config.chunk_size, config.max_action_dim, noise_det,
    )
    dt_det = time.time() - t0
    print(f"      确定性采样推理耗时: {dt_det*1000:.0f} ms")
    summarize("确定性采样 - 原始输出（后处理前）", orig_det)
    summarize("确定性采样 - 后处理输出", post_det)

    if args.compare_noise:
        noise_rand = torch.randn(
            1, config.chunk_size, config.max_action_dim, device=args.device
        )
        t0 = time.time()
        orig_rand, post_rand = run_inference(
            policy, preprocess, postprocess, obs, args.device,
            config.chunk_size, config.max_action_dim, noise_rand,
        )
        print(f"\n      随机采样推理耗时: {(time.time()-t0)*1000:.0f} ms")
        summarize("随机采样 - 原始输出（后处理前）", orig_rand)

    print(f"\n{'='*60}")
    print(f"[DONE] 验证完成。若动作统计合理（无 NaN、量级正常），可进入真机推理：")
    print(f"  python -m lerobot.inference_pi05_adarms_fix \\")
    print(f"    --robot.type=single_piper --robot.id=single_piper_robot \\")
    print(f"    --robot.ctrl_mode=joint --robot.port=can3 \\")
    print(f"    --robot.cameras=\"{{wrist: {{type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}},top: {{type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}}}\" \\")
    print(f"    --ckpt_path={args.ckpt} \\")
    print(f"    --raw_mode=true --task_idx=2")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
