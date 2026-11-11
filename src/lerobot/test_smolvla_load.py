#!/usr/bin/env python3
"""SmolVLA 模型完整加载诊断脚本（验证 evo-rl-videos 环境可加载模型）。"""
import os
import sys
import time

# 离线加载 SmolVLM2 backbone（本机 HF 缓存）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_CACHE", "/home/ub/VLA/pretrained_models/hub")

sys.path.insert(0, "/home/ub/Videos/Evo-RL/src")

CKPT = "/home/ub/Videos/models/smolVLa_5W_bs64/pretrained_model"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def main():
    t0 = time.time()

    # 1) 加载 config（经基类注册，正确处理 "type" 字段）
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401 触发注册

    config = PreTrainedConfig.from_pretrained(CKPT, local_files_only=True)
    print(f"[1] config 类型: {type(config).__name__} | 原始 vlm_model_name: {config.vlm_model_name}")

    # 2) 覆盖 vlm_model_name -> 本机 HF 缓存的 SmolVLM2 backbone
    config.vlm_model_name = BACKBONE
    print(f"[2] 覆盖 vlm_model_name -> {config.vlm_model_name}")
    print(f"    chunk_size={config.chunk_size} n_action_steps={config.n_action_steps} "
          f"max_state_dim={config.max_state_dim} max_action_dim={config.max_action_dim}")

    # 3) 加载完整 policy（含 backbone 权重 + expert 权重）
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print("[3] 开始加载完整模型（SmolVLM2 backbone + expert），可能需要数分钟 ...")
    policy = SmolVLAPolicy.from_pretrained(CKPT, config=config, local_files_only=True)
    print(f"[3] ✅ policy 加载完成，耗时 {time.time()-t0:.1f}s")

    # 4) 移到 GPU 并设为 eval
    policy.to("cuda").eval()
    print("[4] ✅ policy 已移至 cuda 并设为 eval 模式")

    # 5) 参数统计
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[5] 总参数: {total/1e6:.1f}M | 可训练参数: {trainable/1e6:.1f}M")
    print(f"[5] 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print("\n✅✅ 模型完整加载成功，evo-rl-videos 环境可运行 SmolVLA 推理 ✅✅")


if __name__ == "__main__":
    import torch
    main()
