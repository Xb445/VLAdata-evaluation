#!/usr/bin/env python3
"""
VLM 原始输出诊断脚本 — 抓取真实相机图像并打印 VLM 未解析的原始响应

用途:
    判断"标签可见=True"是 VLM 模型自己的判断, 还是代码解析/推导造成的。

用法:
    conda activate evo-rl-videos
    python tools/test_vlm_raw.py
"""
import os
import sys
import time
import json

import numpy as np
import cv2

sys.path.insert(0, "/home/ub/Videos/Evo-RL/src")

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig
from lerobot.act_vlm_orchestrator_v2.camera_manager import CameraManager
from lerobot.act_vlm_orchestrator_v2.vlm_client import VLMClient, VLM_PROMPT

SAVE_DIR = "/home/ub/Videos/Evo-RL/act_vlm_logs/diag"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    config = OrchestratorConfig()
    config.vlm_base_url = "http://localhost:8000/v1"
    config.vlm_model_name = "qwen3-vl-2b"
    config.top_camera_id = 10
    config.wrist_camera_id = 4

    # 1. 抓取相机图像
    cam_mgr = CameraManager(config)
    if not cam_mgr.connect():
        print("❌ 相机连接失败")
        sys.exit(1)
    cam_mgr.start_async()
    time.sleep(1.0)

    top_img, wrist_img = cam_mgr.capture_vlm_images()
    cam_mgr.disconnect()

    if top_img is None or wrist_img is None:
        print("❌ 图像获取失败")
        sys.exit(1)

    top_path = os.path.join(SAVE_DIR, "top.jpg")
    wrist_path = os.path.join(SAVE_DIR, "wrist.jpg")
    cv2.imwrite(top_path, top_img)
    cv2.imwrite(wrist_path, wrist_img)
    print(f"📷 图像已保存: {top_path}, {wrist_path}")

    # 2. 直接调用 vLLM API, 打印原始响应 (绕过 _parse_response)
    import base64
    from io import BytesIO
    from PIL import Image as PILImage
    import requests

    def _b64(img: np.ndarray) -> str:
        pil_img = PILImage.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": "qwen3-vl-2b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(top_img)}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(wrist_img)}"}},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0.0,
    }

    print("\n" + "=" * 60)
    print("🔍 原始 API 响应 (未经过代码解析):")
    print("=" * 60)
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions", json=payload, timeout=15
    )
    raw_text = resp.json()["choices"][0]["message"]["content"].strip()
    print(f"  模型原始输出: {raw_text}")

    # 3. 同时展示代码解析结果对比
    print("\n" + "=" * 60)
    print("🧩 代码解析后 (VLMClient._parse_response):")
    print("=" * 60)
    client = VLMClient(config)
    decision = client.classify(top_img, wrist_img)
    print(f"  parcel_location: {decision.parcel_location}")
    print(f"  label_visible:   {decision.label_visible}")
    print(f"  subtask:         {decision.subtask}")
    print(f"  target_model:    {decision.target_model}")
    print(f"  raw_response:    {decision.raw_response}")


if __name__ == "__main__":
    main()
