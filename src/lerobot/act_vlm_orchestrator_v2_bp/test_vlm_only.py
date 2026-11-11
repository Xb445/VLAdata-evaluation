#!/usr/bin/env python3
"""
VLM 场景分类独立测试脚本 — 不依赖机械臂, 仅测试 VLM + 相机联调

用法:
    # 使用实时相机
    python -m lerobot.act_vlm_orchestrator.test_vlm_only --use-cameras

    # 使用单张测试图片
    python -m lerobot.act_vlm_orchestrator.test_vlm_only --image-top /path/to/top.jpg --image-wrist /path/to/wrist.jpg

    # 持续监控模式 (每 2 秒检测一次)
    python -m lerobot.act_vlm_orchestrator.test_vlm_only --use-cameras --continuous --interval 2.0

    # 测试窗口投票机制
    python -m lerobot.act_vlm_orchestrator.test_vlm_only --use-cameras --continuous --vote-window 5 --vote-threshold 3
"""

import argparse
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import cv2

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig, SUBTASK_DESCRIPTIONS
from lerobot.act_vlm_orchestrator_v2.vlm_client import VLMClient
from lerobot.act_vlm_orchestrator_v2.window_voter import WindowVoter
from lerobot.act_vlm_orchestrator_v2.camera_manager import CameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vlm_test")


def parse_args():
    parser = argparse.ArgumentParser(description="VLM 场景分类独立测试")
    parser.add_argument("--use-cameras", action="store_true", help="使用实时相机")
    parser.add_argument("--image-top", type=str, help="顶部图像路径 (测试用)")
    parser.add_argument("--image-wrist", type=str, help="腕部图像路径 (测试用)")
    parser.add_argument("--continuous", action="store_true", help="持续监控模式")
    parser.add_argument("--interval", type=float, default=2.0, help="持续监控间隔 (秒)")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm-model", type=str, default="qwen3-vl-2b")
    parser.add_argument("--top-camera", type=int, default=10)
    parser.add_argument("--wrist-camera", type=int, default=4)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--vote-threshold", type=int, default=3)
    parser.add_argument("--num-tests", type=int, default=1, help="测试次数 (非持续模式)")
    return parser.parse_args()


def test_single(vlm_client: VLMClient, voter: WindowVoter,
                top_img: np.ndarray, wrist_img: np.ndarray, test_idx: int):
    """单次 VLM 分类测试"""
    logger.info(f"\n{'─'*50}")
    logger.info(f"测试 #{test_idx}")
    logger.info(f"{'─'*50}")

    t_start = time.time()
    decision = vlm_client.classify(top_img, wrist_img)
    elapsed = time.time() - t_start

    if decision.success:
        desc = SUBTASK_DESCRIPTIONS.get(decision.subtask, "未知分类")
        logger.info(f"  ✅ 分类成功 ({elapsed*1000:.0f}ms)")
        logger.info(f"  📦 包裹位置:   {decision.parcel_location}")
        logger.info(f"  🏷️  标签可见:   {decision.label_visible}")
        logger.info(f"  🎯 Subtask:    {decision.subtask}")
        logger.info(f"  🤖 ACT 模型:   {decision.target_model}")
        logger.info(f"  📝 描述:       {desc}")
        # 打印模型原始输出 (便于区分: 模型误判 vs 代码解析问题)
        if decision.raw_response:
            logger.info(f"  💬 模型原始输出: {decision.raw_response.strip()[:200]}")
    else:
        logger.error(f"  ❌ 分类失败 ({elapsed*1000:.0f}ms): {decision.error_message}")

    # 窗口投票
    consensus_reached = voter.add_decision(decision)
    ws = voter.window_status
    logger.info(f"  🗳️  投票窗口:   {ws['size']}/{ws['max']}, "
                f"分布: {ws['distribution']}")

    if consensus_reached:
        consensus = voter.get_consensus()
        if consensus:
            subtask, model = consensus
            logger.info(f"  🎯 达成共识!   subtask={subtask} → model={model}")

    return decision


def main():
    args = parse_args()

    # 构建配置
    config = OrchestratorConfig()
    config.vlm_base_url = args.vlm_url
    config.vlm_model_name = args.vlm_model
    config.top_camera_id = args.top_camera
    config.wrist_camera_id = args.wrist_camera
    config.vote_window_size = args.vote_window
    config.vote_consensus_threshold = args.vote_threshold

    # 创建 VLM 客户端
    vlm_client = VLMClient(config)
    voter = WindowVoter(config)

    # 加载测试图像
    if args.image_top and args.image_wrist:
        logger.info(f"📷 从文件加载图像:")
        logger.info(f"   top:   {args.image_top}")
        logger.info(f"   wrist: {args.image_wrist}")
        top_img = cv2.imread(args.image_top)
        wrist_img = cv2.imread(args.image_wrist)

        if top_img is None:
            logger.error(f"无法读取图像: {args.image_top}")
            sys.exit(1)
        if wrist_img is None:
            logger.error(f"无法读取图像: {args.image_wrist}")
            sys.exit(1)

        for i in range(args.num_tests):
            test_single(vlm_client, voter, top_img, wrist_img, i + 1)
            if i < args.num_tests - 1:
                time.sleep(1.0)

    elif args.use_cameras:
        # 使用实时相机
        cam_mgr = CameraManager(config)
        if not cam_mgr.connect():
            logger.error("❌ 相机连接失败")
            sys.exit(1)

        cam_mgr.start_async()
        time.sleep(1.0)

        if args.continuous:
            logger.info(f"🔄 持续监控模式 (间隔 {args.interval}s), 按 Ctrl+C 停止")
            test_idx = 0
            try:
                while True:
                    test_idx += 1
                    top_img, wrist_img = cam_mgr.capture_vlm_images()
                    if top_img is not None and wrist_img is not None:
                        test_single(vlm_client, voter, top_img, wrist_img, test_idx)
                    else:
                        logger.warning("相机图像获取失败")
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("\n用户中断")
        else:
            logger.info("📷 单次实时相机测试")
            top_img, wrist_img = cam_mgr.capture_vlm_images()
            if top_img is not None and wrist_img is not None:
                test_single(vlm_client, voter, top_img, wrist_img, 1)
            else:
                logger.error("相机图像获取失败")

        cam_mgr.disconnect()

    else:
        logger.error("请指定 --use-cameras 或 --image-top/--image-wrist")
        sys.exit(1)

    # 最终统计
    logger.info(f"\n{'='*50}")
    logger.info(f"VLM 统计: {vlm_client.stats}")
    logger.info(f"投票统计: {voter.stats}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
