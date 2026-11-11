#!/usr/bin/env python3
"""
ACT + VLM 多任务联调编排器 — 启动入口
======================================

用法:
    # 默认配置启动 (加载所有模型, 连接 vLLM + 机械臂 + 相机)
    python -m lerobot.act_vlm_orchestrator.run_orchestrator

    # 仅预加载 Task_A (节省 GPU 显存)
    python -m lerobot.act_vlm_orchestrator.run_orchestrator --default-model Task_A

    # 使用 CPU 推理 (调试用)
    python -m lerobot.act_vlm_orchestrator.run_orchestrator --device cpu

    # 调整投票参数
    python -m lerobot.act_vlm_orchestrator.run_orchestrator --vote-window 7 --vote-threshold 4

    # 详细调试模式
    python -m lerobot.act_vlm_orchestrator.run_orchestrator --debug

    # 指定日志目录
    python -m lerobot.act_vlm_orchestrator.run_orchestrator --log-dir ./my_logs

前置条件:
    1. vLLM 服务已启动:
       VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
         --model /home/lh/Videos/0407/model/Qwen_model/qwen3_vl_2b_parcel_subtask_merged \
         --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 \
         --max-model-len 8192 --max-num-seqs 4 --dtype auto \
         --trust-remote-code --served-model-name qwen3-vl-2b

    2. ACT 模型已训练并存放在:
       /home/lh/Videos/0407/model/Task_A/pretrained_model/
       /home/lh/Videos/0407/model/Task_B/pretrained_model/
       /home/lh/Videos/0407/model/Task_C/pretrained_model/

    3. 相机已连接 (top: ID=10, wrist: ID=4)
    4. Piper 机械臂已连接 (can1)
"""

import argparse
import logging
import math
import os
import sys
import time

# 确保项目路径在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig
from lerobot.act_vlm_orchestrator_v2.orchestrator import ACTVLMOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="ACT + VLM 多任务联调编排器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置
  python -m lerobot.act_vlm_orchestrator.run_orchestrator

  # CPU 推理 (调试)
  python -m lerobot.act_vlm_orchestrator.run_orchestrator --device cpu

  # 自定义投票窗口
  python -m lerobot.act_vlm_orchestrator.run_orchestrator --vote-window 7 --vote-threshold 5

  # 只预加载 Task_A, 其他按需加载 (节省 GPU 显存)
  python -m lerobot.act_vlm_orchestrator.run_orchestrator --default-model Task_A --no-preload-all
        """,
    )

    # ── VLM 配置 ──
    vlm_group = parser.add_argument_group("VLM 服务配置")
    vlm_group.add_argument("--vlm-url", type=str, default=None,
                           help="vLLM API base URL (默认: http://localhost:8000/v1)")
    vlm_group.add_argument("--vlm-model", type=str, default=None,
                           help="vLLM 服务模型名 (默认: qwen3-vl-2b)")
    vlm_group.add_argument("--vlm-timeout", type=float, default=None,
                           help="VLM API 超时秒数 (默认: 8.0)")

    # ── ACT 模型配置 ──
    model_group = parser.add_argument_group("ACT 模型配置")
    model_group.add_argument("--model-a", type=str, default=None,
                             help="Task_A 模型路径")
    model_group.add_argument("--model-b", type=str, default=None,
                             help="Task_B 模型路径")
    model_group.add_argument("--model-c", type=str, default=None,
                             help="Task_C 模型路径")
    model_group.add_argument("--default-model", type=str, default=None,
                             choices=["Task_A", "Task_B", "Task_C"],
                             help="默认激活的 ACT 模型")
    model_group.add_argument("--no-preload-all", action="store_true",
                             help="不预加载所有模型, 按需加载 (节省 GPU 显存)")
    model_group.add_argument("--device", type=str, default=None,
                             choices=["cuda", "cpu"],
                             help="推理设备 (默认: cuda)")
    model_group.add_argument("--temporal-ensemble-coeff", type=float, default=None,
                             help="ACT 时序集成系数, 抑制 chunk 边界抖动 (默认: 0.01, "
                                  "夹爪维度自动透传原始值; 设 0 禁用=队列模式)")
    model_group.add_argument("--no-ensemble-exclude-gripper", action="store_true",
                             help="时序集成时夹爪也参与集成 (默认排除夹爪, 保持开/合信号清晰)")

    # ── 投票机制配置 ──
    vote_group = parser.add_argument_group("窗口投票机制")
    vote_group.add_argument("--vote-window", type=int, default=None,
                            help="投票窗口大小 (默认: 5 帧)")
    vote_group.add_argument("--vote-threshold", type=int, default=None,
                            help="共识阈值 (默认: 3)")

    # ── 静止检测配置 ──
    stationary_group = parser.add_argument_group("静止检测 → VLM 连拍投票")
    stationary_group.add_argument("--stationary-time", type=float, default=None,
                                  help="静止持续时间阈值 (默认: 2.0s)")
    stationary_group.add_argument("--stationary-joint", type=float, default=None,
                                  help="关节变化阈值 rad (默认: 0.01745 ≈ 1°)")
    stationary_group.add_argument("--burst-count", type=int, default=None,
                                  help="静止后 VLM 连拍帧数 (默认: 5)")
    stationary_group.add_argument("--burst-interval", type=float, default=None,
                                  help="VLM 连拍帧间间隔秒数 (默认: 1.5)")
    stationary_group.add_argument("--fast-recheck-time", type=float, default=None,
                                  help="快速重评估静止阈值秒数, 连拍后未切换时加速重评估 (默认: 1.0)")
    stationary_group.add_argument("--no-fast-recheck", action="store_true",
                                  help="禁用快速重评估 (连拍后未切换也保持正常静止阈值)")

    # ── 子任务超时配置 ──
    timeout_group = parser.add_argument_group("子任务超时保护")
    timeout_group.add_argument("--subtask-timeout", type=float, default=None,
                               help="子任务超时秒数, 超时后触发 VLM 重评估 (默认: 30.0, 设为 0 禁用)")
    timeout_group.add_argument("--no-subtask-timeout", action="store_true",
                               help="禁用子任务超时检测")

    # ── 模型切换保护 ──
    switch_group = parser.add_argument_group("模型切换保护")
    switch_group.add_argument("--switch-cooldown", type=float, default=None,
                              help="模型切换后保护时间秒数, 保护期内禁止 VLM 连拍 (默认: 3.0)")

    # ── 运行配置 ──
    run_group = parser.add_argument_group("运行配置")
    run_group.add_argument("--fps", type=float, default=None,
                           help="ACT 推理频率 (默认: 25.0 Hz)")
    run_group.add_argument("--max-time", type=int, default=None,
                           help="最大运行时间秒数 (默认: 36000)")
    run_group.add_argument("--log-dir", type=str, default="./act_vlm_logs",
                           help="日志输出目录 (默认: ./act_vlm_logs)")
    run_group.add_argument("--chunk-transition-duration", type=float, default=None,
                           help="chunk 边界过渡平滑时长秒数, 抑制新 chunk 首动作跳变 "
                                "(默认: 0.25, 设 0 禁用)")
    run_group.add_argument("--hold-band", type=float, default=None,
                           help="保持锁定判定带 rad, 抑制任务完成后的微振荡 "
                                "(默认: 0.03≈1.7°, 设 0 禁用)")
    run_group.add_argument("--hold-steps", type=int, default=None,
                           help="保持锁定连续步数阈值 (默认: 60≈2s @30Hz)")
    run_group.add_argument("--dry-run", action="store_true",
                           help="干跑模式: 连接并读取机械臂, 但只打印动作不发送, "
                                "用于安全验证状态机切换")
    run_group.add_argument("--debug", action="store_true",
                           help="详细调试输出")

    # ── 相机配置 ──
    cam_group = parser.add_argument_group("相机配置")
    cam_group.add_argument("--top-camera", type=int, default=None,
                           help="顶部相机 ID (默认: 5)")
    cam_group.add_argument("--wrist-camera", type=int, default=None,
                           help="腕部相机 ID (默认: 14)")

    # ── 安全配置 ──
    safety_group = parser.add_argument_group("安全配置")
    safety_group.add_argument("--joint-limit", type=float, default=None,
                              help="单步关节跳变安全阈值 rad (默认: 0.10)")
    safety_group.add_argument("--no-smooth", action="store_true",
                              help="禁用轨迹平滑")

    return parser.parse_args()


def setup_logging(log_dir: str, debug: bool = False):
    """配置日志系统"""
    if log_dir is None:
        log_dir = "./act_vlm_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.DEBUG if debug else logging.INFO

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有的 handlers
    root_logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # 文件 handler
    log_file = os.path.join(log_dir, f"orchestrator_{time.strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG 级别
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # 抑制 draccus 配置解析的 DEBUG 刷屏
    # (加载 ACT 模型时 draccus.parsers.decoding 会输出大量
    #  "from_dict for ..." / "Decode name = ..." 单行日志, 每模型 ~50 条)
    logging.getLogger("draccus").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")

    return logger


def build_config(args) -> OrchestratorConfig:
    """
    根据命令行参数构建配置。

    CLI 参数优先级高于 config.py 默认值。
    使用 _override 辅助函数: 仅当 CLI 值为非 None 时才覆盖 config 默认值。
    """
    config = OrchestratorConfig()

    def _override(attr: str, value):
        """仅当 value 非 None 时覆盖 config 属性"""
        if value is not None:
            setattr(config, attr, value)

    # VLM
    _override("vlm_base_url", args.vlm_url)
    _override("vlm_model_name", args.vlm_model)
    _override("vlm_timeout", args.vlm_timeout)

    # ACT 模型路径
    if args.model_a is not None:
        config.model_paths["Task_A"] = args.model_a
    if args.model_b is not None:
        config.model_paths["Task_B"] = args.model_b
    if args.model_c is not None:
        config.model_paths["Task_C"] = args.model_c
    _override("default_model", args.default_model)
    _override("device", args.device)
    # 时序集成: --temporal-ensemble-coeff 0 显式禁用 (0 为 falsy, 直接覆盖)
    if args.temporal_ensemble_coeff is not None:
        config.act_temporal_ensemble_coeff = (
            None if args.temporal_ensemble_coeff == 0 else args.temporal_ensemble_coeff
        )
    # 夹爪是否参与时序集成 (默认排除, 保持开/合信号清晰)
    if args.no_ensemble_exclude_gripper:
        config.act_ensemble_exclude_gripper = False

    # 投票
    _override("vote_window_size", args.vote_window)
    _override("vote_consensus_threshold", args.vote_threshold)

    # 静止检测 + 连拍
    _override("stationary_time_threshold", args.stationary_time)
    _override("stationary_joint_threshold", args.stationary_joint)
    _override("vlm_burst_count", args.burst_count)
    _override("vlm_burst_interval", args.burst_interval)
    # 快速重评估
    _override("stationary_fast_recheck_time", args.fast_recheck_time)
    if args.no_fast_recheck:
        config.stationary_fast_recheck_enabled = False

    # 子任务超时
    _override("subtask_timeout", args.subtask_timeout)
    if args.no_subtask_timeout:
        config.subtask_timeout_enabled = False

    # 切换保护
    _override("switch_cooldown", args.switch_cooldown)

    # 运行
    _override("fps", args.fps)
    _override("inference_time_s", args.max_time)
    _override("log_dir", args.log_dir)
    # chunk 边界过渡平滑
    _override("chunk_transition_duration_s", args.chunk_transition_duration)
    # 保持锁定 (抑制任务完成/复位后的微振荡)
    _override("smooth_hold_band_rad", args.hold_band)
    _override("smooth_hold_steps", args.hold_steps)

    # 相机
    _override("top_camera_id", args.top_camera)
    _override("wrist_camera_id", args.wrist_camera)

    # 安全
    _override("joint_limit_rad", args.joint_limit)
    _override("smooth_enabled", False if args.no_smooth else None)

    # 调试
    _override("debug_verbose", args.debug)

    # 干跑模式
    _override("dry_run", args.dry_run)

    return config


def main():
    args = parse_args()
    logger = setup_logging(args.log_dir, args.debug)

    # 构建配置 (config.py 默认值 + CLI 覆盖)
    config = build_config(args)

    logger.info("=" * 70)
    logger.info("  ACT + VLM 多任务联调编排器 v1.0")
    logger.info("=" * 70)
    logger.info(f"  VLM 服务:     {config.vlm_base_url}")
    logger.info(f"  VLM 模型:     {config.vlm_model_name}")
    logger.info(f"  Task_A:       {config.model_paths.get('Task_A', 'N/A')}")
    logger.info(f"  Task_B:       {config.model_paths.get('Task_B', 'N/A')}")
    logger.info(f"  Task_C:       {config.model_paths.get('Task_C', 'N/A')}")
    logger.info(f"  默认模型:     {config.default_model}")
    logger.info(f"  推理设备:     {config.device}")
    logger.info(f"  投票窗口:     {config.vote_window_size} 帧 / 阈值 {config.vote_consensus_threshold}")
    logger.info(f"  静止检测:     {config.stationary_time_threshold}s @ {config.stationary_joint_threshold} rad")
    fast_str = (f"{config.stationary_fast_recheck_time}s"
                if config.stationary_fast_recheck_enabled else "禁用")
    logger.info(f"  快速重评估:   {fast_str} (连拍未切换时加速)")
    logger.info(f"  VLM 连拍:     {config.vlm_burst_count} 帧 × {config.vlm_burst_interval}s 间隔")
    logger.info(f"  推理频率:     {config.fps} Hz")
    logger.info(f"  轨迹平滑:     {'启用' if config.smooth_enabled else '禁用'}")
    logger.info(f"  chunk过渡:    {config.chunk_transition_duration_s}s (抑制 chunk 边界抖动)")
    hold_str = (f"{config.smooth_hold_band_rad*180/math.pi:.2f}° / {config.smooth_hold_steps}步"
                if config.smooth_hold_band_rad > 0 else "禁用")
    logger.info(f"  保持锁定:     {hold_str} (抑制任务完成/复位后微振荡)")
    ens_str = (f"{config.act_temporal_ensemble_coeff}"
               if config.act_temporal_ensemble_coeff else "禁用(队列模式)")
    if config.act_temporal_ensemble_coeff:
        ens_str += f" (夹爪{'排除' if config.act_ensemble_exclude_gripper else '参与'}集成)"
    logger.info(f"  ACT时序集成:  {ens_str}")
    logger.info(f"  夹爪死区:     {config.smooth_gripper_dead_zone}")
    logger.info(f"  干跑模式:     {'✅ 启用 (不发送动作到机械臂)' if config.dry_run else '否 (正常控制机械臂)'}")
    timeout_str = f"{config.subtask_timeout}s" if config.subtask_timeout_enabled and config.subtask_timeout > 0 else "禁用"
    logger.info(f"  子任务超时:   {timeout_str}")
    logger.info(f"  切换保护:     {config.switch_cooldown}s")
    logger.info(f"  日志目录:     {config.log_dir}")
    logger.info("=" * 70)

    # 创建编排器
    orchestrator = ACTVLMOrchestrator(config)

    # 初始化
    if not orchestrator.initialize():
        logger.error("❌ 初始化失败, 退出")
        sys.exit(1)

    # 运行
    orchestrator.run()

    logger.info("程序正常退出")


if __name__ == "__main__":
    main()
