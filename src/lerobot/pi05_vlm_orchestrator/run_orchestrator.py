#!/usr/bin/env python3
"""
PI0.5 + VLM 多任务联调编排器 — 启动入口
=========================================

用法:
    # 默认配置启动 (加载唯一 PI 模型, 连接 vLLM + 机械臂 + 相机)
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator

    # 指定 PI 模型路径 (单个模型, 通过 task 文本切换任务)
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator \
        --model /path/to/pretrained_model

    # 指定默认任务
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --default-task Task_B

    # 调整 RTC 参数
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator \
        --rtc-horizon 16 --rtc-guidance 8.0 --rtc-schedule EXP

    # 关闭 RTC (队列模式)
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --no-rtc

    # 使用 CPU 推理 (调试用)
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --device cpu

    # 调整投票参数
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --vote-window 7 --vote-threshold 4

    # 详细调试模式
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --debug

    # 指定日志目录
    python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --log-dir ./my_logs

前置条件:
    1. vLLM 服务已启动:
       VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server \
         --model /home/lh/Videos/0407/model/Qwen_model/qwen3_vl_2b_parcel_subtask_merged \
         --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.7 \
         --max-model-len 8192 --max-num-seqs 4 --dtype auto \
         --trust-remote-code --served-model-name qwen3-vl-2b

    2. PI0.5 模型已就绪 (单个模型, 支持多任务文本):
       <path>/pretrained_model/

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

from lerobot.pi05_vlm_orchestrator.config import (
    OrchestratorConfig,
    TASK_PROMPTS,
    DEFAULT_MODEL_PATH,
)
from lerobot.pi05_vlm_orchestrator.orchestrator import PIVLMOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(
        description="PI0.5 + VLM 多任务联调编排器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置
  python -m lerobot.pi05_vlm_orchestrator.run_orchestrator

  # 指定 PI 模型路径 (单个模型, task 文本切换任务)
  python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --model .../pretrained_model

  # 关闭 RTC
  python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --no-rtc

  # CPU 推理 (调试)
  python -m lerobot.pi05_vlm_orchestrator.run_orchestrator --device cpu
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

    # ── PI0.5 模型配置 (单模型多任务) ──
    model_group = parser.add_argument_group("PI0.5 模型配置")
    model_group.add_argument("--model", type=str, default=None,
                             help="PI0.5 模型路径 (pretrained_model 目录, 单模型通过 task 文本切换任务)")
    model_group.add_argument("--default-task", dest="default_task", type=str, default=None,
                             choices=["Task_A", "Task_B", "Task_C"],
                             help="默认激活的任务 (默认: Task_A)")
    model_group.add_argument("--device", type=str, default=None,
                             choices=["cuda", "cpu"],
                             help="推理设备 (默认: cuda)")

    # ── PI0.5 RTC 配置 ──
    rtc_group = parser.add_argument_group("PI0.5 RTC 配置")
    rtc_group.add_argument("--rtc-enabled", dest="rtc_enabled", action="store_true", default=None,
                           help="启用 RTC (默认启用)")
    rtc_group.add_argument("--no-rtc", dest="rtc_enabled", action="store_false",
                           help="禁用 RTC (队列模式)")
    rtc_group.add_argument("--rtc-horizon", type=int, default=None,
                           help="RTC 对齐窗口 execution_horizon (默认: 16)")
    rtc_group.add_argument("--rtc-guidance", type=float, default=None,
                           help="RTC 引导强度 max_guidance_weight (默认: 8.0)")
    rtc_group.add_argument("--rtc-schedule", type=str, default=None,
                           choices=["EXP", "LINEAR", "CONST"],
                           help="RTC 前缀注意力调度 (默认: EXP)")
    rtc_group.add_argument("--rtc-debug", action="store_true",
                           help="RTC 调试输出")

    # ── PI0.5 推理参数 ──
    infer_group = parser.add_argument_group("PI0.5 推理参数")
    infer_group.add_argument("--inference-mode", type=str, default=None,
                             choices=["deterministic", "flow"],
                             help="推理采样模式: deterministic=确定性采样(零噪声, 默认推荐, "
                                  "消除采样随机性抖动); flow=流匹配去噪(随机采样, 保留多模态)")
    infer_group.add_argument("--deterministic-noise", dest="deterministic_noise",
                             action="store_true", default=None,
                             help="启用零噪声确定性采样 (默认启用, 等价 --inference-mode deterministic)")
    infer_group.add_argument("--no-deterministic-noise", dest="deterministic_noise",
                             action="store_false",
                             help="禁用确定性采样 (随机流匹配, 等价 --inference-mode flow)")
    infer_group.add_argument("--queue-threshold", type=int, default=None,
                             help="触发新推理的队列剩余阈值 (默认: 15)")

    # ── 任务预设位姿 (切换后缓慢移动到任务平均初始位姿) ──
    preset_group = parser.add_argument_group("任务预设位姿")
    preset_group.add_argument("--preset-pose", dest="preset_pose_enabled",
                              action="store_true", default=None,
                              help="任务切换后移动到任务对应的平均初始位姿 (默认启用)")
    preset_group.add_argument("--no-preset-pose", dest="preset_pose_enabled",
                              action="store_false",
                              help="禁用预设位姿移动")
    preset_group.add_argument("--preset-steps", type=int, default=None,
                              help="预设位姿移动插值步数 (默认: 100 ≈ 2s)")
    preset_group.add_argument("--preset-step-dt", type=float, default=None,
                              help="预设位姿移动每步间隔秒 (默认: 0.02)")
    preset_group.add_argument("--preset-gripper", type=float, default=None,
                              help="预设位姿夹爪开度 0~1 (默认: 1.0)")
    preset_group.add_argument("--preset-joints-a", nargs=6, type=float, default=None,
                              help="Task_A 平均初始位姿 (6 关节 rad, 默认取训练 state 均值)")
    preset_group.add_argument("--preset-joints-b", nargs=6, type=float, default=None,
                              help="Task_B 平均初始位姿 (6 关节 rad, 默认取训练 state 均值)")
    preset_group.add_argument("--preset-joints-c", nargs=6, type=float, default=None,
                              help="Task_C 平均初始位姿 (6 关节 rad, 默认取训练 state 均值)")
    preset_group.add_argument("--standby-preset-task", type=str, default=None,
                              choices=["Task_A", "Task_B", "Task_C"],
                              help="待机 (包裹在盒子) 时使用的初始位姿对应任务 (默认: Task_A)")
    preset_group.add_argument("--preset-fallback", nargs=6, type=float, default=None,
                              help="任务匹配失败时的兜底位姿 (6 关节 rad)")

    # ── 运动参数 (对齐 HIL 采集) ──
    motion_group = parser.add_argument_group("运动参数")
    motion_group.add_argument("--motion-speed", type=int, default=None,
                              help="运动速度百分比, 与 HIL 采集一致 (默认: 100)")
    motion_group.add_argument("--motion-high-follow", dest="motion_high_follow",
                              action="store_true", default=None,
                              help="MIT/高跟随模式 mit_mode=0xAD (默认启用, 与采集一致)")
    motion_group.add_argument("--no-motion-high-follow", dest="motion_high_follow",
                              action="store_false",
                              help="关闭 MIT/高跟随 (mit_mode=0x00)")

    # ── 流匹配 EMA / 夹爪增强 / 跳变报警 ──
    opt_group = parser.add_argument_group("后处理优化")
    opt_group.add_argument("--flow-ema-alpha", type=float, default=None,
                           help="流匹配去噪模式下的轻量 EMA 系数 (默认: 0.2, 仅关节, 0 禁用; "
                                "确定性采样下自动不启用)")
    opt_group.add_argument("--gripper-gain", type=float, default=None,
                           help="夹爪闭合增强增益 (默认: 0.40, 0 禁用, 仅二值模式关闭时生效)")
    opt_group.add_argument("--gripper-gain-below", type=float, default=None,
                           help="夹爪闭合增强启用阈值 (默认: 0.25)")
    opt_group.add_argument("--gripper-gain-above", type=float, default=None,
                           help="夹爪闭合增强恢复阈值 (默认: 0.50)")
    opt_group.add_argument("--jump-alert-deg", type=float, default=None,
                           help="关节相邻步跳变报警阈值(度) (默认: 5.0)")
    opt_group.add_argument("--gripper-binary", dest="gripper_binary", action="store_true", default=None,
                           help="夹爪二值滞回控制 (默认启用, 解决夹不到/闭合慢)")
    opt_group.add_argument("--no-gripper-binary", dest="gripper_binary", action="store_false",
                           help="回退到连续夹爪模式 (gripper_gain 压缩 + GripperFilter)")
    opt_group.add_argument("--gripper-open-above", type=float, default=None,
                           help="夹爪意图>此值→开 (二值滞回, 默认: 0.55)")
    opt_group.add_argument("--gripper-close-below", type=float, default=None,
                           help="夹爪意图<此值→闭 (二值滞回, 默认: 0.30)")
    opt_group.add_argument("--gripper-min-dwell", type=int, default=None,
                           help="夹爪状态切换最小驻留步数 (防扇动, 默认: 40)")
    opt_group.add_argument("--gripper-jump-alert", type=float, default=None,
                           help="夹爪相邻步跳变报警阈值 (0~1 比例, 默认: 0.15)")
    opt_group.add_argument("--diag", dest="diag_enabled", action="store_true", default=None,
                           help="启用诊断日志 (默认启用)")
    opt_group.add_argument("--no-diag", dest="diag_enabled", action="store_false",
                           help="禁用诊断日志")

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
                           help="控制频率 (默认: 20.0 Hz, 对齐 pi05 推理脚本)")
    run_group.add_argument("--max-time", type=int, default=None,
                           help="最大运行时间秒数 (默认: 36000)")
    run_group.add_argument("--log-dir", type=str, default="./act_vlm_logs",
                           help="日志输出目录 (默认: ./act_vlm_logs)")
    run_group.add_argument("--chunk-transition-duration", type=float, default=None,
                           help="chunk 边界过渡平滑时长秒数 (兼容保留, 恒为 0)")
    run_group.add_argument("--hold-band", type=float, default=None,
                           help="保持锁定判定带 rad, 抑制任务完成后的微振荡 "
                                "(默认: 0.06≈3.4°, 设 0 禁用)")
    run_group.add_argument("--hold-steps", type=int, default=None,
                           help="保持锁定连续步数阈值 (默认: 45)")
    run_group.add_argument("--boundary-max-delta", type=float, default=None,
                           help="chunk 边界过渡每步关节限幅 rad/步 (兼容保留)")
    run_group.add_argument("--gripper-dead-zone", type=float, default=None,
                           help="夹爪死区阈值 (归一化 [0,1]), 低于此值的开爪信号被抑制 "
                                "(默认: 0.15)")
    run_group.add_argument("--force-close-threshold", type=float, default=None,
                           help="夹爪强制闭合阈值 (映射值), 低于此值强制闭合 "
                                "(默认: 35000≈raw<0.5)")
    run_group.add_argument("--gripper-boost-exponent", type=float, default=None,
                           help="夹爪开爪信号非线性放大指数 (默认: 1.0=线性)")
    run_group.add_argument("--dry-run", action="store_true",
                           help="干跑模式: 连接并读取机械臂, 但只打印动作不发送, "
                                "用于安全验证状态机切换")
    run_group.add_argument("--debug", action="store_true",
                           help="详细调试输出")

    # ── 相机配置 ──
    cam_group = parser.add_argument_group("相机配置")
    cam_group.add_argument("--top-camera", type=int, default=None,
                           help="顶部相机 ID (默认: 10)")
    cam_group.add_argument("--wrist-camera", type=int, default=None,
                           help="腕部相机 ID (默认: 4)")

    # ── 安全配置 ──
    safety_group = parser.add_argument_group("安全配置")
    safety_group.add_argument("--joint-limit", type=float, default=None,
                              help="单步关节跳变安全阈值 rad (默认: 0.10, 同步 flowmatch 调优)")
    safety_group.add_argument("--safety-gripper-limit", type=float, default=None,
                              help="安全监控夹爪单步最大变化 0~1 (默认: 0.35)")
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

    # PI 模型路径
    _override("model_path", args.model)
    _override("default_model", args.default_task)
    _override("device", args.device)

    # RTC
    if args.rtc_enabled is not None:
        config.rtc_enabled = args.rtc_enabled
    _override("rtc_execution_horizon", args.rtc_horizon)
    _override("rtc_max_guidance_weight", args.rtc_guidance)
    _override("rtc_prefix_attention_schedule", args.rtc_schedule)
    if args.rtc_debug:
        config.rtc_debug = True

    # PI 推理参数
    if args.inference_mode is not None:
        config.pi_deterministic_noise = (args.inference_mode == "deterministic")
    if args.deterministic_noise is not None:
        config.pi_deterministic_noise = args.deterministic_noise
    _override("pi_action_queue_threshold", args.queue_threshold)

    # 任务预设位姿
    _override("preset_pose_enabled", args.preset_pose_enabled)
    _override("preset_pose_steps", args.preset_steps)
    _override("preset_pose_step_dt", args.preset_step_dt)
    _override("preset_pose_gripper", args.preset_gripper)
    if args.preset_joints_a is not None:
        config.preset_poses["Task_A"] = tuple(args.preset_joints_a)
    if args.preset_joints_b is not None:
        config.preset_poses["Task_B"] = tuple(args.preset_joints_b)
    if args.preset_joints_c is not None:
        config.preset_poses["Task_C"] = tuple(args.preset_joints_c)
    _override("standby_preset_task", args.standby_preset_task)
    if args.preset_fallback is not None:
        config.preset_pose_fallback = tuple(args.preset_fallback)

    # 运动参数 (对齐 HIL 采集)
    _override("robot_motion_speed", args.motion_speed)
    if args.motion_high_follow is not None:
        config.robot_motion_high_follow = args.motion_high_follow

    # 后处理优化: 流匹配 EMA / 夹爪 / 跳变报警 / 诊断
    _override("flow_ema_alpha", args.flow_ema_alpha)
    _override("gripper_gain", args.gripper_gain)
    _override("gripper_gain_below", args.gripper_gain_below)
    _override("gripper_gain_above", args.gripper_gain_above)
    # 夹爪二值滞回 / 跳变报警 (同步 flowmatch 调优)
    if args.gripper_binary is not None:
        config.gripper_binary = args.gripper_binary
    _override("gripper_open_above", args.gripper_open_above)
    _override("gripper_close_below", args.gripper_close_below)
    _override("gripper_min_dwell", args.gripper_min_dwell)
    _override("jump_alert_deg", args.jump_alert_deg)
    _override("gripper_jump_alert", args.gripper_jump_alert)
    if args.diag_enabled is not None:
        config.diag_enabled = args.diag_enabled

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
    # chunk 边界过渡平滑 (兼容保留)
    _override("chunk_transition_duration_s", args.chunk_transition_duration)
    # 保持锁定 (抑制任务完成/复位后的微振荡)
    _override("smooth_hold_band_rad", args.hold_band)
    _override("smooth_hold_steps", args.hold_steps)
    # chunk 边界限幅 / 夹爪死区 / 强制闭合 / 开爪放大
    _override("smooth_boundary_max_delta_rad", args.boundary_max_delta)
    _override("smooth_gripper_dead_zone", args.gripper_dead_zone)
    _override("gripper_force_close_threshold", args.force_close_threshold)
    _override("gripper_boost_exponent", args.gripper_boost_exponent)

    # 相机
    _override("top_camera_id", args.top_camera)
    _override("wrist_camera_id", args.wrist_camera)

    # 安全
    _override("joint_limit_rad", args.joint_limit)
    _override("safety_gripper_limit", args.safety_gripper_limit)
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
    logger.info("  PI0.5 + VLM 多任务联调编排器 v1.1 (单模型多任务)")
    logger.info("=" * 70)
    logger.info(f"  VLM 服务:     {config.vlm_base_url}")
    logger.info(f"  VLM 模型:     {config.vlm_model_name}")
    logger.info(f"  PI 模型:      {config.model_path}")
    logger.info(f"  默认任务:     {config.default_model}")
    logger.info(f"  任务文本映射:")
    for task_name, prompt in TASK_PROMPTS.items():
        logger.info(f"    {task_name}: {prompt!r}")
    logger.info(f"  推理设备:     {config.device}")
    logger.info(f"  RTC:          {'启用' if config.rtc_enabled else '禁用'} "
                f"(horizon={config.rtc_execution_horizon}, "
                f"guidance={config.rtc_max_guidance_weight}, "
                f"schedule={config.rtc_prefix_attention_schedule})")
    infer_mode = "确定性采样(零噪声)" if config.pi_deterministic_noise else "流匹配去噪(随机采样)"
    logger.info(f"  推理模式:     {infer_mode}")
    logger.info(f"  队列阈值:     {config.pi_action_queue_threshold}")
    logger.info(f"  预设位姿:     {'启用' if config.preset_pose_enabled else '禁用'} "
                f"(切换后移动到任务平均初始位姿, {config.preset_pose_steps}步×{config.preset_pose_step_dt}s)")
    for task_name in ("Task_A", "Task_B", "Task_C"):
        pose = config.preset_poses.get(task_name)
        if pose:
            pose_str = ",".join(f"{j:.3f}" for j in pose)
            logger.info(f"    {task_name} 预设位姿(rad): [{pose_str}]")
    logger.info(f"  待机位姿:     {config.standby_preset_task} 的初始位姿")
    logger.info(f"  运动参数:     速度{config.robot_motion_speed}% | "
                f"MIT/高跟随={'开(0xAD)' if config.robot_motion_high_follow else '关(0x00)'}")
    flow_ema_str = (f"开(α={config.flow_ema_alpha})"
                    if config.flow_ema_alpha > 0 and not config.pi_deterministic_noise
                    else "关")
    logger.info(f"  流匹配EMA:    {flow_ema_str} (仅流匹配模式)")
    gain_str = (f"<{config.gripper_gain_below}×{config.gripper_gain} "
                f"→>{config.gripper_gain_above} 恢复" if config.gripper_gain > 0 else "禁用")
    logger.info(f"  夹爪闭合增强: {gain_str}")
    logger.info(f"  跳变报警:     >{config.jump_alert_deg}°")
    logger.info(f"  诊断日志:     {'启用' if config.diag_enabled else '禁用'}")
    logger.info(f"  投票窗口:     {config.vote_window_size} 帧 / 阈值 {config.vote_consensus_threshold}")
    logger.info(f"  静止检测:     {config.stationary_time_threshold}s @ {config.stationary_joint_threshold} rad")
    fast_str = (f"{config.stationary_fast_recheck_time}s"
                if config.stationary_fast_recheck_enabled else "禁用")
    logger.info(f"  快速重评估:   {fast_str} (连拍未切换时加速)")
    logger.info(f"  VLM 连拍:     {config.vlm_burst_count} 帧 × {config.vlm_burst_interval}s 间隔")
    logger.info(f"  控制频率:     {config.fps} Hz")
    logger.info(f"  轨迹平滑:     {'启用' if config.smooth_enabled else '禁用'}")
    hold_str = (f"{config.smooth_hold_band_rad*180/math.pi:.2f}° / {config.smooth_hold_steps}步"
                if config.smooth_hold_band_rad > 0 else "禁用")
    logger.info(f"  保持锁定:     {hold_str} (抑制任务完成/复位后微振荡)")
    logger.info(f"  夹爪死区:     {config.smooth_gripper_dead_zone}")
    logger.info(f"  夹爪放大:     {config.gripper_boost_exponent} (0.5=√放大, 1.0=线性)")
    logger.info(f"  夹爪强制闭合: {config.gripper_force_close_threshold} "
                f"(≈raw<{config.gripper_force_close_threshold/(config.gripper_max_open_mm*1000):.3f})")
    logger.info(f"  干跑模式:     {'✅ 启用 (不发送动作到机械臂)' if config.dry_run else '否 (正常控制机械臂)'}")
    timeout_str = f"{config.subtask_timeout}s" if config.subtask_timeout_enabled and config.subtask_timeout > 0 else "禁用"
    logger.info(f"  子任务超时:   {timeout_str}")
    logger.info(f"  切换保护:     {config.switch_cooldown}s")
    logger.info(f"  日志目录:     {config.log_dir}")
    logger.info("=" * 70)

    # 创建编排器
    orchestrator = PIVLMOrchestrator(config)

    # 初始化
    if not orchestrator.initialize():
        logger.error("❌ 初始化失败, 退出")
        sys.exit(1)

    # 运行
    orchestrator.run()

    logger.info("程序正常退出")


if __name__ == "__main__":
    main()
