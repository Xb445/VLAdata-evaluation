"""
PI0.5 + VLM 多任务联调编排器 — 配置模块
========================================

本配置面向 "VLM 调度单个 PI0.5 模型切换任务" 的场景：
  - 只加载 ONE 个 PI0.5 模型 (支持多任务的权重)
  - VLM (Qwen3-VL) 对场景分类 → 窗口投票 → 切换该模型的 task 文本
  - PI0.5 通过不同的 task 文本 (任务描述) 条件化输出不同行为
  - 切换任务文本必须与训练时的 task 逐字一致

任务文本映射 (来自各数据集 meta/tasks.parquet)：
  Task_A ← batch5A_joint_merged: "Grab the package and place it on the barcode scanning table."
  Task_B ← batch5B_joint_merged: "Flip the package if the barcode is not facing up."
  Task_C ← batch5C_joint_merged: "Grab the scanned package and place it into the box."
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================================
# 模型路径常量
# ============================================================================
# 唯一的 PI0.5 模型检查点路径 (该模型支持多任务, 通过 task 文本切换)。
# 可通过 CLI 的 --model 覆盖。
DEFAULT_MODEL_PATH = "/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model"

# 本地离线 tokenizer 路径 (Paligemma)。
# checkpoint 的 config.json 里保存的是训练机路径 (如 /data/zhanghang/...),
# 本机不存在, 加载预处理器前必须强制指向本机路径, 否则 tokenizer 加载失败。
LOCAL_TOKENIZER = "/home/ub/Videos/Evo-RL/model/paligemma-3b-pt-224"

# 默认任务提示 (未匹配到具体模型时使用)
DEFAULT_TASK_PROMPT = "Perform the manipulation task"

# 每个 PI 模型对应的训练任务文本 — 必须与训练数据集的 task 逐字一致!
# (来源: datasets/batch5A/B/C_joint_merged/meta/tasks.parquet)
TASK_PROMPTS: Dict[str, str] = {
    "Task_A": "Grab the package and place it on the barcode scanning table.",
    "Task_B": "Flip the package if the barcode is not facing up.",
    "Task_C": "Grab the scanned package and place it into the box.",
}

# 每个任务的平均初始位姿 (6 关节 rad)。
# 任务切换后缓慢移动到该位姿, 让模型初始状态落在训练分布内, 避免"初始位姿
# 偏离训练分布导致无动作"。
# 来源: 2026-08-27 重新统计 batch5A/B/C 各任务首帧 observation.state 的均值
#   (此前 2026-08-26 实测位姿对 Task B/C 匹配较差, j1/j3/j5 等偏训练分布):
#   - Task_A (放到扫码台 "barcode scanning table"): home/原点 (训练首帧 j1≈0, j2≈0)
#   - Task_B (翻面 "Flip"): 工作区, 训练初始均值 j1≈1.188, j2≈-1.411
#   - Task_C (放入纸箱 "box"): 工作区, 训练初始均值 j1≈1.355, j2≈-1.452
PRESET_POSES: Dict[str, tuple] = {
    "Task_A": (-0.012, 0.004, -0.004, 0.045, 0.443, 0.022),   # home/原点
    "Task_B": (0.341, 1.188, -1.411, -0.003, 1.166, -0.002),  # 训练初始均值
    "Task_C": (0.174, 1.355, -1.452, 0.098, 1.188, -0.052),   # 训练初始均值
}

# 兜底位姿 (任务匹配失败时使用, 训练 state 均值)
PRESET_POSES_FALLBACK: tuple = (0.307, 1.304, -1.277, -0.048, 0.953, 0.116)

# standby (包裹在盒子) 时使用的待机位姿 = Task_A 初始位姿
STANDBY_PRESET_TASK = "Task_A"

# VLM subtask → 任务标签映射 (同一模型切换 task 文本)
SUBTASK_TO_MODEL: Dict[str, str] = {
    "subtask1": "Task_A",   # 包裹在桌面 → 抓取到条码扫描台
    "subtask2": "Task_B",   # 包裹在托盘,条码不可见 → 翻转
    "subtask3": "Task_C",   # 包裹在托盘,条码可见 → 放入盒子
    "subtask4": "standby",  # 包裹在盒子 → 待机
}

# 场景分类的中文描述
SUBTASK_DESCRIPTIONS: Dict[str, str] = {
    "subtask1": "包裹在桌面 → Task_A: 抓取到条码扫描台",
    "subtask2": "包裹在托盘,条码不可见 → Task_B: 翻转包裹",
    "subtask3": "包裹在托盘,条码可见 → Task_C: 放入盒子",
    "subtask4": "包裹在盒子 → 待机等待场景复原",
}


@dataclass
class OrchestratorConfig:
    """PI0.5 + VLM 多任务联调编排器配置 (单模型多任务)"""

    # ================================================================
    # VLM 服务配置 (vLLM OpenAI-compatible API)
    # ================================================================
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model_name: str = "qwen3-vl-2b"
    vlm_timeout: float = 8.0
    vlm_max_tokens: int = 128
    vlm_temperature: float = 0.0  # 确定性输出

    # ================================================================
    # PI0.5 模型配置 (单模型多任务)
    # ================================================================
    model_path: str = DEFAULT_MODEL_PATH   # 唯一的 PI0.5 模型路径
    default_model: str = "Task_A"          # 默认任务标签 (Task_A/B/C)
    device: str = "cuda"
    fps: float = 20.0                  # 控制频率 (Hz), 对齐 pi05 推理脚本默认 20Hz
    inference_time_s: int = 36000      # 最大运行时间 (10 小时)

    # ================================================================
    # PI0.5 RTC (Real-Time Chunking) 配置
    # 默认参数来自 inference_pi05_adarms_fix.py 真机调优 (2026-08-25):
    #   H=16 / G=8 为甜点 (H=8/G=4 → 边界跳变 8.2x GT; H=16/G=8 → 1.6x GT)
    # ================================================================
    rtc_enabled: bool = True
    rtc_execution_horizon: int = 16                    # 对齐窗口
    rtc_max_guidance_weight: float = 8.0               # 引导强度
    rtc_prefix_attention_schedule: str = "EXP"         # EXP: 指数衰减快速过渡
    rtc_debug: bool = False

    # PI0.5 推理参数
    # 采样模式 (通过 pi_deterministic_noise 控制):
    #   True  = 确定性采样 (零噪声): 消除流匹配采样随机性, 实测抖动
    #           2.07x→0.86x GT, 零额外延迟。代价: 失去多模态 (模式平均),
    #           gripper 最大约 1.1 不到全开 1.4 (对抓取类任务影响小)。推荐。
    #   False = 流匹配去噪 (随机采样): 保留多模态行为, 但存在采样随机性抖动。
    pi_deterministic_noise: bool = True
    # 触发新推理的队列阈值: 必须 > execution_horizon + inference_delay, 否则断档
    pi_action_queue_threshold: int = 15
    # 流匹配去噪抖动抑制 (仅 pi_deterministic_noise=False 时生效):
    # 流匹配每次采样噪声随机, chunk 间输出不连续; 在主循环对发送动作做轻量
    # EMA (alpha 较高、滞后小), 抑制高频抖动。0 禁用; 确定性采样下自动不启用。
    # ★ 2026-08-27 同步 flowmatch 调优: α 0.5→0.2, 且仅作用于关节
    #   (夹爪交给二值滞回控制器, 避免 EMA 拖慢夹爪闭合)。
    flow_ema_alpha: float = 0.2

    # ================================================================
    # 任务切换后预设位姿移动 (对齐 inference_pi05_adarms_fix.py 的 preset_pose)
    # ================================================================
    # 每个任务的平均初始位姿 (6 关节 rad), 可通过 --preset-joints-a/b/c 覆盖。
    # standby (包裹在盒子) 时用 standby_preset_task 位姿待机。
    preset_poses: Dict[str, tuple] = field(default_factory=lambda: dict(PRESET_POSES))
    preset_pose_fallback: tuple = PRESET_POSES_FALLBACK  # 任务匹配失败时的兜底位姿
    preset_pose_enabled: bool = True      # 任务切换后是否移动到任务平均初始位姿
    preset_pose_steps: int = 100          # 预设移动步数 (默认 100 步 × 0.02s ≈ 2s)
    preset_pose_step_dt: float = 0.02     # 预设移动每步间隔 (秒)
    preset_pose_gripper: float = 1.0      # 预设时夹爪开度 (0~1)
    # standby (包裹在盒子) 时使用的待机位姿对应的任务 (默认 Task_A 初始位姿)
    standby_preset_task: str = STANDBY_PRESET_TASK

    # ================================================================
    # 窗口投票机制配置
    # ================================================================
    vote_window_size: int = 5          # 投票窗口大小
    vote_consensus_threshold: int = 3  # 共识阈值 (≥3/5)

    # ================================================================
    # 静止检测配置
    # ================================================================
    # 两段式判定:
    #   1) 进入判定: 相邻采样速度 < stationary_velocity_threshold 视为"开始静止"
    #   2) 维持判定: 静止期间位置相对锁定参考的总漂移 < stationary_hold_band
    stationary_velocity_threshold: float = 0.05    # 进入静止的速度阈值 (rad/s ≈ 2.9°/s)
    stationary_hold_band: float = 0.045            # 静止保持带 (rad ≈ 2.6°)
    stationary_reference_renew_interval: float = 0.5
    stationary_joint_threshold: float = 0.01745   # 单步 delta 安全上限 (rad), 默认 1°
    stationary_time_threshold: float = 2.0         # 静止持续时间阈值 (秒), 触发 VLM 重评估
    stationary_check_interval: float = 0.05         # 静止检测间隔 (秒)

    # 快速重评估: 一次 VLM 连拍后若共识未变 (未切换模型), 用更短的静止阈值
    # 加速下一轮重评估, 让任务切换更快。
    stationary_fast_recheck_enabled: bool = True
    stationary_fast_recheck_time: float = 1.0      # 快速重评估的静止阈值 (秒)

    # ================================================================
    # VLM 重评估配置
    # ================================================================
    stationary_recheck_cooldown: float = 5.0   # 静止触发重检查后的冷却时间 (秒)

    vlm_burst_count: int = 5            # 静止后连拍帧数 (填充投票窗口)
    vlm_burst_interval: float = 0.1     # 连拍帧间间隔 (秒)
    vlm_burst_total_timeout: float = 5.0  # 连拍整体超时 (秒)

    # 连拍期间运动中止: 若连拍过程中检测到机械臂恢复运动, 立即中止连拍
    burst_abort_on_motion: bool = True

    # VLM 帧新鲜度阈值 (秒)
    vlm_frame_max_age: float = 0.5

    # ================================================================
    # 模型切换保护配置
    # ================================================================
    switch_cooldown: float = 3.0            # 切换后保护时间 (秒)

    # 待机时是否保留模型在 GPU
    standby_keep_on_gpu: bool = True

    # ================================================================
    # 子任务超时配置
    # ================================================================
    subtask_timeout: float = 30.0         # 子任务超时时间 (秒)
    subtask_timeout_enabled: bool = True   # 是否启用子任务超时

    # ================================================================
    # 机器人配置
    # ================================================================
    robot_type: str = "single_piper"
    robot_port: str = "can1"
    robot_ctrl_mode: str = "joint"
    # 运动参数 (★ 与 HIL 采集对齐, 解决 chunk 切换抖动):
    # 采集时从臂 piper_follower: speed_ratio=100, high_follow=True(mit_mode=0xAD)。
    # 原值 (30, 0x00) 机械臂响应慢/模式不同 → 位置滞后 → RTC 的
    # prev_chunk_left_over 与实际位置错位 → chunk 切换抖动。
    robot_motion_speed: int = 100            # 运动速度百分比 (与采集一致)
    robot_motion_high_follow: bool = True    # MIT/高跟随模式 (mit_mode=0xAD)

    # 夹爪闭合增强 (★ 解决"抓取时夹爪闭合不紧"):
    # 模型输出 gripper 在闭合阶段可能停在 0.05~0.25 (物理上夹爪有间隙)。
    # 分段线性压缩: 比例 < gripper_gain_below 时乘 gripper_gain (更紧),
    # 在 [below, above] 区间线性过渡回正常, ≥ above 不压缩。
    # 映射连续无阶跃, 保留夹爪状态平滑变化 (区别于瞬时置 0, 不破坏模型闭环)。
    # gripper_gain=0 可禁用。
    gripper_gain_below: float = 0.25   # 低于该比例启用压缩
    gripper_gain: float = 0.40         # 压缩增益 (实际发送=比例×gain)
    gripper_gain_above: float = 0.50   # 高于该比例恢复正常 (中间线性过渡)

    # ── 夹爪二值滞回控制 (★ 同步 flowmatch 调优, 解决"夹不到"/"闭合缓慢")──
    # 模型输出的夹爪命令常在 chunk 内从全开写到全闭再写回 (夹爪极差>0.8),
    # 导致夹爪快速扇动、抓取时刻无法稳定闭合。用滞回 + 最小驻留时间把意图
    # 转化为稳定的"开/闭"状态: >open_above 判定开, <close_below 判定闭,
    # 中间保持上一状态; 切换后至少驻留 gripper_min_dwell 步。
    # 闭合时直接发送 0 (完全闭合, 跳过平滑/EMA 造成的缓慢闭合)。
    # gripper_binary=false 可回退到原来的连续处理 (gripper_gain 压缩 + GripperFilter)。
    gripper_binary: bool = True
    gripper_open_above: float = 0.55     # 夹爪意图 > 此值 → 开
    gripper_close_below: float = 0.30    # 夹爪意图 < 此值 → 闭
    gripper_min_dwell: int = 40          # 状态切换后最小驻留步数 (防扇动)

    # 相邻步跳变报警阈值:
    #   关节 (度): Piper 正常高速运动每步可达 2~4° (46~80°/s), 默认 5° 只报警
    #   真正的异常跳变 (chunk 边界大跳变), 避免正常运动误报刷屏。
    #   夹爪 (0~1 比例): 与关节分开, 避免夹爪开合把 [JUMP] 刷屏 (0.3 比例
    #   会被误显示成 17.2°)。
    jump_alert_deg: float = 5.0
    gripper_jump_alert: float = 0.15

    # 诊断日志 (对齐 inference_pi05_adarms_fix.py 的 DiagLogger):
    # 记录推理线程与主循环关键指标到 logs/pi05_diag_*.csv, 便于事后分析
    # (模型无动作 / chunk 衔接跳变 / 流匹配抖动)。
    diag_enabled: bool = True

    # 干跑模式: 连接并读取机械臂, 但 _send_action 只打印动作不发送
    dry_run: bool = False

    # ================================================================
    # 相机配置
    # ================================================================
    top_camera_id: int = 10
    wrist_camera_id: int = 4
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ================================================================
    # 安全配置
    # ================================================================
    safe_mode: bool = True
    # 单步关节跳变安全阈值 (rad): 同步 flowmatch 调优, 由 0.15 收紧到 0.10
    # (≈5.7°/步)。编排器为 EMA 平滑 (无硬限幅), 正常发送单步变化 <0.03 rad,
    # 0.10 能捕获真正的异常 (平滑失效 / RAW 大跳变) 而不误触正常运动。
    joint_limit_rad: float = 0.10
    # 夹爪单步最大变化 (0~1): 略高于二值滞回的切换幅度, 只拦截真正异常
    safety_gripper_limit: float = 0.35
    consecutive_warnings_limit: int = 15 # 连续警告次数上限

    # ================================================================
    # 日志配置
    # ================================================================
    log_dir: str = "./act_vlm_logs"
    debug_verbose: bool = False          # 详细调试输出

    # ================================================================
    # 夹爪配置 (传递给推理 / 发送)
    # ================================================================
    gripper_clip_min: float = 0.0
    gripper_clip_max: float = 1.0
    gripper_max_open_mm: float = 70.0
    # 夹爪强制闭合阈值 (映射值): mapped = raw × max_open_mm × 1000
    gripper_force_close_threshold: float = 35000.0
    gripper_force_close_value: float = 0.0
    # 夹爪开爪信号非线性放大指数 (0~1): 默认 1.0 = 禁用放大 (线性映射)
    gripper_boost_exponent: float = 1.0

    # ================================================================
    # 平滑配置 (对齐 pi05 真机推理: 纯 EMA + 选择性速率限制)
    # ================================================================
    smooth_enabled: bool = True
    # 逐关节 EMA alpha: [J1, J2, J3, J4, J5, J6, gripper]
    smooth_ema_alphas: list = field(default_factory=lambda: [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50])
    # 逐关节速率限制 (rad/帧): None=不限制
    smooth_rate_limits: list = field(default_factory=lambda: [None, 0.015, 0.015, None, None, None, None])
    # 夹爪死区 (归一化 [0,1])
    smooth_gripper_dead_zone: float = 0.15
    # 模型切换过渡平滑时长 (秒)
    switch_transition_duration_s: float = 1.0
    # chunk 边界过渡 (已移除, 保留仅为兼容 CLI, 恒为 0)
    chunk_transition_duration_s: float = 0.0
    smooth_boundary_max_delta_rad: float = 0.10

    # 保持稳定化 (抑制任务完成/复位后的微振荡)
    smooth_hold_band_rad: float = 0.06
    smooth_hold_raw_band_rad: float = 0.02
    smooth_hold_steps: int = 45           # 连续保持步数阈值 (≈1.6s @29Hz) 才锁定

    # ================================================================
    # 辅助方法
    # ================================================================

    def build_rtc_config(self):
        """构建 PI0.5 RTC 配置对象 (RTCConfig)"""
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs.types import RTCAttentionSchedule

        schedule = self.rtc_prefix_attention_schedule
        if isinstance(schedule, str):
            schedule = RTCAttentionSchedule[schedule.strip().upper()]
        return RTCConfig(
            enabled=self.rtc_enabled,
            execution_horizon=self.rtc_execution_horizon,
            max_guidance_weight=self.rtc_max_guidance_weight,
            prefix_attention_schedule=schedule,
            debug=self.rtc_debug,
        )

    def task_prompt_for(self, model_name: str) -> str:
        """返回指定模型对应的训练任务文本"""
        return TASK_PROMPTS.get(model_name, DEFAULT_TASK_PROMPT)
