"""
ACT + VLM 多任务联调编排器 — 配置模块
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ============================================================================
# 模型路径常量
# ============================================================================
# 三个 ACT 子任务模型的检查点路径
# 本机实际路径 (可通过 start_orchestrator.sh 的 --model-a/b/c 覆盖)
MODEL_BASE_DIR = "/home/ub/Videos/0407/model/ACT_model/model"

DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "Task_A": os.path.join(MODEL_BASE_DIR, "Task_A", "pretrained_model"),
    "Task_B": os.path.join(MODEL_BASE_DIR, "Task_B", "pretrained_model"),
    "Task_C": os.path.join(MODEL_BASE_DIR, "Task_C", "pretrained_model"),
}

# VLM subtask → ACT 模型映射
SUBTASK_TO_MODEL: Dict[str, str] = {
    "subtask1": "Task_A",   # 包裹在桌面 → 抓取到托盘
    "subtask2": "Task_B",   # 包裹在托盘,标签不可见 → 翻转
    "subtask3": "Task_C",   # 包裹在托盘,标签可见 → 放入盒子
    "subtask4": "standby",  # 包裹在盒子 → 待机
}

# 场景分类的中文描述
SUBTASK_DESCRIPTIONS: Dict[str, str] = {
    "subtask1": "包裹在桌面 → Task_A: 抓取到托盘",
    "subtask2": "包裹在托盘,标签不可见 → Task_B: 翻转包裹",
    "subtask3": "包裹在托盘,标签可见 → Task_C: 放入盒子",
    "subtask4": "包裹在盒子 → 待机等待场景复原",
}


@dataclass
class OrchestratorConfig:
    """ACT + VLM 多任务联调编排器配置"""

    # ================================================================
    # VLM 服务配置 (vLLM OpenAI-compatible API)
    # ================================================================
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model_name: str = "qwen3-vl-2b"
    vlm_timeout: float = 8.0
    vlm_max_tokens: int = 128
    vlm_temperature: float = 0.0  # 确定性输出

    # ================================================================
    # ACT 模型配置
    # ================================================================
    model_paths: Dict[str, str] = field(default_factory=lambda: DEFAULT_MODEL_PATHS.copy())
    default_model: str = "Task_A"      # 默认启动模型
    device: str = "cuda"
    fps: float = 30.0                  # ACT 推理频率
    inference_time_s: int = 36000      # 最大运行时间 (10 小时)

    # ACT 时序集成 (temporal ensembling) 系数。
    # 模型训练时 temporal_ensemble_coeff=None (队列模式, 每 chunk_size 步重新预测一次)。
    # 队列模式下相邻 chunk 在边界处首动作可能跳变 → 机械臂"两次chunk之间抖动"。
    # 时序集成把每步新 chunk 与前序 chunk 做加权平均, 抑制 chunk 边界抖动。
    # 但标准时序集成会平均所有维度 (含夹爪), 使夹爪开/合信号模糊 (如 0.9/0.0 → 中间值),
    # 导致抓取不干脆。因此默认启用 act_ensemble_exclude_gripper=True:
    # 夹爪维度透传原始值, 只对关节做时序集成 → 既消除 chunk 抖动又保持夹爪信号清晰。
    # 设为 0 或 None 禁用 (队列模式); 建议保持默认。
    act_temporal_ensemble_coeff: Optional[float] = 0.0
    # 时序集成时是否排除夹爪维度 (夹爪透传原始值, 保持开/合信号清晰)
    act_ensemble_exclude_gripper: bool = True

    # ================================================================
    # 窗口投票机制配置
    # ================================================================
    vote_window_size: int = 5          # 投票窗口大小
    vote_consensus_threshold: int = 3  # 共识阈值 (≥3/5)

    # ================================================================
    # 静止检测配置
    # ================================================================
    # 静止检测配置 (v3: 速度阈值进入 + 保持带维持)
    # 旧版用"单步 delta < 0.01745 rad"判定, 25Hz 下等效速度阈值约 0.44 rad/s,
    # 过于宽松, 低速运动会误判为静止。
    # v3 采用两段式判定:
    #   1) 进入判定: 相邻采样速度 < stationary_velocity_threshold 视为"开始静止"
    #   2) 维持判定: 静止期间位置相对锁定参考的总漂移 < stationary_hold_band
    #      才持续累计静止时间。伺服保持时的微小抖动是来回摆动(总位移≈0),
    #      天然被容忍; 缓慢单调漂移会累计超出保持带 → 判定为运动。
    stationary_velocity_threshold: float = 0.05    # 进入静止的速度阈值 (rad/s ≈ 2.9°/s)
    stationary_hold_band: float = 0.02             # 静止保持带 (rad ≈ 1.15°), 总漂移容忍
    # 单步 delta 安全上限 (rad): 兜底, 防止采样异常/微小抖动误判
    stationary_joint_threshold: float = 0.01745   # 关节变化阈值 (rad), 默认 1°
    stationary_time_threshold: float = 2.0         # 静止持续时间阈值 (秒), 触发 VLM 重评估
    stationary_check_interval: float = 0.05         # 静止检测间隔 (秒)

    # 快速重评估: 一次 VLM 连拍后若共识未变 (未切换模型), 说明当前模型可能
    # 已失效/任务已结束, 机械臂仍静止。此时用更短的静止阈值 (fast_recheck_time)
    # 加速下一轮重评估, 让任务切换更快 (避免"静止3s+连拍1s"的 4s 死循环)。
    # 当模型切换成功或模型恢复有效动作后自动恢复正常阈值。
    stationary_fast_recheck_enabled: bool = True
    stationary_fast_recheck_time: float = 1.0      # 快速重评估的静止阈值 (秒)

    # ================================================================
    # VLM 重评估配置
    # ================================================================
    vlm_recheck_interval_steps: int = 50       # (已废弃, 改用 burst 机制)
    stationary_recheck_cooldown: float = 5.0   # 静止触发重检查后的冷却时间 (秒)

    # VLM 连拍投票配置 (静止确认后, 连续多帧 VLM 判断)
    vlm_burst_count: int = 5            # 静止后连拍帧数 (填充投票窗口)
    vlm_burst_interval: float = 0.1     # 连拍帧间间隔 (秒)
    vlm_burst_total_timeout: float = 5.0  # 连拍整体超时 (秒), 防止 VLM 持续失败导致连拍永不结束

    # 连拍期间运动中止: 若连拍过程中检测到机械臂恢复运动, 立即中止连拍。
    # 修复"机械臂还在运动却触发状态切换"的问题。
    # 注: 子任务超时触发的连拍不受此限制 (此时机械臂可能处于循环运动中)。
    burst_abort_on_motion: bool = True

    # VLM 帧新鲜度阈值 (秒): 超过此时间的缓存帧不用于 VLM 分类,
    # 避免用过期图像判断场景 (应对相机流延迟/卡顿)。
    vlm_frame_max_age: float = 0.5

    # ================================================================
    # 模型切换保护配置
    # ================================================================
    # 切换模型后的一段时间内禁止再次触发 VLM 连拍和模型切换,
    # 给新模型足够的"预热"时间来执行任务, 避免刚切换就被静止检测打断。
    switch_cooldown: float = 3.0            # 切换后保护时间 (秒)

    # 待机时是否保留模型在 GPU (True=恢复更快, 占用更多显存; False=移回 CPU 释放显存)
    # 三个 ACT 模型本已全部预加载在 GPU 上, 待机时再移走一个收益有限,
    # 而恢复时移回 GPU 需要 1~2s 会造成"无动作空窗", 故默认保留。
    standby_keep_on_gpu: bool = True

    # ================================================================
    # 子任务超时配置
    # ================================================================
    # 单个子任务最大执行时间，超时后强制触发 VLM 重评估。
    # 防止子任务卡死/循环但关节仍在运动导致静止检测无法触发。
    # 设为 0 可禁用此功能。
    subtask_timeout: float = 30.0         # 子任务超时时间 (秒)
    subtask_timeout_enabled: bool = True   # 是否启用子任务超时

    # ================================================================
    # 机器人配置
    # ================================================================
    robot_type: str = "single_piper"
    robot_port: str = "can1"
    robot_ctrl_mode: str = "joint"

    # 干跑模式: 连接并读取机械臂 (静止检测依赖关节观测),
    # 但 _send_action 只打印动作不发送给机械臂, 用于安全验证
    # 状态机切换 (VLM 分类/投票/模型切换) 而不产生实际运动。
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
    joint_limit_rad: float = 0.10       # 单步关节跳变安全阈值 (rad)
    consecutive_warnings_limit: int = 15 # 连续警告次数上限

    # ================================================================
    # 日志配置
    # ================================================================
    log_dir: str = "./act_vlm_logs"
    debug_verbose: bool = False          # 详细调试输出

    # ================================================================
    # 夹爪配置 (传递给 ACT 推理)
    # ================================================================
    gripper_clip_min: float = 0.0
    gripper_clip_max: float = 1.0
    gripper_max_open_mm: float = 70.0
    gripper_force_close_threshold: float = 10000.0    # 夹爪闭合力阈值 (单位: N)
    gripper_force_close_value: float = 0.0     # 夹爪闭合力阈值触发时的强制闭合值 (归一化 [0,1])

    # ================================================================
    # 平滑配置 (匹配 S0 人类在环策略 — 逐关节差异化参数)
    #
    # 设计原则:
    #   - 不使用窗口平均 (S0 无此步骤)
    #   - 不使用全局增量限幅 (S0 仅对 J2/J3 做速率限制)
    #   - 不使用关节死区 (S0 无此步骤)
    #   - 夹爪纳入 EMA 平滑 (S0 使用 alpha=0.50)
    #   - 夹爪增设死区 (S0 使用 threshold=0.5 防微颤)
    # ================================================================
    smooth_enabled: bool = True
    # 逐关节 EMA alpha: [J1, J2, J3, J4, J5, J6, gripper]
    # J2(alpha=0.12) / J3(alpha=0.18): 模型预测质量差，超强调平滑
    # gripper(alpha=0.50): 比关节更快响应以保证抓取及时
    smooth_ema_alphas: list = field(default_factory=lambda: [0.30, 0.12, 0.18, 0.30, 0.30, 0.30, 0.50])
    # 逐关节速率限制 (rad/帧): None=不限制
    # 仅 J2/J3 限制 0.015 rad/帧 (~0.86°/帧 @30fps)，抑制跳变和反向运动
    smooth_rate_limits: list = field(default_factory=lambda: [None, 0.015, 0.015, None, None, None, None])
    # 夹爪死区 (归一化 [0,1]): 预测值低于此阈值强制置零 (防微颤)。
    # 原为 0.5: 任何 <0.5 的部分打开信号都会被强制闭合, 导致"放置时夹爪打不开"
    # (模型若输出 0.3~0.5 的微开用于释放包裹, 会被死区完全吞掉)。
    # 降为 0.1: 仅抑制 <0.1 的噪声微颤, 保留 0.1 以上的部分打开信号。
    smooth_gripper_dead_zone: float = 0.3
    # 模型切换过渡平滑时长 (秒): 切换模型后的短时间内增强平滑强度
    # (EMA alpha 与速率限制同步收紧, 随时间线性恢复), 抑制新模型起始 chunk
    # 的关节冲击, 让任务间过渡更流畅。设为 0 可禁用。
    switch_transition_duration_s: float = 1.0
    # chunk 边界过渡平滑时长 (秒): 队列模式下 ACT 每 chunk_size 步重新预测
    # 全新 chunk, 新 chunk 首动作可能与上一 chunk 末动作跳变 (chunk 间抖动)。
    # 检测到新 chunk 开始后, 将前 N 步锚定到上一 chunk 末输出 (短暂停顿后衔接),
    # 消除跳变。N 越大停顿越明显; 0.25s (≈7步) 已足够锚定首步且停顿不明显。
    # 设为 0 可禁用。
    chunk_transition_duration_s: float = 0.18

    # 保持稳定化 (抑制任务完成/复位后的微振荡):
    # ACT 在"保持位姿"时, 每次 chunk 重预测的目标会略有偏移 (实测 ~0.5~2°),
    # 机械臂跟随产生小幅振荡。当模型输出连续 smooth_hold_steps 步都在当前关节的
    # smooth_hold_band_rad 范围内 (即处于"保持"状态) 时, 将关节锁定到参考位置,
    # 抑制振荡; 模型输出超出保持带时自动解除锁定 (不影响真实运动)。
    # 设为 0 可禁用。
    smooth_hold_band_rad: float = 0.03    # 保持判定带 (rad ≈ 1.7°)
    smooth_hold_steps: int = 60           # 连续保持步数阈值 (≈2s @30Hz) 才锁定
