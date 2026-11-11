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
    # 静止保持带 (rad ≈ 2.6°): 静止期间相对参考位置的总漂移容忍。
    # 实测机械臂伺服保持时存在 ±1~1.5° 的物理抖动/慢漂移 (复位后静止检测器
    # 实测总漂移 1.15~2.1°), 1.15° 的旧带会反复超带重置导致永不触发 (2026-08-06)。
    stationary_hold_band: float = 0.045
    # 静止参考重锁间隔 (秒): 静止期间每间隔该时长重锁一次参考位置, 将"总漂移"
    # 退化为"窗口内漂移"。只要速度低于阈值 (且每窗口漂移 < 保持带), 缓慢漂移
    # 也能正确累计静止时间, 避免计时被缓慢漂移反复重置而卡死模型切换。
    stationary_reference_renew_interval: float = 0.5
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
    subtask_timeout: float = 45.0         # 子任务超时时间 (秒)
    subtask_timeout_enabled: bool = True   # 是否启用子任务超时

    # ================================================================
    # 试次统计 / 成功率 / 吞吐率
    # ================================================================
    # 单试次总时间阈值 (秒):
    #   计时起点 = 首次 VLM 场景判断 (_main_loop 中首次 _start_vlm_burst)
    #   计时终点 = C 任务完成机械臂回位 (VLM 共识切换到 subtask4/standby,
    #             包裹已在盒子)。
    #   成功判定 = 阈值内执行完 A/B/C 三个任务。
    trial_time_threshold: float = 90.0
    # 试次结果持久化文件 (JSONL, 每次运行追加一条, 跨程序重启累积统计)。
    # 为空时使用 <log_dir>/trial_stats.jsonl
    trial_stats_file: str = ""

    # 试次完成后是否自动退出进程:
    #   True  (默认) = 90s 内完成 A/B/C → 退出并输出执行时间; 运行满 90s → 退出。
    #   False        = 试次完成后进入待机等待场景复原 (连续循环模式)。
    trial_auto_exit: bool = True

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
    top_camera_id: int = 4
    wrist_camera_id: int = 10
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ================================================================
    # 安全配置
    # ================================================================
    safe_mode: bool = True
    # 单步关节跳变安全阈值 (rad ≈ 8.6°): v7 随 J1/J4/J5/J6 限幅放宽到 0.12
    # (6.9°) 后同步放宽, 避免限幅生效时误触发紧急停止。HIL 流程对 20-40°
    # chunk 跳变以 EMA 直通 (单步可达 6-12°) 运行无碍, 说明机械臂可承受;
    # 0.15 阈值仅拦截真正的异常 (单步 >8.6°)。
    joint_limit_rad: float = 0.15
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
    # 夹爪强制闭合阈值 (映射值): mapped = raw × max_open_mm × 1000。
    # v8 (2026-08-07): 35000 ≈ raw<0.5。日志实测 Task_B/C 抓取时模型输出
    # 0.3~0.5 (映射 21k~35k), 旧阈值 10000 (raw<0.143) 只覆盖噪声带, 抓取带
    # 无法闭合 → 抓不到包裹。模型输出分布: 闭合≈0 / 抓取 0.3~0.5 / 打开
    # 0.55~1.23, 阈值 0.5 使抓取带强制闭合、打开带保持打开 (等效二值化)。
    gripper_force_close_threshold: float = 35000.0
    gripper_force_close_value: float = 0.0     # 夹爪闭合力阈值触发时的强制闭合值 (归一化 [0,1])
    # 夹爪开爪信号非线性放大指数 (0~1): 默认 1.0 = 禁用放大 (线性映射)。
    # 曾尝试 0.5 (√放大) 以增强 Task_A 释放时的微弱开爪信号, 但会把 Task_C
    # 抓取阶段的中间值 (raw 0.2~0.5) 放大为 45~71% 开 → 夹爪"完全不闭合"。
    # 模型开/合命令的中间值无法用单一阈值区分意图, 故恢复线性映射。
    gripper_boost_exponent: float = 1.0

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
    # v9 (2026-08-10): 完全对齐 HIL 采集 (recording_loop.py) 的平滑思路。
    #   - J2/J3 (0.015 rad/帧 ≈ 0.86°/帧): 模型预测质量差, 超强调平滑 (同 HIL);
    #   - J1/J4/J5/J6: 不限制 (同 HIL, EMA 直接透过)。
    # 根因: 编排器与 HIL 用同一模型同一观测时, chunk 边界抖动源于"观测-动作
    # 反馈失同步": 平滑限幅过紧使机械臂在 chunk 边界追赶新 chunk 时滞后,
    # 观测状态落后于轨迹, 模型重新预测时 chunk 起点前跳 → 再追赶 → 正反馈
    # 放大 (实测: 观测同步时边界跳变仅 1.9°, 失同步时高达 49°)。
    # HIL 不对 J1/J4/J5/J6 限幅 (EMA 直接透过), 机械臂紧跟 chunk, 观测保持
    # 同步, 因此无抖动。v7 曾给 J1/J4/J5/J6 设 0.12 rad/帧 硬保护, 但会拖慢
    # chunk 边界追赶 (配合 30% 速度驱动更明显), 现完全对齐 HIL 放开。
    smooth_rate_limits: list = field(default_factory=lambda: [None, 0.015, 0.015, None, None, None, None])
    # 夹爪死区 (归一化 [0,1]): 预测值低于此阈值强制置零 (防微颤)。
    # 0.15: 把模型 0.1~0.15 的过渡/噪声信号当作"未打开"处理, 使夹爪在抓取时
    # 可靠闭合 (2026-08-06 实测 Task_C 夹爪能闭合的关键)。Task_A 释放时模型
    # 的微弱开爪信号 (0.085~0.141) 会被抑制, 但该任务通过手臂动作完成释放
    # (包裹已成功放到托盘), 且放大中间值会导致 Task_C 夹爪无法闭合, 故不放大。
    smooth_gripper_dead_zone: float = 0.15
    # 模型切换过渡平滑时长 (秒): 切换模型后的短时间内增强平滑强度
    # (EMA alpha 与速率限制同步收紧, 随时间线性恢复), 抑制新模型起始 chunk
    # 的关节冲击, 让任务间过渡更流畅。设为 0 可禁用。
    switch_transition_duration_s: float = 1.0
    # chunk 边界过渡: 已移除 (2026-08-06 v5)。原因: HIL 采集 (recording_loop.py)
    # 对策略输出仅做纯 EMA + 速率限制, 无任何 chunk 过渡逻辑, 且无抖动; 编排器
    # 的 chunk 过渡机制 (限幅追赶/锚定) 是 chunk 间抖动的来源。现改为与 HIL 一致
    # 的纯 EMA + 全关节限幅 (见 smooth_rate_limits)。该字段保留仅为兼容 CLI
    # (--chunk-transition-duration), 恒为 0 (禁用)。
    chunk_transition_duration_s: float = 0.0
    # chunk 边界过渡每步限幅 (rad/步): 已随 chunk 过渡机制移除 (2026-08-06 v5),
    # 保留仅为兼容 CLI (--boundary-max-delta), 不再被使用。
    smooth_boundary_max_delta_rad: float = 0.10

    # 保持稳定化 (抑制任务完成/复位后的微振荡):
    # ACT 在"保持位姿"时, 每次 chunk 重预测的目标会略有偏移 (实测 ~0.5~2°),
    # 机械臂跟随产生小幅振荡。当模型输出连续 smooth_hold_steps 步都在当前关节的
    # smooth_hold_band_rad 范围内 (即处于"保持"状态) 时, 将关节锁定到参考位置,
    # 抑制振荡; 模型输出超出保持带时自动解除锁定 (不影响真实运动)。
    # 设为 0 可禁用。
    # 保持判定物理带 (rad ≈ 3.4°): 原始目标与当前关节的最大差异容忍。
    # 实测机械臂伺服保持时物理滞后/抖动可达 ±1.5°, 旧带 1.7° 会使计数被
    # 偶发超带反复清零而永不锁定 (2026-08-06)。
    smooth_hold_band_rad: float = 0.06
    # 保持判定原始带 (rad ≈ 1.15°): 原始目标相对保持窗口参考的位移容忍。
    # 判别"模型真的在保持"还是"模型在运动": 保持时目标稳定 (实测 ±0.3°),
    # 任务运动时目标快速漂移 (每步可达数度)。目标漂移超带 → 判定为运动,
    # 不锁定 (即使物理带内, 防止慢速任务运动被误锁)。
    smooth_hold_raw_band_rad: float = 0.02
    smooth_hold_steps: int = 45           # 连续保持步数阈值 (≈1.6s @29Hz) 才锁定
