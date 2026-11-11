#!/usr/bin/env python
"""
================================================================================
Pi0.5 真机推理测试脚本（流匹配去噪专用） — 多线程 RTC
================================================================================

【用途】
  ★ 本脚本为流匹配去噪（随机采样）专用的真机推理测试脚本，
    用于逐步优化流匹配去噪的推理效果（抖动抑制 / 任务流畅度）。
    采样方式固定为流匹配（随机噪声采样），不含确定性采样逻辑
    （确定性采样见 inference_pi05_adarms_fix.py）。
  在 Evo-RL 环境内进行 Piper 真机部署推理。可加载：
    1) 本机训练的 60K 模型：model/Pimodel/60K/pretrained_model（默认推荐，用 --ckpt_path 指定）
    2) 0407 项目重训的 pi05_adarms_fix 权重（/data/zhanghang/Videos/0407/...，仅限训练机）
  本脚本在 inference_new_joint_rtc.py 基础上，新增了对 checkpoint 格式的兼容加载
  （config 适配 + 权重键重映射 + tokenizer 本地化），其余 RTC/平滑/安全/多线程
  架构与参考脚本完全一致。

【★ 本机硬件配置（重要）】
  - CAN 口：执行臂在 can3（不是 can1！USB-CAN 适配器枚举到 can3）
  - 相机：wrist=/dev/video4，top=/dev/video10（docstring 旧示例曾写反）
  - 训练数据集任务（batch5_joint_merged 合并后 3 个任务）：
      [0] Grab the package and place it on the barcode scanning table.
      [1] Flip the package if the barcode is not facing up.
      [2] Grab the scanned package and place it into the box.

【0407 checkpoint 兼容要点】
  1. config.json 含 0407 特有字段 "type"（Evo-RL PI05Config 无此字段）→ 剔除
  2. input_features/output_features 为普通 dict → 转 PolicyFeature
  3. 权重键带 "model." 前缀 + AdaLN(.dense) 键 → PI05Policy.from_pretrained
     内部 _fix_pytorch_state_dict_keys 已处理（验证: 812 键全部加载成功）
  4. tokenizer 硬编码 HF 名 → factory.py 已注入 override 指向本地路径

【用法示例】
  # ★ 本机 60K 模型 + 流匹配去噪 + 完整优化（RTC + 预设位姿 + 运动对齐 + 抖动抑制）
  python -m lerobot.inference_pi05_flowmatch \
    --robot.type=single_piper \
    --robot.id=single_piper_robot \
    --robot.ctrl_mode="joint" \
    --robot.port=can1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30},top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
    --ckpt_path=/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model \
    --preset_pose=true --diag=true \
    --task="Grab the scanned package and place it into the box."

  # ★ 逐步优化调整（各开关）
  #   抖动抑制 EMA（0=关闭对比，越大越平滑/略增滞后）:
  #     --flow_ema_alpha=0.5
  #   运动速度/模式（与 HIL 采集对齐，解决 chunk 切换抖动）:
  #     --motion_speed=80 --motion_high_follow=true
  #   夹爪闭合增强（0=关闭）:
  #     --gripper_gain=0.40 --gripper_gain_below=0.35 --gripper_gain_above=0.50
  #   RTC 对齐:
  #     --rtc.execution_horizon=16 --rtc.max_guidance_weight=8.0
  #   诊断日志（CSV + 终端）:
  #     --diag=true

  # 指定加载 15000 步的 checkpoint（0407 格式）
  python -m lerobot.inference_pi05_flowmatch \
    ... \
    --ckpt_step=15000 \
    --task="..."

  # 直接指定 pretrained_model 目录（优先级最高，60K 模型用法）
  python -m lerobot.inference_pi05_flowmatch \
    ... \
    --ckpt_path=/home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model \
    --task="..."

  # ★ 按索引从训练数据集任务表选择任务（batch5_joint_merged 有 3 个任务）
  #   0 = Grab the package and place it on the barcode scanning table.
  #   1 = Flip the package if the barcode is not facing up.
  #   2 = Grab the scanned package and place it into the box.
  python -m lerobot.inference_pi05_flowmatch \
    ... \
    --task_idx=1

  # 仅列出数据集可用任务（不启动推理）
  python -m lerobot.inference_pi05_flowmatch --list_tasks=true

  # RTC 参数自定义 / 关闭
  ... --rtc.execution_horizon=16 --rtc.max_guidance_weight=8.0 --rtc.prefix_attention_schedule=EXP
  ... --rtc.enabled=false

【注意事项】
  - 本脚本固定流匹配去噪（随机采样），不含确定性采样逻辑
  - 运行前请确认 ckpt_root 目录存在且包含 005000/010000/015000 等数字子目录
  - 0407 训练仍在进行时，GPU 0/1 被占用；部署请用空闲 GPU 或 --device=cuda:2
================================================================================
"""

import math
import os
import sys
import time
import traceback
import json
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# ── 机器人相关 ──
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
)
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

# ── 工具函数 ──
from lerobot.utils.robot_utils import busy_wait

# ── 策略模型 ──
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs import parser
from lerobot.configs.types import RTCAttentionSchedule, FeatureType, PolicyFeature

# ── RTC 核心模块 ──
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.action_queue import ActionQueue

# ── 机械臂 SDK ──
from piper_sdk import *

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# 第一部分：0407 checkpoint 加载辅助
# =============================================================================

DEFAULT_CKPT_ROOT = "/data/zhanghang/Videos/0407/lerobot/outputs/train/pi05_adarms_fix/checkpoints"

# 训练数据集根目录（用于读取任务表 meta/tasks.parquet）
DEFAULT_DATASET_ROOT = "/data/zhanghang/Videos/datasets/batch5_joint_merged"


def load_dataset_tasks(dataset_root: str) -> list[str]:
    """从 lerobot 数据集的 meta/tasks.parquet 读取任务列表（按 task_index 排序）。

    tasks.parquet 结构：index = 任务文本，列 task_index = 任务编号。
    推理时任务文本必须与训练时完全一致，故直接从数据集读取最可靠。
    """
    tasks_path = os.path.join(dataset_root, "meta", "tasks.parquet")
    if not os.path.isfile(tasks_path):
        return []
    try:
        df = pd.read_parquet(tasks_path)
        if "task_index" in df.columns:
            df = df.sort_values("task_index")
        return [str(t) for t in df.index.tolist()]
    except Exception as e:
        logger.warning(f"读取任务表失败 {tasks_path}: {e}")
        return []


def find_local_datasets() -> list[str]:
    """探测本机可用的 batch5 数据集目录（含 meta/tasks.parquet）。

    本机是推理机，训练机路径 /data/... 不存在；
    这里列出本机实际存在的数据集，供 --dataset_root 使用。
    """
    candidates = [
        "/home/ub/Videos/Evo-RL/datasets/batch5_joint_merged",
        "/home/ub/Videos/Evo-RL/datasets/batch5A_joint_merged",
        "/home/ub/Videos/Evo-RL/datasets/batch5B_joint_merged",
        "/home/ub/Videos/Evo-RL/datasets/batch5C_joint_merged",
    ]
    found = []
    for c in candidates:
        if os.path.isfile(os.path.join(c, "meta", "tasks.parquet")):
            found.append(c)
    return found


def resolve_task(cfg) -> str:
    """解析最终用于推理的任务文本。

    优先级：显式 --task 自由文本 > --task_idx 数据集索引。
    自由文本与数据集任务不一致时给出警告（提示拼写/措辞差异）。
    两者均未指定时，报错并列出数据集可用任务。
    """
    tasks = load_dataset_tasks(cfg.dataset_root)

    # 1. 自由文本优先
    if cfg.task and str(cfg.task).strip():
        task_text = str(cfg.task).strip()
        if tasks and task_text not in tasks:
            print(
                f"[WARNING] ⚠️ 任务文本与数据集任务不完全一致：\n"
                f"    输入: {task_text!r}\n"
                f"    数据集任务: {tasks}\n"
                f"  建议使用 --task_idx 选择数据集中的任务，或确保文本与训练时完全一致。"
            )
        return task_text

    # 2. 按索引从数据集任务表选择
    if cfg.task_idx >= 0:
        if not tasks:
            # ⚠️ 不再静默降级为占位文本（如 "task_2"）——那会让模型收到错误的
            # 任务 token，行为完全不可预期。直接报错并提示本机可用数据集。
            local_ds = find_local_datasets()
            raise ValueError(
                f"task_idx={cfg.task_idx} 已指定，但 dataset_root 下找不到任务表:\n"
                f"    {cfg.dataset_root}\n"
                f"  本机可用数据集: {local_ds if local_ds else '无'}\n"
                f"  请用 --dataset_root 指向本机数据集（如 .../datasets/batch5A_joint_merged）后重试，\n"
                f"  或改用 --task='<训练任务文本>' 直接指定（推荐，无需数据集）。"
            )
        if cfg.task_idx >= len(tasks):
            raise ValueError(
                f"task_idx={cfg.task_idx} 超出范围（0~{len(tasks)-1}）。"
                f"可用任务: {list(enumerate(tasks))}"
            )
        return tasks[cfg.task_idx]

    # 3. 都未指定 → 报错并列出可用任务
    if tasks:
        task_list = "\n".join(f"  [{i}] {t}" for i, t in enumerate(tasks))
        raise ValueError(
            "请指定任务：--task='<文本>' 或 --task_idx=<索引>。\n"
            f"数据集可用任务:\n{task_list}"
        )
    raise ValueError("请指定任务：--task='<文本>' 或 --task_idx=<索引>")


def load_0407_config(ckpt_path: str) -> PI05Config:
    """加载 0407 项目的 config.json 并适配为 Evo-RL 的 PI05Config。

    适配内容：
      1. 剔除 0407 特有的 "type" 字段（Evo-RL PI05Config 无此字段）
      2. input_features / output_features 从普通 dict 转为 PolicyFeature 对象
         （Evo-RL 的 image_features 等属性依赖 ft.type / ft.shape 属性访问）
      3. tokenizer_name 字段（已加入 Evo-RL PI05Config）默认指向本地路径，
         可直接读取 0407 保存的值；若缺失则用默认值
    """
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 config.json: {config_path}")

    with open(config_path) as f:
        cfg_dict = json.load(f)

    # 1. 剔除 0407 特有字段
    cfg_dict.pop("type", None)

    # 2. 特征 dict → PolicyFeature
    for feat_key in ("input_features", "output_features"):
        if feat_key in cfg_dict and isinstance(cfg_dict[feat_key], dict):
            converted = {}
            for name, ft in cfg_dict[feat_key].items():
                if isinstance(ft, PolicyFeature):
                    converted[name] = ft
                else:
                    converted[name] = PolicyFeature(
                        type=FeatureType(ft["type"]),
                        shape=tuple(ft["shape"]),
                    )
            cfg_dict[feat_key] = converted

    # 3. tokenizer_name 强制指向本机离线路径。
    #    训练机保存的 /data/... 在本机不存在，必须覆盖为本地路径，
    #    否则 preprocessor 的 TokenizerProcessorStep 加载 tokenizer 会失败。
    LOCAL_TOKENIZER = "/home/ub/Videos/Evo-RL/model/paligemma-3b-pt-224"
    cfg_dict["tokenizer_name"] = LOCAL_TOKENIZER

    return PI05Config(**cfg_dict)


def resolve_ckpt_path(
    ckpt_root: str = DEFAULT_CKPT_ROOT,
    ckpt_step: int = 0,
    ckpt_path: str | None = None,
) -> str:
    """解析要加载的 0407 checkpoint 路径。

    优先级：显式 ckpt_path > ckpt_step > 自动选最新数字 checkpoint。

    0407 的 checkpoint 目录结构：
        <ckpt_root>/<step 六位数字>/pretrained_model/{config.json, model.safetensors, ...}
    例如：checkpoints/015000/pretrained_model
    """
    if ckpt_path:
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"ckpt_path 不存在: {ckpt_path}")
        return ckpt_path

    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(
            f"ckpt_root 不存在: {ckpt_root}\n"
            f"请确认 0407 的 pi05_adarms_fix 训练已产出 checkpoint，"
            f"或改用 --ckpt_path 直接指定 pretrained_model 目录"
        )

    # 指定步数
    if ckpt_step > 0:
        candidate = os.path.join(ckpt_root, f"{ckpt_step:06d}", "pretrained_model")
        if not os.path.isdir(candidate):
            raise FileNotFoundError(
                f"未找到 step={ckpt_step} 的 checkpoint: {candidate}\n"
                f"可用目录: {[n for n in sorted(os.listdir(ckpt_root)) if os.path.isdir(os.path.join(ckpt_root, n))]}"
            )
        return candidate

    # 自动选最大数字 checkpoint
    steps = []
    for name in os.listdir(ckpt_root):
        full = os.path.join(ckpt_root, name)
        if os.path.isdir(full) and name.isdigit():
            steps.append((int(name), full))
    if not steps:
        raise FileNotFoundError(f"ckpt_root 下没有数字 checkpoint: {ckpt_root}")

    steps.sort()
    best_step, best_dir = steps[-1]
    best_path = os.path.join(best_dir, "pretrained_model")
    print(f"[INFO] 📦 自动选择最新 checkpoint: step={best_step} → {best_path}")
    return best_path


def load_pi05_policy_from_0407(ckpt_path: str) -> PI05Policy:
    """加载 0407 训练的 pi05_adarms_fix checkpoint，返回可推理的 PI05Policy。

    流程：
      1. load_0407_config() 适配 config.json
      2. PI05Policy.from_pretrained(ckpt_path, config=config)
         —— 内部已处理权重键重映射（model. 前缀 / AdaLN .dense / lm_head tied）
    """
    print(f"[INFO] 🧠 适配 0407 config.json ...")
    config = load_0407_config(ckpt_path)
    print(f"[INFO]    ├─ tokenizer_name = {config.tokenizer_name}")
    print(f"[INFO]    ├─ input_features = {list(config.input_features.keys())}")
    print(f"[INFO]    ├─ chunk_size = {config.chunk_size}, max_state_dim = {config.max_state_dim}")
    print(f"[INFO]    └─ pretrained_path = {config.pretrained_path}")

    print(f"[INFO] 🔄 加载模型权重（from_pretrained）...")
    policy = PI05Policy.from_pretrained(ckpt_path, config=config)
    return policy


# =============================================================================
# 第二部分：轨迹平滑器
# =============================================================================

class TrajectorySmoother:
    """
    关节空间轨迹平滑器 — RTC 后处理补充

    【RTC 模式下的策略调整】
    当 RTC 启用时，chunk 间连续性已由 RTC 保证，EMA 滤波可能导致
    过度平滑（响应变慢、动作滞后）。因此 RTC 模式下建议禁用 EMA。
    仍保留窗口平均 + 限幅 + 死区用于消除 chunk 内部的微观抖动。
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,    # EMA 平滑系数（0.0~1.0，越大越快响应）
        pos_max_delta: float = 0.03,    # 关节单步最大变化（rad）
        gripper_max_delta: float = 0.15,   # 夹爪单步最大变化（0~1）
        pos_dead_zone: float = 0.001,    # 关节死区（rad）
        gripper_dead_zone: float = 0.001,   # 夹爪死区（0~1）
        window_size: int = 3,    # 窗口平均大小（步数）
        use_window_average: bool = True,    # 是否使用窗口平均
        use_ema: bool = True,    # 是否使用 EMA 平滑
        use_clipping: bool = True,    # 是否使用限幅
        use_dead_zone: bool = True,    # 是否使用死区
    ):
        self.ema_alpha = ema_alpha
        self.pos_max_delta = pos_max_delta
        self.gripper_max_delta = gripper_max_delta
        self.pos_dead_zone = pos_dead_zone
        self.gripper_dead_zone = gripper_dead_zone
        self.window_size = window_size
        self.use_window_average = use_window_average
        self.use_ema = use_ema
        self.use_clipping = use_clipping
        self.use_dead_zone = use_dead_zone

        self._ema_action = None    # EMA 平滑后的动作
        self._prev_raw_action = None  # 上一帧原始动作（用于限幅和死区计算）
        self._action_history = deque(maxlen=window_size)
        self._initialized = False

    def reset(self):
        self._ema_action = None
        self._prev_raw_action = None
        self._action_history.clear()
        self._initialized = False

    def smooth(
        self,
        action: np.ndarray,
        rtc_enabled: bool = False,
        rtc_multiplier: float = 2.0,
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        if len(action) != 7:
            raise ValueError(f"Expected action of length 7, got {len(action)}")

        if self.use_window_average:
            self._action_history.append(action)
            if len(self._action_history) >= 2:
                action = np.mean(np.stack(list(self._action_history)), axis=0)

        if self.use_ema:
            if self._ema_action is None:
                self._ema_action = action
            else:
                self._ema_action = (
                    self.ema_alpha * action + (1 - self.ema_alpha) * self._ema_action
                )
            action = self._ema_action.copy()

        # ── RTC 模式下动态放宽限幅 ──
        # RTC 通过 guidance 修正模型输出使其与当前轨迹接轨，
        # 如果 smoothing clip 限制太死，实际执行位置会落后于模型预期，
        # 导致下一轮 RTC 修正压力增大 → 震荡。因此 RTC 模式下放宽。
        # 放宽倍数由 --rtc_smoother_multiplier 调整（默认 1.5，原硬编码
        # 2.0 会让大 chunk 以 85°/s 快速执行 → "猛冲"）。
        current_pos_limit = self.pos_max_delta * (rtc_multiplier if rtc_enabled else 1.0)
        current_grip_limit = self.gripper_max_delta * (rtc_multiplier if rtc_enabled else 1.0)

        if self.use_clipping and self._prev_raw_action is not None:
            joint_delta = action[:6] - self._prev_raw_action[:6]
            joint_delta_clipped = np.clip(joint_delta, -current_pos_limit, current_pos_limit)
            action[:6] = self._prev_raw_action[:6] + joint_delta_clipped

            gripper_delta = action[6] - self._prev_raw_action[6]
            action[6] = self._prev_raw_action[6] + np.clip(
                gripper_delta, -current_grip_limit, current_grip_limit
            )

        if self.use_dead_zone and self._prev_raw_action is not None:
            joint_change = np.abs(action[:6] - self._prev_raw_action[:6])
            mask_joint = joint_change < self.pos_dead_zone
            action[:6][mask_joint] = self._prev_raw_action[:6][mask_joint]
            if abs(action[6] - self._prev_raw_action[6]) < self.gripper_dead_zone:
                action[6] = self._prev_raw_action[6]

        self._prev_raw_action = action.copy()
        self._initialized = True
        return action


# =============================================================================
# 第三部分：安全监控器（适配 Piper）
# =============================================================================

class SafetyMonitor:
    """
    关节跳变安全监控器

    【Piper 适配说明】
    Piper 小型机械臂在 20Hz 下的正常运动速度约 30~60°/s，
    对应每步 (50ms) 变化约 1.5~3.0°。旧阈值 0.05 rad ≈ 2.86°/步
    对正常运动过于严格。调整为 0.10 rad ≈ 5.7°/步，允许正常运动
    同时仍能捕获真正的异常跳变。
    """

    def __init__(
        self,
        joint_limit_rad: float = 0.10,       # ≈ 5.7°/步，适配 Piper 低速操作
        gripper_limit: float = 0.5,           # 夹爪单步最大变化
        consecutive_warnings_limit: int = 15, # 连续异常次数（放宽以容忍偶尔波动）
    ):
        self.joint_limit_rad = joint_limit_rad
        self.gripper_limit = gripper_limit
        self.consecutive_warnings_limit = consecutive_warnings_limit
        self._prev_action = None    # 上一帧动作
        self._consecutive_warnings = 0    # 连续异常计数
        self._total_warnings = 0    # 总异常计数

    def check(self, action: np.ndarray) -> tuple:
        warnings_list = []
        is_safe = True

        if self._prev_action is not None:
            joint_delta = np.max(np.abs(action[:6] - self._prev_action[:6]))
            gripper_delta = abs(action[6] - self._prev_action[6])

            if joint_delta > self.joint_limit_rad:
                delta_deg = joint_delta * 180.0 / math.pi
                limit_deg = self.joint_limit_rad * 180.0 / math.pi
                warnings_list.append(f"[关节跳变] {delta_deg:.1f}° > {limit_deg:.1f}°")
            if gripper_delta > self.gripper_limit:
                warnings_list.append(f"[夹爪跳变] {gripper_delta:.3f} > {self.gripper_limit:.3f}")

        self._prev_action = action.copy()

        if warnings_list:
            self._consecutive_warnings += 1
            self._total_warnings += 1
        else:
            self._consecutive_warnings = 0

        if self._consecutive_warnings >= self.consecutive_warnings_limit:
            warnings_list.append(f"[紧急] 连续{self._consecutive_warnings}次警告，急停!")
            is_safe = False

        return is_safe, " | ".join(warnings_list) if warnings_list else ""


# =============================================================================
# 第 3.5 部分：智能延迟追踪器（SmartLatencyTracker）
# =============================================================================

class SmartLatencyTracker:
    """
    EMA 平滑的推理延迟追踪器，替代简单 max() 估计。

    【设计动机】
    LatencyTracker.max() 对系统偶发抖动（Jitter）过于敏感：
    一次 300ms 的异常推理会导致后续所有 RTC 对齐都按 300ms 预留窗口，
    动作反应变迟钝甚至变形。

    【方案】
    - 使用 EMA（指数移动平均）平滑延迟估计，alpha=0.15 给近期值低权重
    - 乘以 margin（默认 1.1，即预留 10% 安全余量）应对温和波动
    - get_estimate() 返回平滑后的安全估计值（秒）
    """

    def __init__(self, alpha: float = 0.15, margin: float = 1.1, initial_estimate: float = 0.200):
        self.alpha = alpha          # EMA 平滑系数（越小越平滑）
        self.margin = margin        # 安全余量倍数
        self._ema_latency = initial_estimate  # EMA 平滑后的延迟（秒）
        self._initialized = True

    def update(self, new_latency: float):
        """用新的延迟测量值更新 EMA 估计。

        快速上升机制：当新延迟明显高于当前 EMA 时，
        临时提高 alpha 到 0.5 以加速跟进，防止估算偏低导致对齐不足。
        """
        effective_alpha = self.alpha if new_latency <= self._ema_latency else 0.5
        self._ema_latency = (
            effective_alpha * new_latency + (1 - effective_alpha) * self._ema_latency
        )

    def get_estimate(self) -> float:
        """返回带安全余量的平滑延迟估计（秒）。"""
        return self._ema_latency * self.margin

    @property
    def raw_ema(self) -> float:
        """返回原始 EMA 值（不含余量），用于调试。"""
        return self._ema_latency


# =============================================================================
# 第 3.6 部分：推理诊断日志（DiagLogger）
# =============================================================================

class DiagLogger:
    """
    推理诊断日志：把推理线程与主循环的关键指标写入 CSV + 终端，便于事后分析。

    【用途】定位三个典型问题：
      1. 模型"无动作/幅度小" → chunk_joint_span(rad) 很小 + 归一化观测状态偏离训练均值
      2. "chunk 衔接跳变"     → 主循环 step_jump_deg 在某个时刻突然 >2°（大概率在 chunk 边界）
      3. "流匹配抖动"         → step_jump_deg 频繁超阈值 + 相邻推理 chunk_joint_span 波动大
    """

    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        self.enabled = enabled
        self.lock = Lock()
        self.infer_path = None
        self.exec_path = None
        if enabled:
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.infer_path = os.path.join(log_dir, f"pi05_diag_infer_{ts}.csv")
            self.exec_path = os.path.join(log_dir, f"pi05_diag_exec_{ts}.csv")
            with open(self.infer_path, "w") as f:
                f.write(
                    "idx,latency_ms,ema_ms,est_steps,real_steps,qsize,threshold,"
                    "joint_span_rad,gripper_span,obs_j0,obs_j1,obs_j2,obs_j3,obs_j4,obs_j5,obs_gripper\n"
                )
            with open(self.exec_path, "w") as f:
                f.write(
                    "step,t_s,j0_deg,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,gripper,step_jump_deg\n"
                )

    def log_infer(
        self,
        idx: int,
        latency_ms: float,
        ema_ms: float,
        est_steps: int,
        real_steps: int,
        qsize: int,
        threshold: int,
        joint_span_rad: float,
        gripper_span: float,
        obs_state,  # 归一化 7 维状态 [j1..j6, gripper]
    ):
        if not self.enabled:
            return
        with self.lock:
            with open(self.infer_path, "a") as f:
                obs = ",".join(f"{v:.4f}" for v in obs_state)
                f.write(
                    f"{idx},{latency_ms:.1f},{ema_ms:.1f},{est_steps},{real_steps},{qsize},{threshold},"
                    f"{joint_span_rad:.5f},{gripper_span:.4f},{obs}\n"
                )

    def log_exec(self, step: int, t_s: float, joint_deg, gripper: int, step_jump_deg: float):
        if not self.enabled:
            return
        with self.lock:
            with open(self.exec_path, "a") as f:
                j = ",".join(f"{v:.2f}" for v in joint_deg)
                f.write(f"{step},{t_s:.3f},{j},{gripper},{step_jump_deg:.3f}\n")


# =============================================================================
# 第四部分：配置类
# =============================================================================

@dataclass
class InferenceRTCConfig:
    """0407 ADARMS 修复版推理的顶层配置"""

    robot: RobotConfig
    policy: PreTrainedConfig | None = None

    # ── 0407 checkpoint 选择 ──
    ckpt_root: str = DEFAULT_CKPT_ROOT    # 0407 pi05_adarms_fix checkpoints 根目录
    ckpt_step: int = 0                    # 指定步数；0 = 自动选最新数字 checkpoint
    ckpt_path: str = None                 # 直接指定 pretrained_model 目录（优先级最高）

    task: str = None                      # 任务文本（自由指定，优先级高于 task_idx）

    # ── 任务选择（训练数据集含 3 个任务）──
    task_idx: int = -1                    # 从数据集任务表按索引选择（0/1/2）；-1=未指定
    dataset_root: str = DEFAULT_DATASET_ROOT   # 用于读取 meta/tasks.parquet 的数据集根目录
    list_tasks: bool = False              # 仅打印数据集可用任务并退出

    # ── RTC 配置（针对 Piper 小型机械臂 / joint control / 低速操作优化）──
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            # ★ 2026-08-25 实测调优: 默认 H=8/G=4 → 边界跳变 8.2x GT;
            #   H=16/G=8 → 1.6x GT (边界跳变 0.0076 rad)。G=8 为甜点, G=16 过矫正。
            execution_horizon=16,                       # 对齐窗口 (默认8→16, 更稳)
            max_guidance_weight=8.0,                    # 引导强度 (默认4→8, 更贴合上一chunk)
            prefix_attention_schedule=RTCAttentionSchedule.EXP,  # 指数衰减：快速过渡
            debug=False,    # 是否打印调试信息
            debug_maxlen=100,    # 最大调试日志长度
        )
    )

    # ── 推理参数 ──
    fps: float = 30.0    # 推理频率（Hz），用于计算每步时间间隔
    device: str = "cuda"    # 推理设备（cuda / cpu）
    inference_time_s: int = 60    # 推理总时长（秒），用于自动退出

    # ── 采样方式（本脚本固定流匹配去噪）──
    # 流匹配去噪（随机噪声采样）：每次采样挑选一个动作模式，能完成任务但
    # 存在采样随机性抖动。确定性采样（零噪声）见 inference_pi05_adarms_fix.py。
    # 本脚本专用于流匹配去噪的逐步优化调整。

    # ── 轨迹平滑开关 ──
    use_trajectory_smoother: bool = True

    # ── 触发新推理的队列阈值 ──
    # 多线程下，必须 > execution_horizon + inference_delay，否则可能断档
    action_queue_size_to_get_new_actions: int = 15

    # ── RAW 模式（★ 模型原始输出测试）──
    # raw_mode=true 时强制关闭 RTC 与轨迹平滑，直接输出模型原始动作 chunk。
    # 用于第一步验证模型在真机上的原始行为；验证通过后再逐步开启
    # 平滑（--use_trajectory_smoother=true）与 RTC（--rtc.enabled=true）。
    raw_mode: bool = False

    # ── 诊断日志（★ 排查无动作/拼接跳变/抖动）──
    # 记录推理线程与主循环关键指标到 logs/pi05_diag_*.csv，并打印关键诊断。
    # 关闭可减少终端输出与 IO 开销。
    diag: bool = False

    # ── [JUMP]/[GRIP] 相邻跳变报警阈值 ──
    # ★ 关节与夹爪分开报警：夹爪是 0~1 比例，混进"度"的关节指标会把
    #   [JUMP] 刷屏（夹爪快开快合 0.3 比例会被显示成 17.2°）。
    # 关节阈值参考：Piper 正常高速运动每步 2~4°（46~80°/s），
    # <4° 观感不明显，默认 5° 只报警真正的异常跳变（chunk 边界大跳变）。
    jump_alert_deg: float = 5.0          # 关节：单步最大变化（度）
    gripper_jump_alert: float = 0.15     # 夹爪：单步最大变化（0~1 比例）

    # ── 安全监控阈值（最后防线，与平滑限幅对齐）──
    # RTC 模式下平滑器关节限幅 = 0.03×2 = 0.06 rad/步（3.4°/步），
    # 安全阈值设 0.06 rad：正常运动（≤3.4°/步）不会误触，但能捕获
    # 平滑失效 / RAW 模式下的真实异常跳变。夹爪限幅 0.35（略高于
    # 平滑器 RTC 限幅 0.30），只拦截真正的异常。
    safety_joint_limit_rad: float = 0.06          # ≈ 3.4°/步（对齐平滑器 RTC 限幅）
    safety_gripper_limit: float = 0.35            # 夹爪单步最大变化（0~1）
    safety_consecutive_warnings: int = 15         # 连续异常次数 → 急停

    # ── 预设位姿（★ 解决"初始位姿偏离训练分布导致无动作"）──
    # 推理前先把机械臂缓慢移到**对应任务的训练初始位姿**（见 resolve_preset_joints）：
    #   任务A(放到扫码台) → home/原点（训练首帧 j1≈0,j2≈0）
    #   任务B(翻面)/C(放入纸箱) → 工作区
    # 让初始状态落在对应任务的训练初始分布内，模型才会正常执行任务动作。
    # --preset_pose=false 可关闭（保持机械臂当前位置开始）。
    preset_pose: bool = True
    # 兜底位姿（弧度）: [j1..j6]（按任务匹配失败时使用）
    preset_joints: tuple = (0.307, 1.304, -1.277, -0.048, 0.953, 0.116)
    # 预设时夹爪打开比例
    preset_gripper: float = 0.7
    # 预设移动步数与每步间隔（秒），决定移动速度（默认 100 步 × 0.02s ≈ 2s）
    preset_steps: int = 100
    preset_step_dt: float = 0.02

    # ── 夹爪闭合增强（★ 解决"抓取时夹爪闭合不紧"）──
    # 模型输出 gripper 在闭合阶段可能停在 0.05~0.25（物理上夹爪有间隙）。
    # 用分段线性压缩：比例 < gripper_gain_below 时乘 gripper_gain（更紧），
    # 在 [below, above] 区间线性过渡回正常，≥ above 不压缩。
    # 映射连续无阶跃，保留夹爪状态平滑变化（区别于瞬时置 0，不破坏模型闭环）。
    # gripper_gain=0 可禁用。
    gripper_gain_below: float = 0.35   # 低于该比例启用压缩
    gripper_gain: float = 0.40         # 压缩增益（实际发送=比例×gain）
    gripper_gain_above: float = 0.50   # 高于该比例恢复正常（中间线性过渡）

    # ── 运动参数（★ 与 HIL 采集对齐，解决 chunk 切换抖动）──
    # 采集时从臂 piper_follower: speed_ratio=100, high_follow=True(mit_mode=0xAD)。
    # 原推理用 速度30 + mit_mode=0x00 → 机械臂响应慢/模式不同 → 位置滞后 →
    # RTC 的 prev_chunk_left_over 与实际位置错位 → chunk 切换抖动。
    motion_speed: int = 100            # 运动速度百分比（与采集一致）
    motion_high_follow: bool = True    # MIT/高跟随模式（与采集一致，mit_mode=0xAD）

    # ── 流匹配去噪抖动抑制（本脚本始终生效）──
    # 流匹配每次采样噪声随机，chunk 间输出不连续；在主循环对发送动作做
    # 轻量 EMA（alpha 较高、滞后小），抑制高频抖动。
    # 0 禁用（逐步优化时可用于对比）。
    # ★ 2026-08-27 调优：α 0.4→0.2，且仅作用于关节（夹爪交给二值滞回
    #   控制器，避免 EMA 拖慢夹爪闭合）。EMA 会给 RTC 对齐引入少量滞后，
    #   不宜过大。
    flow_ema_alpha: float = 0

    # ── HIL 风格逐关节 EMA + 限速（★ 2026-08-28 同步 HIL 采集优化）──
    # 真机对比发现 HIL 采集(recording_loop.py)的 PI05 推理效果比本脚本好，
    # 核心差异在关节平滑: HIL 对模型学得差的 J2/J3 用更低 EMA α(0.12/0.18)
    # + 显式限速(0.015 rad/帧≈0.86°/帧) 强力抑制突跳/反转，其余关节 α=0.30。
    # 本脚本原为"窗口平均+硬限幅+单α流匹配EMA"，J2/J3 未特别照顾。
    # hil_smooth_enabled=true 时，把 9e.0 的单α流匹配 EMA 替换为逐关节 EMA
    # + J2/J3 限速（保留 9c 平滑器的硬限幅作为底层安全）。
    hil_smooth_enabled: bool = True
    hil_ema_alphas: tuple = (0.30, 0.12, 0.18, 0.30, 0.30, 0.30)  # J1..J6 (对齐 HIL)
    hil_rate_limits: tuple = (0.0, 0.015, 0.015, 0.0, 0.0, 0.0)   # J1..J6 rad/帧 (0=不限)

    # ── RTC 模式下平滑限幅放宽倍数（★ 抑制"快速运动"观感）──
    # 平滑器关节限幅 = pos_max_delta(0.03) × 倍数。原硬编码 2.0 →
    # 3.4°/步 ≈ 85°/s，超过 Piper 正常速度(30~60°/s)，大 chunk 被以
    # 最大速度执行导致"猛冲"。改为可配置，默认 1.5 → 2.6°/步 ≈ 64°/s。
    rtc_smoother_multiplier: float = 1.5

    # ── 夹爪二值滞回控制（★ 解决"夹不到"/"闭合缓慢"）──
    # 模型输出的夹爪命令常在 chunk 内从全开写到全闭再写回（夹爪极差>0.8），
    # 导致夹爪快速扇动、抓取时刻无法稳定闭合。用滞回 + 最小驻留时间把
    # 意图转化为稳定的"开/闭"状态：>open_above 判定开，<close_below 判定闭，
    # 中间保持上一状态；切换后至少驻留 gripper_min_dwell 步。
    # 闭合时直接发送 0（完全闭合，跳过平滑/EMA 造成的缓慢闭合）。
    # gripper_binary=false 可回退到原来的连续模式 + 低位压缩。
    gripper_binary: bool = True
    gripper_open_above: float = 0.55     # 夹爪意图 > 此值 → 开
    gripper_close_below: float = 0.30    # 夹爪意图 < 此值 → 闭
    gripper_min_dwell: int = 40          # 状态切换后最小驻留步数（防扇动）

    def __post_init__(self):
        # RAW 模式：一键关闭 RTC + 平滑，输出模型原始动作
        if self.raw_mode:
            self.rtc.enabled = False
            self.use_trajectory_smoother = False
        # 兼容参考脚本：支持 --policy.path=<path> / --policy.type=pi05
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# =============================================================================
# 第五部分：机械臂使能
# =============================================================================

def enable_fun(piper: C_PiperInterface):
    """使能机械臂并检测使能状态，超时 5 秒"""
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False

    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print(f"使能状态: {enable_flag}")
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)

    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)


def motion_ctrl(piper, speed: int = 100, high_follow: bool = True):
    """发送运动控制模式（默认与 HIL 采集一致：速度100 + MIT/high_follow 0xAD）。"""
    mit_mode = 0xAD if high_follow else 0x00
    piper.MotionCtrl_2(0x01, 0x01, speed, mit_mode)


def resolve_preset_joints(task_text: str, default_joints) -> tuple:
    """按任务文本选择对应的训练初始位姿（弧度）。

    数据来源：batch5A/B/C 数据集各任务首帧 observation.state 的均值
    （2026-08-27 重新统计，对齐训练初始分布）：
      - 任务A(放到扫码台 "barcode scanning table"): home/原点附近 j1≈0,j2≈0
      - 任务B(翻面 "Flip"): 工作区，训练初始均值 j1≈1.188,j2≈-1.411
      - 任务C(放入纸箱 "box"): 工作区，训练初始均值 j1≈1.355,j2≈-1.452
    """
    t = task_text or ""
    if "barcode scanning table" in t:
        return (-0.012, 0.004, -0.004, 0.045, 0.443, 0.022)   # 任务A: home/原点
    if "Flip" in t:
        return (0.341, 1.188, -1.411, -0.003, 1.166, -0.002)   # 任务B: 训练初始均值
    if "box" in t:
        return (0.174, 1.355, -1.452, 0.098, 1.188, -0.052)   # 任务C: 训练初始均值
    return tuple(default_joints)                               # 兜底


def move_to_preset_pose(
    robot,
    target_joints_rad,
    gripper_open: float = 1.0,
    steps: int = 100,
    step_dt: float = 0.02,
    motion_speed: int = 100,
    motion_high_follow: bool = True,
):
    """缓慢将机械臂移动到目标位姿（训练均值/工作区），避免跳变。

    - 读取当前关节（Piper 毫度 → 弧度）
    - 线性插值分 steps 步发送，每步 step_dt 秒 → 平滑移动
    - 到位后打开夹爪
    """
    RAD_TO_MILLIDEG = 1000.0 * 180.0 / math.pi
    # 读取当前关节：必须用 GetArmJointMsgs()（GetArmLowSpdInfoMsgs() 返回电机
    # 驱动信息，没有 joint_state，会报 AttributeError）
    joint_state = robot.piper.GetArmJointMsgs().joint_state
    current = np.array(
        [joint_state.joint_1, joint_state.joint_2, joint_state.joint_3,
         joint_state.joint_4, joint_state.joint_5, joint_state.joint_6],
        dtype=np.float64,
    ) * (0.001 * math.pi / 180.0)   # 毫度 → 弧度
    target = np.asarray(target_joints_rad, dtype=np.float64)

    if np.max(np.abs(target - current)) < 0.02:
        print(f"[INFO] 🎯 已在目标位姿附近，跳过预设移动")
    else:
        print(f"[INFO] 🎯 缓慢移动到预设位姿（{steps} 步 × {step_dt*1000:.0f}ms）...")
        for i in range(1, steps + 1):
            alpha = i / steps
            interp = current + (target - current) * alpha
            joints = [int(j * RAD_TO_MILLIDEG) for j in interp]
            motion_ctrl(robot.piper, motion_speed, motion_high_follow)
            robot.piper.JointCtrl(
                joints[0], joints[1], joints[2], joints[3], joints[4], joints[5]
            )
            time.sleep(step_dt)
        print(f"[INFO] ✅ 已到位: {np.round(target, 3)} rad")

    robot.piper.GripperCtrl(round(gripper_open * 70 * 1000), 2000, 0x01, 0)
    print(f"[INFO] 🦾 夹爪打开: {gripper_open:.1f}")


# =============================================================================
# 第六部分：观测预处理（供后台线程使用）
# =============================================================================

def prepare_observation(
    robot,
    device: str,
    task: str,
    preprocess_fn,
) -> dict:
    """
    从机器人获取观测并转换为模型输入格式。

    添加 obs_timestamp 用于相机/状态同步分析。
    """
    obs_timestamp = time.perf_counter()

    observation = robot.get_observation()
    observation_frame = {}
    state_values = []

    for key, value in observation.items():
        if key.endswith('.pos'):
            state_values.append(np.float32(value))
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            observation_frame[f'observation.images.{key}'] = value

    observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)

    # ── 图像预处理 ──
    for name in observation_frame:
        if "image" in name:
            obs = observation_frame[name].astype(np.float32) / 255.0
            obs = np.transpose(obs, (2, 0, 1))  # HWC → CHW
        else:
            obs = observation_frame[name]
        obs = np.expand_dims(obs, axis=0)
        observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=device)

    observation_frame["task"] = [task]
    # 记录时间戳（调试用）
    observation_frame["_obs_timestamp"] = obs_timestamp

    return preprocess_fn(observation_frame)


# =============================================================================
# 第七部分：后台推理线程
# =============================================================================

def get_actions_thread(
    policy,
    robot,
    robot_lock: Lock,
    action_queue: ActionQueue,
    shutdown_event: Event,
    cfg: InferenceRTCConfig,
    preprocess_fn,
    postprocess_fn,
    device: str,
    diag=None,  # DiagLogger 实例，用于推理诊断记录
):
    """
    后台推理线程 — 使用 SmartLatencyTracker + 动态阈值

    职责：
    - 监控 action_queue 剩余动作数
    - 动态计算触发阈值：预估延迟步数 + execution_horizon 的一半
    - 在队列即将耗尽时获取最新观测并触发一次模型推理
    - 将生成的 action chunk 合并到队列中
    - 推理在后台进行，主线程持续执行动作（真正的异步 RTC）
    """
    try:
        logger.info("[GET_ACTIONS] 后台推理线程启动 (SmartLatencyTracker)")

        # ── 使用 EMA 平滑的延迟追踪器替代 LatencyTracker ──
        latency_tracker = SmartLatencyTracker(
            alpha=0.15,             # EMA 平滑系数
            margin=1.1,             # 10% 安全余量
            initial_estimate=0.200, # pi05 首次推理保守估计 200ms
        )

        infer_count = 0    # 诊断：推理序号

        fps = cfg.fps
        time_per_step = 1.0 / fps
        base_threshold = cfg.action_queue_size_to_get_new_actions

        if not cfg.rtc.enabled:
            base_threshold = 0

        last_steps_passed = None    # 上次实际消耗步数（用于 inference_delay 兜底）

        while not shutdown_event.is_set():
            # ── 动态阈值计算 ──
            # 阈值 = max(预估延迟步数 + 执行窗口的一半, 基础阈值)
            # 这样既能确保 RTC 有足够的 leftover 用于对齐，
            # 又不会过早触发推理（浪费计算资源）
            estimated_delay_s = latency_tracker.get_estimate()
            inference_delay_steps = math.ceil(estimated_delay_s / time_per_step)
            # ★ 用上次实际 steps_passed 兜底：EMA 可能低估（首次/抖动），
            #   而拼接点必须落在 RTC 硬对齐区（权重=1），若 est < real 则
            #   对齐点偏早 → chunk 边界跳变。max(EMA, 上次实际+1) 保证不低估。
            if last_steps_passed is not None:
                inference_delay_steps = max(inference_delay_steps, last_steps_passed + 1)
            dynamic_threshold = inference_delay_steps + (cfg.rtc.execution_horizon // 2)
            threshold = max(dynamic_threshold, base_threshold)

            if action_queue.qsize() <= threshold:
                # ── 步骤1：捕获当前同步点 ──
                t_start = time.perf_counter()
                idx_at_start = action_queue.get_action_index()
                prev_chunk_left_over = action_queue.get_left_over()

                # ── 步骤2：获取观测 + 预处理（不加锁）──
                # 摄像头读取（USB）与控制指令发送（CAN）走不同硬件通路，
                # 无需互斥锁。去掉锁避免摄像头 IO 阻塞主线程控制指令。
                processed_obs = prepare_observation(
                    robot, device, cfg.task, preprocess_fn
                )
                obs_ts = processed_obs.pop("_obs_timestamp", None)

                # ── 步骤3：模型推理（RTC 引导）──
                # inference_delay_steps 基于 EMA 延迟估计，
                # 告诉 RTC 去噪循环前 N 步需要硬对齐
                # ★ 必须用 torch.no_grad() 而非 inference_mode：RTC 的 guidance 需要
                #   torch.enable_grad() 内的 autograd.grad，而 inference_mode 无法被覆盖！
                # 流匹配去噪：不传 noise → 模型内部随机采样（多模态）
                with torch.no_grad():
                    actions_chunk = policy.predict_action_chunk(
                        processed_obs,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay_steps,
                        execution_horizon=cfg.rtc.execution_horizon,
                    )

                # ── 步骤4：后处理 ──
                original_actions = actions_chunk.squeeze(0).clone()
                postprocessed_chunk = postprocess_fn(actions_chunk)
                postprocessed_chunk = postprocessed_chunk.squeeze(0)

                # ── 步骤5：更新延迟统计（EMA 平滑）──
                t_end = time.perf_counter()
                actual_latency = t_end - t_start
                latency_tracker.update(actual_latency)

                # ── 步骤5.5：诊断指标（模型输出是否"有动作"）──
                # joint_span: chunk 内 6 关节极差最大值(rad)，判断模型输出幅度
                #   （<0.05rad≈保持不动；>0.3rad≈有明显动作）
                # gripper_span: chunk 内夹爪极差（0~1）
                # obs_state: 归一化观测状态（约 0 = 训练均值附近，偏离大 = 分布外）
                post_chunk_np = postprocessed_chunk.cpu().numpy()
                joint_span = float(
                    np.abs(post_chunk_np[:, :6].max(0) - post_chunk_np[:, :6].min(0)).max()
                )
                gripper_span = float(np.abs(post_chunk_np[:, 6].max() - post_chunk_np[:, 6].min()))
                obs_state_np = processed_obs["observation.state"].cpu().numpy().reshape(-1)[:7]

                # ── 步骤6：计算实际消耗步数并合并 ──
                # 从"获取观测"到"准备好动作"期间，主线程实际跑了多少步
                idx_at_end = action_queue.get_action_index()
                steps_passed = idx_at_end - idx_at_start
                last_steps_passed = steps_passed

                action_queue.merge(
                    original_actions,
                    postprocessed_chunk,
                    steps_passed,
                    idx_at_start,
                )

                # ── 步骤7：诊断记录（CSV + 终端）──
                infer_count += 1
                if diag is not None:
                    diag.log_infer(
                        infer_count, actual_latency * 1000, latency_tracker.raw_ema * 1000,
                        inference_delay_steps, steps_passed, action_queue.qsize(), threshold,
                        joint_span, gripper_span, obs_state_np,
                    )
                print(
                    f"[DIAG] 推理#{infer_count:3d} | 耗时:{actual_latency*1000:4.0f}ms | "
                    f"EMA:{latency_tracker.raw_ema*1000:4.0f}ms | "
                    f"预估{inference_delay_steps}实{steps_passed}步 | "
                    f"队列:{action_queue.qsize():2d} | "
                    f"chunk关节幅度:{joint_span*180.0/math.pi:5.1f}° | "
                    f"夹爪极差:{gripper_span:.3f} | "
                    f"观测j0:{obs_state_np[0]:+.2f} j5:{obs_state_np[5]:+.2f} g:{obs_state_np[6]:+.2f}"
                )

                logger.debug(
                    f"[GET_ACTIONS] 推理完成 | "
                    f"耗时:{actual_latency*1000:.0f}ms | "
                    f"EMA延迟:{latency_tracker.raw_ema*1000:.0f}ms | "
                    f"预估步数:{inference_delay_steps} | "
                    f"实际步数:{steps_passed} | "
                    f"队列剩余:{action_queue.qsize()} | "
                    f"阈值:{threshold}"
                )
            else:
                # 队列充足，短暂休眠避免 busy-wait（降低 CPU 占用）
                time.sleep(0.005)

        logger.info("[GET_ACTIONS] 后台推理线程正常退出")

    except Exception:
        logger.error(f"[GET_ACTIONS] 后台推理线程致命异常:\n{traceback.format_exc()}")
        shutdown_event.set()
        sys.exit(1)


# =============================================================================
# 第八部分：主推理函数
# =============================================================================

@parser.wrap()
def inference(cfg: InferenceRTCConfig) -> None:
    """
    RTC 增强版推理主函数（多线程架构）— 加载 0407 pi05_adarms_fix 权重

    主线程职责：
    - 创建并管理后台推理线程
    - 以固定频率从 ActionQueue 取动作 → 平滑 → 安全检查 → 发送
    - 处理 Ctrl+C 优雅退出
    """

    # =====================================================================
    # 步骤0：解析任务文本（数据集三任务支持）—— 先于硬件连接，快速失败
    # =====================================================================
    _avail_tasks = load_dataset_tasks(cfg.dataset_root)
    if _avail_tasks:
        print("[INFO] 📋 数据集可用任务:")
        for _i, _t in enumerate(_avail_tasks):
            print(f"      [{_i}] {_t}")
    if cfg.list_tasks:
        print("[INFO] 仅列出任务，退出。")
        return
    cfg.task = resolve_task(cfg)
    print(f"[INFO] 📝 最终任务文本: {cfg.task!r}")

    # =====================================================================
    # 步骤0.5：日志配置（终端 + 文件）+ 诊断日志
    # =====================================================================
    if cfg.diag:
        os.makedirs("logs", exist_ok=True)
        _ts = time.strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(os.path.join("logs", f"pi05_run_{_ts}.log")),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    diag = DiagLogger(log_dir="logs", enabled=cfg.diag)
    if cfg.diag:
        print(f"[INFO] 📈 诊断日志已开启 → logs/pi05_diag_infer_*.csv / pi05_diag_exec_*.csv")

    # =====================================================================
    # 步骤1：连接机器人
    # =====================================================================
    robot = make_robot_from_config(cfg.robot)
    if not robot.is_connected:
        robot.connect()

    # =====================================================================
    # 步骤2：推理参数
    # =====================================================================
    fps = cfg.fps    # 推理频率（Hz），用于计算每步时间间隔
    device = cfg.device    # 推理设备（cuda / cpu）
    RAD_TO_MILLIDEG = 1000.0 * 180.0 / math.pi    # 关节弧度转毫角度（rad → mdeg）

    # =====================================================================
    # 步骤3：解析并加载模型（0407 checkpoint 兼容）
    # =====================================================================
    ckpt_path = resolve_ckpt_path(cfg.ckpt_root, cfg.ckpt_step, cfg.ckpt_path)
    policy_type = getattr(cfg.policy, "type", "pi05")

    print(f"{'='*60}")
    print(f"[INFO] 🎯 0407 ADARMS 修复版推理（多线程 RTC）")
    print(f"[INFO] 📂 checkpoint: {ckpt_path}")
    print(f"[INFO] 🧠 策略类型: {policy_type}")
    print(f"[INFO] 📝 任务描述: {cfg.task}")
    print(f"[INFO] ⚡ 频率: {fps} Hz")
    print(f"[INFO] 🧪 模式: {'RAW（原始输出，无RTC无平滑）' if cfg.raw_mode else '优化（RTC+平滑）'}")
    print(f"[INFO] 🔄 RTC: {'启用' if cfg.rtc.enabled else '禁用'}")
    if cfg.rtc.enabled:
        print(f"[INFO]    ├─ 调度策略: {cfg.rtc.prefix_attention_schedule}")
        print(f"[INFO]    ├─ 对齐窗口: {cfg.rtc.execution_horizon} 步")
        print(f"[INFO]    └─ 引导强度: {cfg.rtc.max_guidance_weight}")
        print(f"[INFO] 🔧 轨迹平滑: 启用（窗口+限幅+死区，EMA已自动关闭）")
    else:
        print(f"[INFO] 🔧 轨迹平滑: {'启用（含EMA）' if cfg.use_trajectory_smoother else '禁用'}")
    print(f"[INFO] 🎲 采样模式: 随机(流匹配去噪)")
    print(f"[INFO] 🛡️ 安全监控: 关节阈值 {cfg.safety_joint_limit_rad}rad"
          f"(~{cfg.safety_joint_limit_rad*180.0/math.pi:.1f}°/步), "
          f"连续{cfg.safety_consecutive_warnings}次急停")
    print(f"[INFO] ⏰ 最长运行时长: {cfg.inference_time_s}s"
          + ("（到达后自动退出）" if cfg.inference_time_s > 0 else "（不限时）"))
    print(f"[INFO] 🦾 夹爪: "
          + ("二值滞回(开>%.2f/闭<%.2f, 驻留%d步)" % (
              cfg.gripper_open_above, cfg.gripper_close_below, cfg.gripper_min_dwell)
             if cfg.gripper_binary else "连续+低位压缩"))
    print(f"[INFO] 🏃 运动参数: 速度{cfg.motion_speed}% | "
          f"MIT/高跟随={'开(0xAD)' if cfg.motion_high_follow else '关(0x00)'} | "
          f"流匹配EMA: {'开(α=' + str(cfg.flow_ema_alpha) + ', 仅关节)' if cfg.flow_ema_alpha > 0 else '关'} | "
          f"RTC限幅×{cfg.rtc_smoother_multiplier}")
    print(f"{'='*60}")

    if cfg.robot.ctrl_mode != "joint":
        raise ValueError(f"此脚本要求 ctrl_mode='joint'，当前为 '{cfg.robot.ctrl_mode}'")
    if policy_type != "pi05":
        raise ValueError(f"当前仅支持 pi05 策略，不支持 {policy_type}")

    # ── 加载 0407 训练的 PI05 策略 ──
    policy = load_pi05_policy_from_0407(ckpt_path)
    print(f"[INFO] ✅ 模型加载完成")

    # ── 注入 RTC 配置 ──
    policy.config.rtc_config = cfg.rtc
    policy.init_rtc_processor()
    rtc_status = "已启用" if policy._rtc_enabled() else "未启用"
    print(f"[INFO] ✅ RTC 配置已注入，当前状态: {rtc_status}")

    # ── 加载预处理器/后处理器 ──
    # ⚠️ 必须 override tokenizer_processor 的 tokenizer_name：
    #    checkpoint 的 policy_preprocessor.json 里保存的是训练机路径
    #    （/data/zhanghang/...，本机不存在），需强制指向本机离线 tokenizer，
    #    否则 PolicyProcessorPipeline.from_pretrained 加载时会失败。
    print(f"[INFO] 📊 加载预处理器/后处理器...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        ckpt_path,
        preprocessor_overrides={
            "tokenizer_processor": {
                "tokenizer_name": policy.config.tokenizer_name,
            }
        },
    )
    print(f"[INFO] ✅ 预处理器/后处理器加载完成 (tokenizer: {policy.config.tokenizer_name})")

    policy.eval()
    policy.to(device)
    policy.reset()

    # =====================================================================
    # 步骤4：初始化共享组件
    # =====================================================================
    robot_lock = Lock()                     # 线程安全的机器人访问锁
    action_queue = ActionQueue(cfg.rtc)     # 线程安全的动作队列
    shutdown_event = Event()                # 全局退出信号

    # =====================================================================
    # 步骤5：初始化轨迹平滑器（RTC 模式下自动禁用 EMA）
    # =====================================================================
    if cfg.use_trajectory_smoother:
        _use_ema = not cfg.rtc.enabled      # RTC 开启时自动关闭 EMA
        smoother = TrajectorySmoother(
            ema_alpha=0.30,
            pos_max_delta=0.03,
            gripper_max_delta=0.15,
            pos_dead_zone=0.001,
            window_size=3,
            use_window_average=True,
            use_ema=_use_ema,
            use_clipping=True,
            use_dead_zone=True,
        )
        _ema_status = "关闭（RTC已启用）" if not _use_ema else "开启"
        print(f"[INFO] 🔧 轨迹平滑器已启用（EMA: {_ema_status}）")
    else:
        smoother = None
        print(f"[INFO] 🔧 轨迹平滑器已禁用")

    # =====================================================================
    # 步骤6：初始化安全监控器（适配 Piper）
    # =====================================================================
    safety_monitor = SafetyMonitor(
        joint_limit_rad=cfg.safety_joint_limit_rad,        # 与平滑器 RTC 限幅对齐
        gripper_limit=cfg.safety_gripper_limit,            # 夹爪单步最大变化（0~1）
        consecutive_warnings_limit=cfg.safety_consecutive_warnings,
    )
    print(f"[INFO] 🛡️ 安全监控器已初始化")

    # =====================================================================
    # 步骤7：使能机械臂
    # =====================================================================
    if cfg.robot.type == "single_piper":
        robot.piper.EnableArm(7)
        enable_fun(piper=robot.piper)
        # 运动模式/速度与 HIL 采集对齐（速度100 + MIT 0xAD）
        motion_ctrl(robot.piper, cfg.motion_speed, cfg.motion_high_follow)
        if cfg.preset_pose:
            # 按任务选择初始位姿（任务A=home原点，任务B/C=工作区）
            preset_joints = resolve_preset_joints(cfg.task, cfg.preset_joints)
            print(f"[INFO] 🎯 任务初始位姿: {preset_joints}")
            move_to_preset_pose(
                robot,
                preset_joints,
                gripper_open=cfg.preset_gripper,
                steps=cfg.preset_steps,
                step_dt=cfg.preset_step_dt,
                motion_speed=cfg.motion_speed,
                motion_high_follow=cfg.motion_high_follow,
            )
        else:
            robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
    else:
        raise ValueError("仅支持 single_piper 机器人类型")

    time.sleep(2.0)
    np.set_printoptions(suppress=True, precision=6)

    # =====================================================================
    # 步骤7.5：CUDA 模型预热（Warmup）
    # =====================================================================
    # Pi0.5 首次推理需要 1-2 秒进行 CUDA 算子编译和显存分配。
    # 如果不预热，第一次实际推理的 steps_passed 会远超
    # inference_delay_steps，导致开局动作剧烈跳变。
    # 预热 3 次确保所有 lazy 初始化完成。
    print(f"[INFO] 🔥 正在预热模型 (CUDA Warmup)...")
    _warmup_obs = prepare_observation(robot, device, cfg.task, preprocess)
    # ★ 必须用 no_grad() 而非 inference_mode（原因同上：RTC guidance 需要 autograd）
    with torch.no_grad():
        for _warmup_i in range(3):
            _ = policy.predict_action_chunk(
                _warmup_obs,
                inference_delay=0,
                execution_horizon=cfg.rtc.execution_horizon,
            )
    print(f"[INFO] ✅ 模型预热完成")

    # =====================================================================
    # 步骤8：启动后台推理线程
    # =====================================================================
    infer_thread = Thread(
        target=get_actions_thread,
        args=(
            policy, robot, robot_lock, action_queue,
            shutdown_event, cfg, preprocess, postprocess, device, diag,
        ),
        daemon=True,
        name="RTC-Inference",
    )
    infer_thread.start()
    print(f"[INFO] 🧵 后台推理线程已启动")

    # =====================================================================
    # 步骤9：主执行循环（actor）
    # =====================================================================
    loop_count = 0
    total_loop_time = 0.0
    stats_interval = 100

    # 关节限位
    joint_limits = [(-3, 3)] * 6
    joint_limits[0] = (-2.687, 2.687)
    joint_limits[1] = (0.0, 3.403)
    joint_limits[2] = (-3.0541012, 0.0)
    joint_limits[3] = (-1.5499, 1.5499)
    joint_limits[4] = (-1.22, 1.22)
    joint_limits[5] = (-1.7452, 1.7452)

    def clamp(value, min_val, max_val):
        return max(min(value, max_val), min_val)

    print(f"\n[INFO] 🚀 开始 RTC 推理循环，按 Ctrl+C 停止...\n")

    t_start_run = time.perf_counter()   # 诊断：运行时刻（兼作运行时长计时）
    last_sent_action = None             # 诊断：相邻步跳变检测
    last_flow_action = None             # 流匹配 EMA 上一帧动作
    last_hil_action = None              # HIL 逐关节 EMA 上一帧动作 (J1..J6)
    last_gripper_ratio = 0.5            # 夹爪上一帧发送比例（预设开0.5）
    gripper_state = "open"              # 夹爪二值滞回状态（预设已打开夹爪）
    gripper_dwell = 0                   # 当前夹爪状态已持续步数

    try:
        while not shutdown_event.is_set():
            # ── 0. 运行时长限制（cfg.inference_time_s>0 时到达后自动退出）──
            if cfg.inference_time_s > 0 and (
                time.perf_counter() - t_start_run >= cfg.inference_time_s
            ):
                print(
                    f"[INFO] ⏰ 已运行 {cfg.inference_time_s}s，"
                    f"达到最长运行时长，正在退出..."
                )
                break

            start_loop_t = time.perf_counter()

            # ── 9a. 从队列取一个动作 ──
            action = action_queue.get()

            if action is None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)
                continue

            # ── 9b. 转换为 numpy 并限位 ──
            numpy_action = action.cpu().numpy()
            for i in range(6):
                numpy_action[i] = clamp(
                    numpy_action[i], joint_limits[i][0], joint_limits[i][1]
                )
            numpy_action[6] = clamp(numpy_action[6], 0.0, 1.0)

            # ── 9c. 轨迹平滑（RTC 模式下动态放宽限幅）──
            if smoother is not None:
                smoothed_action = smoother.smooth(
                    numpy_action,
                    rtc_enabled=cfg.rtc.enabled,
                    rtc_multiplier=cfg.rtc_smoother_multiplier,
                )
            else:
                smoothed_action = numpy_action

            # ── 9d. 安全检查 ──
            is_safe, warning_msg = safety_monitor.check(smoothed_action)
            if warning_msg:
                print(f"[WARNING] ⚠️  {warning_msg}")
            if not is_safe:
                print("[CRITICAL] 🛑 安全监控触发紧急停止!")
                break

            # ── 9e.0 关节平滑 EMA（HIL 风格逐关节 或 单α流匹配）──
            # ★ 2026-08-28 同步 HIL 采集优化: hil_smooth_enabled=true 时用
            #   HIL 的逐关节 EMA(α: J1=0.30,J2=0.12,J3=0.18,J4-6=0.30) + J2/J3
            #   限速(0.015 rad/帧)，对模型学得差的 J2/J3 强力抑制突跳/反转
            #   (HIL 采集实测效果优于本脚本原平滑)。
            #   否则回退为原来的单α流匹配 EMA(flow_ema_alpha)。
            #   夹爪不经过 EMA：交给 9e.0b 的二值滞回控制器即时响应。
            if cfg.hil_smooth_enabled:
                if last_hil_action is not None:
                    for _i in range(6):
                        _alpha = cfg.hil_ema_alphas[_i]
                        _ema_v = _alpha * smoothed_action[_i] + (1.0 - _alpha) * last_hil_action[_i]
                        _rate = cfg.hil_rate_limits[_i]
                        if _rate > 0.0:
                            _delta = _ema_v - last_hil_action[_i]
                            if abs(_delta) > _rate:
                                _ema_v = last_hil_action[_i] + (_rate if _delta > 0 else -_rate)
                        smoothed_action[_i] = _ema_v
                last_hil_action = smoothed_action[:6].copy()
            elif cfg.flow_ema_alpha > 0:
                if last_flow_action is not None:
                    smoothed_action[:6] = (
                        cfg.flow_ema_alpha * smoothed_action[:6]
                        + (1.0 - cfg.flow_ema_alpha) * last_flow_action[:6]
                    )
                last_flow_action = smoothed_action.copy()

            # ── 9e.0b 夹爪控制：二值滞回（默认）或连续压缩 ──
            # 模型输出的夹爪意图常在 chunk 内全开↔全闭往返（夹爪极差>0.8），
            # 导致夹爪快速扇动、抓取时刻无法稳定闭合。二值滞回 + 最小驻留
            # 时间把意图转化为稳定"开/闭"状态，闭合时直接发送 0（完全闭合）。
            _ratio = float(smoothed_action[6])
            if cfg.gripper_binary:
                gripper_dwell += 1
                if _ratio > cfg.gripper_open_above and gripper_state != "open":
                    if gripper_dwell >= cfg.gripper_min_dwell:
                        gripper_state = "open"
                        gripper_dwell = 0
                elif _ratio < cfg.gripper_close_below and gripper_state != "closed":
                    if gripper_dwell >= cfg.gripper_min_dwell:
                        gripper_state = "closed"
                        gripper_dwell = 0
                gripper = 0 if gripper_state == "closed" else 70000
            else:
                # 连续模式 + 低位压缩（夹得更紧）
                gripper = round(_ratio * 70 * 1000)
                if 5000 < gripper < 10000:
                    gripper = max(0, gripper - 5000)
                if cfg.gripper_gain > 0 and _ratio < cfg.gripper_gain_above:
                    if _ratio <= cfg.gripper_gain_below:
                        _target = _ratio * cfg.gripper_gain
                    else:
                        _span = cfg.gripper_gain_above - cfg.gripper_gain_below
                        _t = (_ratio - cfg.gripper_gain_below) / _span
                        _target = (
                            cfg.gripper_gain_below * cfg.gripper_gain
                            + _t * (cfg.gripper_gain_above - cfg.gripper_gain_below * cfg.gripper_gain)
                        )
                    gripper = round(_target * 70 * 1000)

            # ── 9e. 发送控制指令（线程安全）──
            joint_0 = int(smoothed_action[0] * RAD_TO_MILLIDEG)
            joint_1 = int(smoothed_action[1] * RAD_TO_MILLIDEG)
            joint_2 = int(smoothed_action[2] * RAD_TO_MILLIDEG)
            joint_3 = int(smoothed_action[3] * RAD_TO_MILLIDEG)
            joint_4 = int(smoothed_action[4] * RAD_TO_MILLIDEG)
            joint_5 = int(smoothed_action[5] * RAD_TO_MILLIDEG)

            if joint_4 < -70000:
                joint_4 = -70000

            with robot_lock:
                motion_ctrl(robot.piper, cfg.motion_speed, cfg.motion_high_follow)
                robot.piper.JointCtrl(
                    joint_0, joint_1, joint_2, joint_3, joint_4, joint_5
                )
                robot.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)

            # ── 9e.5 相邻步跳变检测（定位 chunk 衔接跳变）──
            # ★ 关节与夹爪分开计算：夹爪是 0~1 比例，不能混进"度"的
            #   关节指标（否则夹爪快开快合会把 [JUMP] 刷屏）。
            #   关节：chunk 内相邻步应 <2°；若某步突然 >阈值，大概率
            #   是 chunk 边界（拼接点对齐不准）。夹爪单独用 [GRIP] 报警。
            step_jump_deg = 0.0     # 关节跳变（度）
            gripper_jump = 0.0      # 夹爪跳变（0~1 比例，基于实际发送值）
            if last_sent_action is not None:
                step_jump_deg = float(
                    np.max(np.abs(smoothed_action[:6] - last_sent_action[:6]))
                    * 180.0 / math.pi
                )
                gripper_jump = float(abs(gripper / 70000.0 - last_gripper_ratio))
            last_sent_action = smoothed_action.copy()
            last_gripper_ratio = gripper / 70000.0
            if step_jump_deg > cfg.jump_alert_deg:
                print(
                    f"[JUMP] ⚠️ 步{loop_count + 1} 关节跳变 {step_jump_deg:.1f}° "
                    f"(qsize={action_queue.qsize()})"
                )
            if gripper_jump > cfg.gripper_jump_alert:
                print(
                    f"[GRIP] ⚠️ 步{loop_count + 1} 夹爪跳变 {gripper_jump:.3f} "
                    f"(qsize={action_queue.qsize()})"
                )

            # 执行诊断：间隔 10 步写入 CSV
            if diag is not None and loop_count % 10 == 0:
                diag.log_exec(
                    loop_count,
                    time.perf_counter() - t_start_run,
                    smoothed_action[:6] * 180.0 / math.pi,
                    gripper,
                    step_jump_deg,
                )

            # ── 9f. 频率控制 ──
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            # ── 9g. 统计与日志（loop_dt 含 busy_wait 等待 = 真实控制周期）──
            loop_count += 1
            loop_dt = time.perf_counter() - start_loop_t
            total_loop_time += loop_dt

            if loop_count % stats_interval == 0:
                avg_loop = total_loop_time / loop_count * 1000
                joint_deg = [j * 180.0 / math.pi for j in smoothed_action[:6]]
                qsize = action_queue.qsize()
                print(
                    f"[STATS] 步数:{loop_count:6d} | "
                    f"队列:{qsize:2d} | "
                    f"循环:{avg_loop:5.1f}ms | "
                    f"关节(°):[{joint_deg[0]:6.1f},{joint_deg[1]:6.1f},{joint_deg[2]:6.1f},"
                    f"{joint_deg[3]:6.1f},{joint_deg[4]:6.1f},{joint_deg[5]:6.1f}] | "
                    f"夹爪:{gripper}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] ⏹️  用户中断推理循环")

    finally:
        # ── 优雅退出 ──
        shutdown_event.set()
        infer_thread.join(timeout=5.0)

        print(f"\n{'='*60}")
        print(f"[INFO] 📊 推理结束 - 最终统计")
        print(f"[INFO]    总执行步数: {loop_count}")
        if loop_count > 0:
            avg_loop_ms = total_loop_time / loop_count * 1000
            actual_fps = loop_count / total_loop_time if total_loop_time > 0 else 0
            print(f"[INFO]    平均循环时间: {avg_loop_ms:.1f} ms")
            print(f"[INFO]    实际控制频率: {actual_fps:.1f} Hz")
        print(f"[INFO]    安全警告总数: {safety_monitor._total_warnings}")
        print(f"{'='*60}")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    """
    使用方式：
      python -m lerobot.inference_pi05_flowmatch [参数...]

    【checkpoint 选择】
      --ckpt_path=<pretrained_model 目录>      # 本机 60K 模型 /home/ub/Videos/Evo-RL/model/Pimodel/60K/pretrained_model
      --ckpt_root=<0407 checkpoints 根目录>   # 0407 格式（数字子目录），仅训练机可用
      --ckpt_step=15000                        # 指定步数；0=自动选最新（0407 格式）

    【任务选择（batch5_joint_merged 含 3 个任务，本机无合并数据集时用手动文本）】
      --task_idx=2                             # 从数据集任务表按索引选择（需 --dataset_root 指向本机 batch5X）
      --task='<自由文本>'                      # 自由文本（优先级高于 task_idx，推荐本机用法）
      --list_tasks=true                        # 仅打印可用任务并退出

    【★ 阶段化测试方案（先原始输出，再逐步加优化）】
      第一步（RAW 原始输出）：--raw_mode=true            # 关闭 RTC + 平滑
      第二步（加平滑）：      --raw_mode=false --rtc.enabled=false --use_trajectory_smoother=true
      第三步（平滑+RTC）：    --raw_mode=false --rtc.enabled=true --use_trajectory_smoother=true

    【默认 RTC 参数（针对 Piper 优化，2026-08-25 实测调优）】
      --rtc.enabled=true
      --rtc.execution_horizon=16      # 默认8→16，边界跳变 8.2x→1.6x GT
      --rtc.max_guidance_weight=8.0   # 默认4→8，G=8 为甜点，G=16 过矫正
      --rtc.prefix_attention_schedule=EXP

    【关键参数（本脚本固定流匹配去噪）】
      --inference_time_s=60                      # 最长运行时长（秒），0=不限时，到达自动退出
      --hil_smooth_enabled=true                  # HIL 风格逐关节 EMA+J2/J3限速（对齐 HIL 采集）
      --hil_ema_alphas="0.30,0.12,0.18,0.30,0.30,0.30"  # 逐关节 EMA α (J1..J6)
      --hil_rate_limits="0.0,0.015,0.015,0.0,0.0,0.0"   # 逐关节限速 rad/帧 (0=不限)
      --jump_alert_deg=5.0                       # 关节相邻跳变报警阈值（度）
      --gripper_jump_alert=0.15                  # 夹爪相邻跳变报警阈值（0~1 比例）
      --safety_joint_limit_rad=0.06              # 安全监控关节限幅（rad/步，≈3.4°/步）
      --safety_gripper_limit=0.35                # 安全监控夹爪限幅（0~1）
      --rtc_smoother_multiplier=1.5              # RTC 模式平滑限幅放宽倍数（抑制快速运动）
      --gripper_binary=true                      # 夹爪二值滞回控制（0=回退连续模式）
      --gripper_open_above=0.55                  # 夹爪意图>此值→开
      --gripper_close_below=0.30                 # 夹爪意图<此值→闭
      --gripper_min_dwell=40                     # 夹爪状态切换最小驻留步数
      --action_queue_size_to_get_new_actions=15  # 触发新推理的队列阈值
      --use_trajectory_smoother=true             # 轨迹平滑开关
      --flow_ema_alpha=0.2                       # 流匹配抖动抑制 EMA（仅关节，0 关闭）
      --motion_speed=80                          # 运动速度%（与采集对齐）
      --motion_high_follow=true                  # MIT/高跟随模式（mit_mode=0xAD）
      --gripper_gain=0.40                        # 连续模式夹爪闭合增强（gripper_binary=false 时）
      --raw_mode=true                            # RAW 模式：一键关闭 RTC + 平滑
    """
    inference()
