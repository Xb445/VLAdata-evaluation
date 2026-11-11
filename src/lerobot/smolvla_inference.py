#!/usr/bin/env python3
"""
SmolVLA 模型实时推理与机械臂控制脚本
=============================================

功能：
  1. 加载训练好的 SmolVLA 模型
  2. 接收摄像头图像 + 关节状态，输出动作指令
  3. 支持两种推理模式：
     - predict_action_chunk: 返回完整动作序列（原始模型输出）
     - select_action: 逐帧返回动作（内部队列管理，更平滑）
  4. 可选关节平滑：
     - none: 无平滑（直接使用模型输出）
     - moving_average: 滑动窗口平均
     - exponential: 指数移动平均 (EMA)
  5. 可调整平滑参数

使用方法:
  # 1. 作为模块导入
  from smolvla_inference import SmolVLAInference, ActionSmoother

  # 2. 命令行测试（模拟数据）
  python smolvla_inference.py --mode predict_action_chunk --smoothing moving_average --window_size 5

依赖:
  - torch, numpy, PIL, lerobot (已安装于 lerobot_0407 环境)
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

# ⚠ 必须在任何 transformers / huggingface_hub 导入之前设置离线环境变量！
# huggingface_hub 在导入时缓存 HF_HUB_OFFLINE / HF_HUB_CACHE，之后修改不生效。
_HF_HUB_CACHE = "/home/ub/VLA/pretrained_models/hub"  # 本机 SmolVLM2 backbone 缓存
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = _HF_HUB_CACHE

# 将 lerobot 源码路径加入 sys.path
_LEROBOT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401  # 触发 SmolVLAConfig 注册
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad as _resize_with_pad
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


# ==============================================================================
#  关节平滑器
# ==============================================================================

class ActionSmoother:
    """关节动作平滑器，支持多种平滑策略。

    使用方式:
        smoother = ActionSmoother(method="moving_average", window_size=5)
        smoothed_action = smoother(raw_action)

    Methods:
        none:             不进行平滑，直接返回原始动作
        moving_average:   滑动窗口平均（保留最近 N 帧取均值）
        exponential:      指数移动平均 (EMA)，new = alpha * raw + (1-alpha) * prev
    """

    def __init__(
        self,
        method: str = "none",
        window_size: int = 5,
        alpha: float = 0.3,
        action_dim: int = 7,
    ):
        """
        Args:
            method: 平滑方法，可选 "none", "moving_average", "exponential"
            window_size: 滑动窗口大小（仅 moving_average 有效）
            alpha: EMA 平滑系数（仅 exponential 有效），越小越平滑
            action_dim: 动作维度
        """
        if method not in ("none", "moving_average", "exponential"):
            raise ValueError(f"未知的平滑方法: {method}，可选 none/moving_average/exponential")

        self.method = method
        self.window_size = max(1, window_size)
        self.alpha = alpha
        self.action_dim = action_dim

        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._ema_value: Optional[np.ndarray] = None

    def reset(self):
        """重置平滑器状态（新 episode 开始时调用）。"""
        self._history.clear()
        self._ema_value = None

    def __call__(self, action: np.ndarray) -> np.ndarray:
        """对单帧动作进行平滑处理。

        Args:
            action: shape (action_dim,) 的 numpy 数组

        Returns:
            平滑后的动作，shape (action_dim,)
        """
        if self.method == "none":
            return action

        if self.method == "moving_average":
            self._history.append(action.copy())
            return np.mean(np.stack(list(self._history)), axis=0)

        if self.method == "exponential":
            if self._ema_value is None:
                self._ema_value = action.copy()
            else:
                self._ema_value = self.alpha * action + (1 - self.alpha) * self._ema_value
            return self._ema_value.copy()

    def get_state(self) -> dict:
        """返回平滑器当前状态（用于调试）。"""
        return {
            "method": self.method,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "history_len": len(self._history),
            "ema_initialized": self._ema_value is not None,
        }


# ==============================================================================
#  SmolVLA 推理封装类
# ==============================================================================

class SmolVLAInference:
    """SmolVLA 模型推理封装，处理图像/状态预处理与动作后处理。

    典型用法:
        model = SmolVLAInference(
            checkpoint_path="outputs/train/.../checkpoints/050000/pretrained_model",
        )
        model.load()

        # 从摄像头获取图像（numpy HWC uint8 [0,255]）
        wrist_img = camera.read()   # (480, 640, 3) uint8
        top_img = camera.read()     # (480, 640, 3) uint8
        joint_state = robot.get_joints()  # (7,) float32

        # 获取动作
        action = model.step(
            wrist_img=wrist_img,
            top_img=top_img,
            joint_state=joint_state,
            task="pick up the red block",
        )
        robot.execute(action)

    Modes:
        - select_action (default):  逐帧返回，内部队列管理，推荐用于实时控制
        - predict_action_chunk:     返回完整 chunk，适合离线评估
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        dtype: str = "float32",
        mode: str = "select_action",
        smoother: Optional[ActionSmoother] = None,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        hub_cache: str = "/home/ub/VLA/pretrained_models/hub",
    ):
        """
        Args:
            checkpoint_path: 训练好的模型路径（pretrained_model 目录）
            device: 推理设备
            dtype: 推理精度 ("float32", "bfloat16", "float16")
            mode: 推理模式 ("select_action" 或 "predict_action_chunk")
            smoother: 关节平滑器实例（None 则不进行平滑）
            vlm_model_name: SmolVLM2 backbone 名称/路径。
                训练 config 中记录的绝对路径在本机通常不存在，
                这里默认覆盖为本机 HF 缓存中的 SmolVLM2-500M-Video-Instruct。
            hub_cache: HuggingFace Hub 缓存目录（离线加载 backbone 权重）。
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.dtype = self._parse_dtype(dtype)
        self.mode = mode
        self.smoother = smoother
        self.vlm_model_name = vlm_model_name
        self.hub_cache = hub_cache

        if mode not in ("select_action", "predict_action_chunk"):
            raise ValueError(f"mode 必须为 'select_action' 或 'predict_action_chunk'，当前: {mode}")

        self.policy: Optional[SmolVLAPolicy] = None
        self._dataset_stats: Optional[dict] = None
        self._tokenizer = None
        self._loaded = False

        # 图像预处理参数
        self.resize_size = (512, 512)  # SmolVLA 默认

        # 缓存的 task tokens（避免每帧重复 tokenize）
        self._cached_lang_tokens: Optional[Tensor] = None
        self._cached_lang_masks: Optional[Tensor] = None
        self._cached_task: Optional[str] = None

    @staticmethod
    def _parse_dtype(dtype_str: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"不支持的 dtype: {dtype_str}，可选: {list(mapping.keys())}")
        return mapping[dtype_str]

    # ------------------------------------------------------------------
    #  模型加载
    # ------------------------------------------------------------------

    def load(self):
        """加载模型、数据集统计量和 tokenizer。"""
        print(f"[SmolVLA] 加载模型: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {self.checkpoint_path}")

        # 离线加载 SmolVLM2 backbone（本机 HF 缓存，网络通常不可达）
        # 注意：必须强制赋值（而非 setdefault），否则终端若已设置 HF_HUB_OFFLINE=0
        # 等旧值，仍会尝试联网下载。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_CACHE"] = self.hub_cache

        # 加载策略配置，并覆盖 VLM backbone 名称（训练 config 中的绝对路径在本机不存在）
        config = PreTrainedConfig.from_pretrained(self.checkpoint_path, local_files_only=True)
        config.vlm_model_name = self.vlm_model_name
        print(f"[SmolVLA] VLM backbone: {self.vlm_model_name}")

        # 加载策略（backbone 权重来自 HF 缓存，expert 权重来自 checkpoint）
        self.policy = SmolVLAPolicy.from_pretrained(
            self.checkpoint_path,
            config=config,
            local_files_only=True,
        )
        self.policy.to(self.device).to(self.dtype).eval()
        print(f"[SmolVLA] 模型已加载 (device={self.device}, dtype={self.dtype})")

        # 加载数据集统计量（用于状态归一化/动作反归一化）
        self._load_dataset_stats()

        # 初始化 tokenizer
        self._init_tokenizer()

        self._loaded = True
        print("[SmolVLA] 推理器就绪 ✓")

    def _load_dataset_stats(self):
        """从 checkpoint 目录加载数据集统计量（状态归一化 / 动作反归一化）。

        优先从 checkpoint 自带的 normalizer/unnormalizer safetensors 中提取
        observation.state 与 action 的 mean/std；训练数据集的 meta/stats.json
        通常只存在于训练机上，本机一般拿不到。
        """
        # 优先方案：从 checkpoint 的 preprocessor/postprocessor safetensors 提取
        normalizer_path = (
            self.checkpoint_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        )
        unnormalizer_path = (
            self.checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        try:
            from safetensors import safe_open

            stats = {}
            for key, path, prefix in (
                ("observation.state", normalizer_path, "observation.state"),
                ("action", unnormalizer_path, "action"),
            ):
                if path.exists():
                    with safe_open(str(path), framework="numpy") as f:
                        stats[key] = {
                            "mean": f.get_tensor(f"{prefix}.mean").astype(np.float32),
                            "std": f.get_tensor(f"{prefix}.std").astype(np.float32),
                        }
            if len(stats) == 2:
                self._dataset_stats = stats
                print("[SmolVLA] 已从 checkpoint 提取归一化统计量 "
                      f"(state mean={stats['observation.state']['mean'][:3]}...)")
                return
            print("[SmolVLA] ⚠ checkpoint 中缺少 normalizer/unnormalizer 统计量文件")
        except Exception as e:
            print(f"[SmolVLA] ⚠ 从 checkpoint 提取统计量失败: {e}")

        # 备用方案：从 train_config.json 指向的数据集加载 stats.json
        config_path = self.checkpoint_path / "train_config.json"
        dataset_root = None

        if config_path.exists():
            with open(config_path) as f:
                train_config = json.load(f)
            dataset_root = train_config.get("dataset", {}).get("root", None)

        if dataset_root:
            stats_path = Path(dataset_root) / "meta" / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    self._dataset_stats = json.load(f)
                print(f"[SmolVLA] 加载数据集统计量: {stats_path}")
                return

        print("[SmolVLA] ⚠ 无法定位数据集统计量，将不使用归一化（推理结果可能异常！）")

    def _init_tokenizer(self):
        """初始化 SmolVLM tokenizer。"""
        from transformers import AutoTokenizer

        vlm_name = self.policy.config.vlm_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            vlm_name,
            padding_side="right",
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        print(f"[SmolVLA] Tokenizer 已加载: {vlm_name}")

    # ------------------------------------------------------------------
    #  数据预处理
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: np.ndarray) -> Tensor:
        """将单张图像从 numpy (H,W,C) uint8 [0,255] 转为模型输入 tensor。

        Args:
            img: numpy 数组 (H, W, 3) uint8 [0, 255]

        Returns:
            Tensor (1, 3, 512, 512) float [0, 1]
        """
        # numpy -> tensor, uint8 -> float [0,1]
        img_t = torch.from_numpy(img).float() / 255.0
        # HWC -> CHW
        img_t = img_t.permute(2, 0, 1)  # (3, H, W)
        # 添加 batch 维度
        img_t = img_t.unsqueeze(0)  # (1, 3, H, W)
        # Resize + pad 到 (512, 512)
        img_t = _resize_with_pad(img_t, *self.resize_size, pad_value=0)
        return img_t

    def _preprocess_state(self, joint_state: np.ndarray) -> Tensor:
        """归一化关节状态。

        Args:
            joint_state: (action_dim,) numpy 数组

        Returns:
            归一化后的 Tensor (1, max_state_dim)
        """
        # 使用数据集统计量进行 Z-score 归一化
        if self._dataset_stats is not None:
            try:
                mean = np.array(self._dataset_stats["observation.state"]["mean"], dtype=np.float32)
                std = np.array(self._dataset_stats["observation.state"]["std"], dtype=np.float32)
                joint_state = (joint_state - mean) / (std + 1e-8)
            except (KeyError, ValueError):
                pass

        state_t = torch.from_numpy(joint_state).float()
        state_t = state_t.unsqueeze(0)  # (1, dim)

        # Pad 到 max_state_dim
        max_dim = self.policy.config.max_state_dim
        if state_t.shape[-1] < max_dim:
            padding = torch.zeros(1, max_dim - state_t.shape[-1])
            state_t = torch.cat([state_t, padding], dim=-1)

        return state_t

    def _tokenize_task(self, task: str) -> tuple[Tensor, Tensor]:
        """Tokenize 任务描述文本。

        Args:
            task: 任务描述字符串

        Returns:
            (lang_tokens, lang_masks) 两个 Tensor，shape (1, max_length)
        """
        if task == self._cached_task and self._cached_lang_tokens is not None:
            return self._cached_lang_tokens, self._cached_lang_masks

        # 确保任务描述以换行符结尾（SmolVLA 预处理要求）
        if not task.endswith("\n"):
            task = task + "\n"

        max_length = self.policy.config.tokenizer_max_length
        encoded = self._tokenizer(
            task,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"]  # (1, max_length)
        masks = encoded["attention_mask"]  # (1, max_length)

        self._cached_lang_tokens = tokens
        self._cached_lang_masks = masks
        self._cached_task = task

        return tokens, masks

    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反归一化动作到原始关节空间。

        Args:
            action: 归一化后的动作 (chunk_size,) 或 (action_dim,)

        Returns:
            反归一化后的动作
        """
        if self._dataset_stats is not None:
            try:
                mean = np.array(self._dataset_stats["action"]["mean"], dtype=np.float32)
                std = np.array(self._dataset_stats["action"]["std"], dtype=np.float32)
                action = action * (std + 1e-8) + mean
            except (KeyError, ValueError):
                pass
        return action

    # ------------------------------------------------------------------
    #  推理接口
    # ------------------------------------------------------------------

    def predict_action_chunk(
        self,
        wrist_img: np.ndarray,
        top_img: np.ndarray,
        joint_state: np.ndarray,
        task: str = "",
    ) -> np.ndarray:
        """返回完整动作序列（chunk_size, action_dim）。

        这是模型的原始输出，不经过队列管理。

        Args:
            wrist_img: 腕部摄像头图像 (H, W, 3) uint8
            top_img: 顶部摄像头图像 (H, W, 3) uint8
            joint_state: 当前关节状态 (action_dim,) float
            task: 任务描述文本

        Returns:
            actions: (chunk_size, action_dim) 完整的动作序列
        """
        self._ensure_loaded()

        # 预处理图像
        wrist_t = self._preprocess_image(wrist_img).to(self.device).to(self.dtype)
        top_t = self._preprocess_image(top_img).to(self.device).to(self.dtype)

        # 预处理状态
        state_t = self._preprocess_state(joint_state).to(self.device).to(self.dtype)

        # Tokenize 任务（attention_mask 需转为 bool 类型，模型内部要求）
        lang_tokens, lang_masks = self._tokenize_task(task)
        lang_tokens = lang_tokens.to(self.device)
        lang_masks = lang_masks.to(dtype=torch.bool, device=self.device)

        # 构建 batch
        batch = {
            f"{OBS_IMAGES}.wrist": wrist_t,
            f"{OBS_IMAGES}.top": top_t,
            OBS_STATE: state_t,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        }

        with torch.no_grad():
            # 显式生成与模型同 dtype 的噪声（smolvla 内部 sample_noise 固定用 float32，
            # 当模型使用 bfloat16 时会导致 dtype mismatch，因此需要外部传入）
            bsize = 1
            noise = torch.normal(
                mean=0.0, std=1.0,
                size=(bsize, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                dtype=self.dtype, device=self.device,
            )
            actions = self.policy.predict_action_chunk(batch, noise=noise)  # (1, chunk_size, action_dim)

        actions = actions.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
        actions = self._unnormalize_action(actions)

        return actions

    def select_action(
        self,
        wrist_img: np.ndarray,
        top_img: np.ndarray,
        joint_state: np.ndarray,
        task: str = "",
    ) -> np.ndarray:
        """返回单帧动作（action_dim,）。

        使用内部队列管理：模型只在队列耗尽时推理一次，后续从队列取结果。

        Args:
            wrist_img: 腕部摄像头图像 (H, W, 3) uint8
            top_img: 顶部摄像头图像 (H, W, 3) uint8
            joint_state: 当前关节状态 (action_dim,) float
            task: 任务描述文本

        Returns:
            action: (action_dim,) 单帧动作
        """
        self._ensure_loaded()

        # 预处理图像
        wrist_t = self._preprocess_image(wrist_img).to(self.device).to(self.dtype)
        top_t = self._preprocess_image(top_img).to(self.device).to(self.dtype)

        # 预处理状态
        state_t = self._preprocess_state(joint_state).to(self.device).to(self.dtype)

        # Tokenize 任务（attention_mask 需转为 bool 类型）
        lang_tokens, lang_masks = self._tokenize_task(task)
        lang_tokens = lang_tokens.to(self.device)
        lang_masks = lang_masks.to(dtype=torch.bool, device=self.device)

        # 构建 batch
        batch = {
            f"{OBS_IMAGES}.wrist": wrist_t,
            f"{OBS_IMAGES}.top": top_t,
            OBS_STATE: state_t,
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        }

        with torch.no_grad():
            # 显式传入与模型同 dtype 的噪声，避免 bfloat16 下 dtype mismatch
            bsize = 1
            noise = torch.normal(
                mean=0.0, std=1.0,
                size=(bsize, self.policy.config.chunk_size, self.policy.config.max_action_dim),
                dtype=self.dtype, device=self.device,
            )
            action = self.policy.select_action(batch, noise=noise)  # (1, action_dim)

        action = action.squeeze(0).cpu().numpy()  # (action_dim,)
        action = self._unnormalize_action(action)

        return action

    def step(
        self,
        wrist_img: np.ndarray,
        top_img: np.ndarray,
        joint_state: np.ndarray,
        task: str = "",
    ) -> np.ndarray:
        """统一推理入口。根据初始化时指定的 mode 调用对应方法。

        如果配置了 smoother，在此处自动进行平滑处理。
        """
        if self.mode == "select_action":
            action = self.select_action(wrist_img, top_img, joint_state, task)
        else:
            # predict_action_chunk 模式：返回第一个动作
            chunk = self.predict_action_chunk(wrist_img, top_img, joint_state, task)
            action = chunk[0]  # 仅取第一帧

        # 应用平滑
        if self.smoother is not None:
            action = self.smoother(action)

        return action

    def reset(self):
        """重置模型内部状态和新 episode 的平滑器。"""
        if self.policy is not None:
            self.policy.reset()
        if self.smoother is not None:
            self.smoother.reset()
        self._cached_lang_tokens = None
        self._cached_lang_masks = None
        self._cached_task = None

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("模型尚未加载，请先调用 .load()")

    # ------------------------------------------------------------------
    #  工具方法
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self.policy.config.chunk_size if self.policy else 50

    @property
    def n_action_steps(self) -> int:
        return self.policy.config.n_action_steps if self.policy else 50

    @property
    def action_dim(self) -> int:
        if self.policy is not None and hasattr(self.policy.config, "action_feature"):
            return self.policy.config.action_feature.shape[0]
        return 7

    def get_queue_size(self) -> int:
        """返回当前 select_action 内部队列大小（用于调试）。"""
        if self.policy is None:
            return 0
        queue = self.policy._queues.get(ACTION, None)
        if queue is None:
            return 0
        return len(queue)


# ==============================================================================
#  命令行接口（用于快速测试）
# ==============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA 模型推理脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ub/Videos/models/smolVLa_5W_bs64/pretrained_model",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--vlm_model_name",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="SmolVLM2 backbone 名称（使用本机 HF 缓存离线加载）",
    )
    parser.add_argument(
        "--hub_cache",
        type=str,
        default="/home/ub/VLA/pretrained_models/hub",
        help="HuggingFace Hub 缓存目录（离线加载 backbone）",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"], help="推理精度"
    )
    parser.add_argument(
        "--mode", type=str, default="predict_action_chunk",
        choices=["predict_action_chunk", "select_action"],
        help="推理模式: predict_action_chunk (返回完整chunk) / select_action (逐帧返回)"
    )
    parser.add_argument(
        "--smoothing", type=str, default="none",
        choices=["none", "moving_average", "exponential"],
        help="关节平滑方法"
    )
    parser.add_argument(
        "--window_size", type=int, default=5,
        help="滑动窗口大小（moving_average 有效）"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="EMA 平滑系数（exponential 有效，越小越平滑）"
    )
    parser.add_argument(
        "--task", type=str, default="perform the task\n",
        help="任务描述文本"
    )
    parser.add_argument(
        "--steps", type=int, default=3,
        help="测试推理步数（仅模拟模式）"
    )
    parser.add_argument(
        "--image_height", type=int, default=480, help="模拟图像高度"
    )
    parser.add_argument(
        "--image_width", type=int, default=640, help="模拟图像宽度"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # -- 创建平滑器 --
    smoother = ActionSmoother(
        method=args.smoothing,
        window_size=args.window_size,
        alpha=args.alpha,
    )

    # -- 创建推理器 --
    model = SmolVLAInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        mode=args.mode,
        smoother=smoother,
        vlm_model_name=args.vlm_model_name,
        hub_cache=args.hub_cache,
    )
    model.load()

    # -- 打印模型信息 --
    print(f"\n{'='*60}")
    print(f"  模型配置")
    print(f"{'='*60}")
    print(f"  chunk_size:       {model.chunk_size}")
    print(f"  n_action_steps:   {model.n_action_steps}")
    print(f"  action_dim:       {model.action_dim}")
    print(f"  推理模式:         {args.mode}")
    print(f"  平滑方法:         {args.smoothing}")
    if args.smoothing != "none":
        print(f"    窗口大小:       {args.window_size}" if args.smoothing == "moving_average" else f"    alpha:          {args.alpha}")
    print(f"{'='*60}\n")

    # -- 模拟推理 --
    print("开始推理测试（使用模拟图像数据）...\n")

    for step_idx in range(args.steps):
        # 模拟摄像头数据
        wrist_img = np.random.randint(0, 255, (args.image_height, args.image_width, 3), dtype=np.uint8)
        top_img = np.random.randint(0, 255, (args.image_height, args.image_width, 3), dtype=np.uint8)

        # 模拟关节状态（7维：6个关节 + 1个夹爪）
        joint_state = np.array([0.0, -0.5, 0.3, 1.2, -0.8, 0.1, 0.5], dtype=np.float32)

        t0 = time.time()

        if args.mode == "predict_action_chunk":
            actions = model.predict_action_chunk(wrist_img, top_img, joint_state, task=args.task)
            elapsed = time.time() - t0
            print(f"[{step_idx+1}/{args.steps}] predict_action_chunk | "
                  f"shape={actions.shape} | "
                  f"耗时={elapsed*1000:.1f}ms")
            print(f"      前3帧动作: {actions[:3]}")
        else:
            # select_action 模式：连续调用 n_action_steps 次测试队列行为
            all_actions = []
            for i in range(model.n_action_steps):
                action = model.step(wrist_img, top_img, joint_state, task=args.task)
                all_actions.append(action)
            elapsed = time.time() - t0
            print(f"[{step_idx+1}/{args.steps}] select_action × {model.n_action_steps} | "
                  f"耗时={elapsed*1000:.1f}ms | "
                  f"队列剩余={model.get_queue_size()}")
            print(f"      最后3帧动作: {np.stack(all_actions[-3:])}")

        model.reset()

    print(f"\n推理测试完成 ✓")

    # 打印平滑器状态
    print(f"\n平滑器状态: {smoother.get_state()}")


def _resize_with_pad(img, width, height, pad_value=-1):
    """图像 resize + pad，保持宽高比。"""
    import torch.nn.functional as F
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")
    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


if __name__ == "__main__":
    main()
