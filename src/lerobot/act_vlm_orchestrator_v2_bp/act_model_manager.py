"""
ACT 模型管理器 — 负责多 ACT 模型的加载、切换与缓存清理

核心功能:
    1. 预加载/懒加载多个 ACT 子任务模型 (Task_A, Task_B, Task_C)
    2. 切换模型时清除 GPU 缓存 + 重置模型状态, 消除累积误差
    3. 支持 standby 模式 (不加载模型, 机械臂待机)
    4. 模型状态追踪与统计
"""

import gc
import logging
import time
from typing import Optional, Dict, Any, Tuple

import torch
import numpy as np

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig

logger = logging.getLogger(__name__)


class GripperPassthroughEnsembler:
    """
    ACT 时序集成包装器 — 夹爪维度透传原始值。

    背景:
        标准 ACTTemporalEnsembler 对所有动作维度做加权平均。夹爪的开/合信号是
        二值的 (0=闭合, ~1=打开), 时序集成会把不同时刻的预测平均成中间值
        (如 0.9 与 0.0 平均 → ~0.45), 导致夹爪无法干净开合, 抓取/放置不可靠。

    本包装器:
        - 关节维度: 交由内部 ACTTemporalEnsembler 做时序集成 (抑制 chunk 边界抖动)
        - 夹爪维度: 透传当前步的原始 (未集成) 值, 保持开/合信号清晰

    接口与 ACTTemporalEnsembler 完全一致 (reset / update), 可无缝替换。
    """

    def __init__(self, inner_ensembler, gripper_idx: int = 6):
        self._inner = inner_ensembler
        self._gripper_idx = gripper_idx

    def reset(self):
        """重置内部时序集成状态"""
        self._inner.reset()

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """
        对当前预测 chunk 做时序集成, 但夹爪维度用原始值。

        Args:
            actions: (batch, chunk_size, action_dim) 当前预测的完整 chunk

        Returns:
            (batch, action_dim) 集成后的当前步动作 (夹爪为原始值)
        """
        # 当前步的原始夹爪值: actions[..., 0, gripper_idx]
        raw_gripper = actions[..., 0, self._gripper_idx].clone()
        # 关节做时序集成
        ensembled = self._inner.update(actions)
        # 夹爪维度替换为原始值
        ensembled[..., self._gripper_idx] = raw_gripper
        return ensembled


class ACTModelManager:
    """
    ACT 模型生命周期管理器

    模型切换流程:
        1. 保存当前模型状态
        2. 将当前模型移到 CPU
        3. 删除模型引用
        4. 清空 CUDA 缓存
        5. 运行 Python GC
        6. 加载目标模型到 GPU
        7. 重置模型内部状态

    这确保每次切换都是"干净"的, 不会有累积的隐状态误差。
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._device = config.device
        self._model_paths = config.model_paths

        # 模型注册表: {model_name: {"policy": ..., "preprocess": ..., "postprocess": ...}}
        self._models: Dict[str, Dict[str, Any]] = {}

        # 当前激活的模型
        self._active_model: str = ""  # "Task_A" / "Task_B" / "Task_C" / "" (standby)

        # 预处理器和后处理器
        self._preprocess_fn = None
        self._postprocess_fn = None

        # 统计
        self._switch_count: int = 0
        self._total_inference_steps: int = 0
        self._model_inference_steps: Dict[str, int] = {}

    # ========================================================================
    # 模型加载
    # ========================================================================

    def load_model(self, model_name: str, preload: bool = False) -> bool:
        """
        加载单个 ACT 模型。

        Args:
            model_name: "Task_A" / "Task_B" / "Task_C"
            preload: 是否只是预加载 (不激活)

        Returns:
            True 如果加载成功
        """
        if model_name in self._models:
            logger.info(f"[ModelMgr] 模型 {model_name} 已加载, 跳过")
            return True

        model_path = self._model_paths.get(model_name)
        if not model_path:
            logger.error(f"[ModelMgr] 模型 {model_name} 路径未配置")
            return False

        logger.info(f"[ModelMgr] 加载 ACT 模型: {model_name} ← {model_path}")
        t_start = time.time()

        try:
            from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
            from lerobot.policies.factory import make_pre_post_processors

            policy = ACTPolicy.from_pretrained(model_path)
            preprocess, postprocess = make_pre_post_processors(policy.config, model_path)

            # ── 启用 ACT 时序集成 (temporal ensembling) ──
            # 模型训练时 temporal_ensemble_coeff=None (队列模式), 每 chunk_size 步
            # 重新预测一次 chunk。队列模式在 chunk 边界处, 相邻 chunk 的首动作可能
            # 大幅跳变 → 机械臂"小幅度上升下降循环"振荡 (场景/姿态偏离训练分布时尤甚)。
            # 推理时开启时序集成: 每步预测完整 chunk 并与前序 chunk 按指数衰减权重
            # 融合, 显著抑制 chunk 边界振荡。实测单次 chunk 推理 ~5ms, 不影响实时性。
            # 注意: ACTPolicy.__init__ 根据加载时的 config 决定是否创建 ensembler,
            # 这里在加载后手动启用 (select_action 会检查 config.temporal_ensemble_coeff)。
            ens_coeff = self.config.act_temporal_ensemble_coeff
            if ens_coeff:
                policy.config.temporal_ensemble_coeff = float(ens_coeff)
                inner_ensembler = ACTTemporalEnsembler(
                    float(ens_coeff), policy.config.chunk_size
                )
                # 默认排除夹爪维度: 夹爪透传当前原始值, 保持开/合信号清晰。
                # 标准时序集成会平均夹爪信号 (如 0.9/0.0 → 中间值), 导致夹爪
                # 无法干净开合, 影响抓取/放置。
                if self.config.act_ensemble_exclude_gripper:
                    policy.temporal_ensembler = GripperPassthroughEnsembler(
                        inner_ensembler, gripper_idx=6
                    )
                    logger.info(
                        f"[ModelMgr] ✅ {model_name} 启用时序集成 "
                        f"(coeff={ens_coeff}, 夹爪维度透传原始值)"
                    )
                else:
                    policy.temporal_ensembler = inner_ensembler
                    logger.info(
                        f"[ModelMgr] ✅ {model_name} 启用时序集成 "
                        f"(coeff={ens_coeff}, 全部维度参与集成)"
                    )
            else:
                logger.info(
                    f"[ModelMgr] {model_name} 未启用时序集成 "
                    f"(act_temporal_ensemble_coeff={ens_coeff})"
                )

            policy.eval()
            policy.to(self._device)
            policy.reset()

            self._models[model_name] = {
                "policy": policy,
                "preprocess": preprocess,
                "postprocess": postprocess,
                "path": model_path,
                "config": policy.config,
            }
            self._model_inference_steps[model_name] = 0

            elapsed = time.time() - t_start
            logger.info(
                f"[ModelMgr] ✅ {model_name} 加载完成 ({elapsed:.1f}s) "
                f"chunk_size={policy.config.chunk_size}, "
                f"device={self._device}"
            )

            if not preload:
                self._activate_model(model_name)

            return True

        except Exception as e:
            logger.error(f"[ModelMgr] ❌ 加载模型 {model_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _activate_model(self, model_name: str):
        """激活指定模型 (设置预处理器/后处理器引用)"""
        if model_name not in self._models:
            raise RuntimeError(f"模型 {model_name} 未加载, 无法激活")

        model_info = self._models[model_name]
        self._preprocess_fn = model_info["preprocess"]
        self._postprocess_fn = model_info["postprocess"]
        self._active_model = model_name
        logger.info(f"[ModelMgr] 激活模型: {model_name}")

    # ========================================================================
    # 模型切换 (核心功能)
    # ========================================================================

    def switch_model(self, target_model: str) -> bool:
        """
        切换到目标 ACT 模型, 同时清除旧模型的 GPU 缓存和累积误差。

        Args:
            target_model: "Task_A" / "Task_B" / "Task_C" / "standby"

        Returns:
            True 如果切换成功
        """
        # standby 模式: 卸载所有模型, 机械臂待机
        if target_model == "standby":
            return self._enter_standby()

        # 已经是目标模型, 无需切换 (同模型切换直接返回成功)
        if target_model == self._active_model:
            logger.debug(f"[ModelMgr] 已是目标模型 {target_model}, 无需切换")
            return True

        # ── 目标模型未加载 → 先加载 ──
        if target_model not in self._models:
            # 检查模型路径是否已配置
            model_path = self._model_paths.get(target_model)
            if not model_path:
                logger.error(
                    f"[ModelMgr] ❌ 模型 {target_model} 路径未配置, "
                    f"已配置的模型: {list(self._model_paths.keys())}"
                )
                return False

            logger.info(f"[ModelMgr] 目标模型 {target_model} 未加载, 正在加载...")
            if not self.load_model(target_model):
                logger.error(f"[ModelMgr] ❌ 加载模型 {target_model} 失败, 切换中止")
                return False
            # load_model 中已调用 _activate_model
            self._switch_count += 1
            return True

        # ── 执行完整的模型切换流程 ──
        logger.info(f"[ModelMgr] 🔄 模型切换: {self._active_model or '无'} → {target_model}")
        t_start = time.time()

        old_model = self._active_model

        # 步骤1: 重置旧模型状态 (如果有)
        if old_model and old_model in self._models:
            old_policy = self._models[old_model]["policy"]
            try:
                old_policy.reset()
            except Exception as e:
                logger.debug(f"[ModelMgr] 旧模型 reset 异常 (可忽略): {e}")

        # 步骤2: 清空激活引用
        self._preprocess_fn = None
        self._postprocess_fn = None
        self._active_model = ""

        # 步骤3: 清理 GPU 缓存
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug(f"[ModelMgr] CUDA 缓存已清空 "
                         f"(allocated: {torch.cuda.memory_allocated() / 1024**2:.0f}MB, "
                         f"reserved: {torch.cuda.memory_reserved() / 1024**2:.0f}MB)")

        # 步骤4: Python GC
        gc.collect()

        # 步骤5: 激活目标模型
        try:
            self._activate_model(target_model)
        except RuntimeError as e:
            logger.error(f"[ModelMgr] ❌ 激活模型 {target_model} 失败: {e}")
            # 尝试恢复旧模型
            if old_model and old_model in self._models:
                try:
                    self._activate_model(old_model)
                    logger.warning(f"[ModelMgr] 已回退到旧模型: {old_model}")
                except Exception:
                    logger.error("[ModelMgr] 回退旧模型也失败, 进入待机状态")
                    self._enter_standby()
            return False

        # 步骤5.5: 确保目标模型在正确的设备上 (可能从待机恢复, 在 CPU 上)
        target_policy = self._models[target_model]["policy"]
        try:
            # 检查模型当前所在设备
            target_device = next(target_policy.parameters()).device
            if str(target_device) != str(self._device):
                logger.info(f"[ModelMgr] 模型从 {target_device} 移动到 {self._device}")
                target_policy.to(self._device)
        except (StopIteration, Exception) as e:
            logger.debug(f"[ModelMgr] 无法检测模型设备 (可能无参数): {e}")

        # 步骤6: 重置目标模型状态
        try:
            target_policy.reset()
        except Exception as e:
            logger.debug(f"[ModelMgr] 目标模型 reset 异常 (可忽略): {e}")

        self._switch_count += 1
        elapsed = time.time() - t_start
        logger.info(f"[ModelMgr] ✅ 切换完成 ({elapsed:.1f}s), 当前模型: {target_model}")

        return True

    def _enter_standby(self) -> bool:
        """进入待机状态: 将模型移到 CPU, 清理 GPU"""
        if not self._active_model:
            logger.debug("[ModelMgr] 已在待机状态")
            return True

        logger.info(f"[ModelMgr] ⏸️ 进入待机状态 (卸载 {self._active_model})")
        old_model = self._active_model

        # 将当前模型移到 CPU 以释放 GPU 显存 (保留模型引用以便快速恢复)
        # standby_keep_on_gpu=True 时保留在 GPU, 恢复更迅速 (避免 1~2s 空窗)
        if old_model in self._models:
            old_policy = self._models[old_model]["policy"]
            try:
                old_policy.reset()
            except Exception:
                pass
            if self._device != "cpu" and not self.config.standby_keep_on_gpu:
                try:
                    old_policy.to("cpu")
                    logger.debug(f"[ModelMgr] 模型 {old_model} 已移到 CPU")
                except Exception as e:
                    logger.warning(f"[ModelMgr] 模型移到 CPU 失败: {e}")

        # 清空激活状态
        self._preprocess_fn = None
        self._postprocess_fn = None
        self._active_model = ""

        # 清理 GPU
        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        logger.info("[ModelMgr] ✅ 待机就绪, GPU 缓存已清空")
        return True

    # ========================================================================
    # 推理接口
    # ========================================================================

    def infer_step(
        self,
        observation_frame: Dict[str, Any],
        task_prompt: str = "",
    ) -> np.ndarray:
        """
        执行一步 ACT 模型推理。

        Args:
            observation_frame: 预处理前的原始观测帧
            task_prompt: 任务描述 (可选, 用于某些模型)

        Returns:
            numpy action array, shape (7,) [joint1~6, gripper]

        Raises:
            RuntimeError: 没有激活的 ACT 模型
            KeyError: 观测帧缺少必需字段
        """
        if not self._active_model or self._active_model == "standby":
            raise RuntimeError("没有激活的 ACT 模型, 无法推理")

        # 验证必需字段
        if 'observation.state' not in observation_frame:
            raise KeyError("observation_frame 缺少 'observation.state' 字段")
        required_image_key = None
        for key in observation_frame:
            if 'image' in key:
                required_image_key = key
                break
        if required_image_key is None:
            raise KeyError("observation_frame 缺少图像字段 (observation.images.*)")

        self._total_inference_steps += 1
        self._model_inference_steps[self._active_model] = \
            self._model_inference_steps.get(self._active_model, 0) + 1

        policy = self._models[self._active_model]["policy"]

        # 预处理
        processed = {}
        for name in observation_frame:
            raw = observation_frame[name]
            if raw is None:
                raise ValueError(f"observation_frame 字段 '{name}' 为 None")
            if "image" in name:
                obs = raw.astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))
            else:
                obs = raw
            obs = np.expand_dims(obs, axis=0)
            processed[name] = torch.tensor(obs, dtype=torch.float32, device=self._device)

        processed["task"] = [task_prompt] if task_prompt else ["Perform the manipulation task"]
        processed = self._preprocess_fn(processed)

        # 推理
        with torch.inference_mode():
            action = policy.select_action(processed)

        # 后处理
        action = self._postprocess_fn(action)
        numpy_action = action.squeeze(0).cpu().numpy()
        return numpy_action

    def reset_active_policy(self) -> bool:
        """
        重置当前激活模型的内部状态 (清空动作队列 / 时序集成器)。

        用途: VLM 连拍结束后调用, 确保 ACT 从当前观测重新预测 chunk,
        避免回放连拍前预测的陈旧动作。

        背景: ACT 模型通过 action queue 缓存整段 chunk (n_action_steps 步),
        每次 select_action 只取 1 步。VLM 连拍期间 ACT 推理被暂停, 但 queue
        不会被清空 (同模型不切换时 switch_model 不会执行), 连拍结束后会
        继续回放连拍前基于旧观测预测的陈旧 chunk → 表现为"先抬升再恢复再抓取"
        的抖动。此方法在连拍结束时强制清空, 让下一次推理基于最新观测重新预测。

        Returns:
            True 如果重置成功
        """
        if not self._active_model or self._active_model == "standby":
            return False

        policy = self._models[self._active_model]["policy"]
        try:
            policy.reset()
            logger.info(f"[ModelMgr] 已重置 {self._active_model} 内部状态 (清空动作队列)")
            return True
        except Exception as e:
            logger.debug(f"[ModelMgr] 重置 {self._active_model} 内部状态失败: {e}")
            return False

    def is_new_chunk(self) -> bool:
        """
        判断下一次 infer_step 是否会开始一个新的动作 chunk (队列模式)。

        队列模式下 ACTPolicy 用 _action_queue 缓存整段 chunk (n_action_steps 步),
        每次 select_action 取 1 步, 队列耗尽时重新预测新 chunk。
        当队列为空时, 下一次 select_action 必然预测新 chunk → 即 chunk 边界。

        用途: orchestrator 在推理前检测到新 chunk 开始, 触发平滑器的
        chunk 边界过渡平滑, 抑制新 chunk 首动作与上一 chunk 末动作的跳变
        (chunk 间抖动)。

        Returns:
            True 如果下一次推理会开始新 chunk
        """
        if not self._active_model or self._active_model == "standby":
            return False

        policy = self._models[self._active_model]["policy"]
        queue = getattr(policy, "_action_queue", None)
        # 队列为空 (或不存在) → 下一次 select_action 会重新预测新 chunk
        return queue is None or len(queue) == 0

    # ========================================================================
    # 预加载
    # ========================================================================

    def preload_all(self) -> Dict[str, bool]:
        """预加载所有配置的 ACT 模型"""
        results = {}
        for model_name in self._model_paths:
            if model_name in ("standby",):
                continue
            results[model_name] = self.load_model(model_name, preload=True)
            logger.info(f"[ModelMgr] 预加载 {model_name}: {'✅' if results[model_name] else '❌'}")

        # 预加载完成后激活默认模型
        if self.config.default_model in self._models:
            self._activate_model(self.config.default_model)
        elif self._models:
            first_model = next(iter(self._models))
            self._activate_model(first_model)

        return results

    def preload_and_activate(self, model_name: str) -> bool:
        """加载并激活指定模型"""
        if self.load_model(model_name):
            self._activate_model(model_name)
            return True
        return False

    # ========================================================================
    # 状态查询
    # ========================================================================

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def is_standby(self) -> bool:
        return self._active_model == "" or self._active_model == "standby"

    @property
    def loaded_models(self) -> list:
        return list(self._models.keys())

    @property
    def stats(self) -> dict:
        return {
            "active_model": self._active_model or "standby",
            "loaded_models": self.loaded_models,
            "total_switches": self._switch_count,
            "total_inference_steps": self._total_inference_steps,
            "per_model_steps": dict(self._model_inference_steps),
            "gpu_allocated_mb": (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available() else 0
            ),
        }

    # ========================================================================
    # 清理
    # ========================================================================

    def cleanup(self):
        """清理所有模型, 释放 GPU 资源"""
        logger.info("[ModelMgr] 清理所有模型...")

        for model_name in list(self._models.keys()):
            try:
                self._models[model_name]["policy"].reset()
                del self._models[model_name]["policy"]
            except Exception:
                pass
            try:
                del self._models[model_name]
            except Exception:
                pass

        self._models.clear()
        self._preprocess_fn = None
        self._postprocess_fn = None
        self._active_model = ""

        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        logger.info("[ModelMgr] ✅ 清理完成")
