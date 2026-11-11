"""
PI0.5 模型管理器 — 单模型 + 任务文本切换
=========================================

核心设计:
    PI0.5 (Paligemma 3B VLA) 天然支持多任务: 同一份权重通过不同的 task 文本
    (任务描述) 条件化输出不同行为。因此编排器只加载 ONE 个 PI 模型,
    由 VLM 调度在 Task_A/B/C 之间切换时, 仅切换 task 文本 + 重置推理状态,
    不卸载/加载模型权重, 切换开销极小。

任务文本映射 (必须与训练数据集的 task 逐字一致, 见 config.TASK_PROMPTS):
    Task_A: "Grab the package and place it on the barcode scanning table."
    Task_B: "Flip the package if the barcode is not facing up."
    Task_C: "Grab the scanned package and place it into the box."

推理架构 (对齐 inference_pi05_adarms_fix.py 的多线程 RTC 方案):
    - 主线程: 从 ActionQueue 取动作 → 平滑 → 安全检查 → 发送机械臂
    - 后台线程: 监控队列剩余步数, 低于阈值时取最新观测 → predict_action_chunk
      (RTC 引导) → 后处理 → merge 回队列

任务切换流程 (switch_model):
    1. 停止当前后台推理线程 (避免旧任务文本的推理写入新队列)
    2. 重置策略内部状态 (policy.reset) + 新建空 ActionQueue (RTC 对齐归零)
    3. 更新当前任务标签与任务文本 (后台线程下次预测使用新文本)
    4. 重启后台推理线程 (模型已加载/预热, 无需重新 warmup)
"""

import gc
import json
import logging
import math
import os
import threading
import time
import traceback
from typing import Optional, Dict, Any

import numpy as np
import torch

from lerobot.pi05_vlm_orchestrator.config import (
    OrchestratorConfig,
    LOCAL_TOKENIZER,
    TASK_PROMPTS,
    DEFAULT_TASK_PROMPT,
)
from lerobot.pi05_vlm_orchestrator.diag_logger import DiagLogger

logger = logging.getLogger(__name__)


# ============================================================================
# 智能延迟追踪器 (EMA 平滑)
# ============================================================================

class SmartLatencyTracker:
    """
    EMA 平滑的推理延迟追踪器。

    对系统偶发抖动 (Jitter) 不敏感: 一次 300ms 的异常推理不会让后续所有
    RTC 对齐都按 300ms 预留窗口, 避免动作反应变迟钝。
    """

    def __init__(self, alpha: float = 0.15, margin: float = 1.1, initial_estimate: float = 0.200):
        self.alpha = alpha
        self.margin = margin
        self._ema_latency = initial_estimate

    def update(self, new_latency: float):
        """快速上升机制: 新延迟明显高于当前 EMA 时临时提高 alpha 到 0.5"""
        effective_alpha = self.alpha if new_latency <= self._ema_latency else 0.5
        self._ema_latency = (
            effective_alpha * new_latency + (1 - effective_alpha) * self._ema_latency
        )

    def get_estimate(self) -> float:
        """返回带安全余量的平滑延迟估计 (秒)"""
        return self._ema_latency * self.margin

    @property
    def raw_ema(self) -> float:
        """返回原始 EMA 值 (不含余量), 用于调试。"""
        return self._ema_latency


# ============================================================================
# PI0.5 checkpoint 加载辅助 (对齐 inference_pi05_adarms_fix.py)
# ============================================================================

def load_pi05_policy(ckpt_path: str) -> Any:
    """
    加载 PI0.5 checkpoint, 返回可推理的 PI05Policy。

    流程:
      1. 读取 config.json 并适配 (剔除 "type" 字段; input/output_features
         普通 dict → PolicyFeature; tokenizer_name 覆盖为本机离线路径)
      2. PI05Policy.from_pretrained(ckpt_path, config=config)
         —— 内部已处理权重键重映射 (model. 前缀 / AdaLN .dense / lm_head tied)
    """
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到 config.json: {config_path}")

    with open(config_path) as f:
        cfg_dict = json.load(f)

    # 1. 剔除 0407 特有的 "type" 字段 (Evo-RL PI05Config 无此字段)
    cfg_dict.pop("type", None)

    # 2. 特征 dict → PolicyFeature
    from lerobot.configs.types import FeatureType, PolicyFeature

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

    # 3. tokenizer_name 强制指向本机离线路径
    cfg_dict["tokenizer_name"] = LOCAL_TOKENIZER

    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = PI05Config(**cfg_dict)
    logger.info(
        f"[ModelMgr] 🧠 PI05 config: chunk_size={config.chunk_size}, "
        f"max_state_dim={config.max_state_dim}, "
        f"image_features={list(config.image_features.keys())}, "
        f"tokenizer={config.tokenizer_name}"
    )
    policy = PI05Policy.from_pretrained(ckpt_path, config=config)
    return policy


def make_deterministic_noise(policy: Any, bsize: int, device: str) -> torch.Tensor:
    """零噪声确定性采样噪声 (形状与模型内部采样一致)。"""
    return torch.zeros(
        bsize,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
    )


# ============================================================================
# PI 模型管理器 (单模型 + 任务文本切换)
# ============================================================================

class PIModelManager:
    """
    PI0.5 模型生命周期管理器 + 后台 RTC 推理引擎 (单模型多任务)

    只加载一个 PI 模型。VLM 调度任务切换时:
        switch_model("Task_A"|"Task_B"|"Task_C") → 仅切换 task 文本 + 重置推理状态
        switch_model("standby")                  → 待机 (模型保留在 GPU)

    推理接口:
        infer_step(observation_frame, task_prompt) 从 ActionQueue 取一个动作
        (已后处理)。队列为空 (后台推理进行中) 时返回 None, 主循环保持姿态等待。
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._device = config.device
        self._model_path = config.model_path

        # RTC 配置
        self._rtc_config = config.build_rtc_config()

        # 单个模型
        self._policy = None
        self._preprocess_fn = None
        self._postprocess_fn = None
        self._policy_config = None

        # 当前任务状态
        self._active_task: str = ""       # 任务标签: "Task_A"/"Task_B"/"Task_C"/"" (standby)
        self._active_task_prompt: str = DEFAULT_TASK_PROMPT
        self._task_prompt_lock = threading.Lock()

        # 动作队列 (RTC) — 后台推理线程填充, 主线程消费
        self._action_queue = None

        # 后台推理线程
        self._infer_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._thread_warned_dead: bool = False
        self._warmup_done: bool = False    # CUDA 预热只需一次 (模型不换)

        # 机器人 / 相机引用 (由 orchestrator.initialize 注入)
        self._robot = None
        self._camera_manager = None

        # 诊断日志 (由 orchestrator 注入, 记录推理指标)
        self._diag: Optional[DiagLogger] = None

        # 统计
        self._switch_count: int = 0        # 任务切换次数
        self._total_inference_steps: int = 0
        self._task_inference_steps: Dict[str, int] = {}

    # ========================================================================
    # 引用注入
    # ========================================================================

    def set_robot_refs(self, robot, camera_manager):
        """注入机器人 (关节观测) 与相机管理器 (图像帧) 引用, 供后台推理线程使用"""
        self._robot = robot
        self._camera_manager = camera_manager

    def set_diag(self, diag: Optional[DiagLogger]):
        """注入诊断日志 (由 orchestrator 创建并传入)"""
        self._diag = diag

    # ========================================================================
    # 模型加载 (只加载一次)
    # ========================================================================

    def load_model(self) -> bool:
        """
        加载唯一的 PI0.5 模型 (含 config 适配 + RTC 注入 + 预/后处理器)。

        Returns:
            True 如果加载成功
        """
        if self._policy is not None:
            logger.info("[ModelMgr] PI 模型已加载, 跳过")
            return True

        model_path = self._model_path
        if not model_path:
            logger.error("[ModelMgr] 模型路径未配置")
            return False
        if not os.path.isdir(model_path):
            logger.error(f"[ModelMgr] 模型路径不存在: {model_path}")
            return False

        logger.info(f"[ModelMgr] 加载 PI0.5 模型: {model_path}")
        t_start = time.time()

        try:
            from lerobot.policies.factory import make_pre_post_processors

            # ── 1. 加载策略 (config 适配 + 权重重映射) ──
            policy = load_pi05_policy(model_path)

            # ── 2. 注入 RTC 配置 ──
            policy.config.rtc_config = self._rtc_config
            policy.init_rtc_processor()
            rtc_status = "已启用" if policy._rtc_enabled() else "未启用"
            logger.info(
                f"[ModelMgr] RTC 配置已注入: enabled={self._rtc_config.enabled}, "
                f"execution_horizon={self._rtc_config.execution_horizon}, "
                f"max_guidance_weight={self._rtc_config.max_guidance_weight}, "
                f"schedule={self._rtc_config.prefix_attention_schedule}, "
                f"状态={rtc_status}"
            )

            # ── 3. 加载预处理器/后处理器 (tokenizer 必须覆盖为本机路径) ──
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                model_path,
                preprocessor_overrides={
                    "tokenizer_processor": {
                        "tokenizer_name": policy.config.tokenizer_name,
                    }
                },
            )

            policy.eval()
            policy.to(self._device)
            policy.reset()

            self._policy = policy
            self._preprocess_fn = preprocess
            self._postprocess_fn = postprocess
            self._policy_config = policy.config

            elapsed = time.time() - t_start
            logger.info(
                f"[ModelMgr] ✅ PI 模型加载完成 ({elapsed:.1f}s) "
                f"chunk_size={policy.config.chunk_size}, device={self._device}"
            )
            return True

        except Exception as e:
            logger.error(f"[ModelMgr] ❌ 加载 PI 模型失败: {e}")
            traceback.print_exc()
            return False

    # ========================================================================
    # 任务切换 (核心: 同模型切换 task 文本)
    # ========================================================================

    def switch_model(self, target: str, restart_thread: bool = True) -> bool:
        """
        切换任务。

        Args:
            target: "Task_A" / "Task_B" / "Task_C" (任务标签, 切换对应 task 文本)
                    "standby" (待机, 模型保留在 GPU)
            restart_thread: 切换完成后是否立即重启后台推理线程。
                           编排器可先设为 False, 在移动到任务平均初始位姿后
                           再调用 start_inference_thread() 重启, 让模型基于
                           新位姿 + 新任务文本重新预测。

        Returns:
            True 如果切换成功
        """
        # standby 模式
        if target == "standby":
            return self._enter_standby()

        # 模型未加载 → 先加载
        if self._policy is None:
            if not self.load_model():
                logger.error("[ModelMgr] ❌ 加载 PI 模型失败, 任务切换中止")
                return False
            self._switch_count += 1
            if restart_thread:
                self._start_inference_thread()
            return True

        # 已经是当前任务, 无需切换
        if target == self._active_task:
            logger.debug(f"[ModelMgr] 已是当前任务 {target}, 无需切换")
            return True

        # ── 执行任务切换: 停止线程 → 重置状态 → 更新 task 文本 → (可选)重启线程 ──
        logger.info(
            f"[ModelMgr] 🔄 任务切换: {self._active_task or 'standby'} → {target} "
            f"(task='{TASK_PROMPTS.get(target, DEFAULT_TASK_PROMPT)}')"
        )
        t_start = time.time()

        # 步骤1: 停止旧后台推理线程 (避免旧任务文本的推理写入新队列)
        self._stop_inference_thread()

        # 步骤2: 重置策略内部状态 + 新建空 ActionQueue (RTC 对齐归零)
        try:
            self._policy.reset()
        except Exception as e:
            logger.debug(f"[ModelMgr] 策略 reset 异常 (可忽略): {e}")
        self._action_queue = self._new_action_queue()

        # 步骤3: 更新任务标签与任务文本
        self._set_task_prompt(TASK_PROMPTS.get(target, DEFAULT_TASK_PROMPT))
        self._active_task = target

        # 步骤4: 重启后台推理线程 (模型已加载/预热, 跳过 warmup)
        if restart_thread:
            self._start_inference_thread()

        self._switch_count += 1
        elapsed = time.time() - t_start
        logger.info(f"[ModelMgr] ✅ 任务切换完成 ({elapsed:.2f}s), 当前任务: {target}")
        return True

    def _enter_standby(self) -> bool:
        """进入待机状态: 停止推理线程, 模型保留在 GPU (便于快速恢复)"""
        self._stop_inference_thread()

        if not self._active_task:
            logger.debug("[ModelMgr] 已在待机状态")
            return True

        logger.info(f"[ModelMgr] ⏸️ 进入待机状态 (当前任务 {self._active_task})")
        try:
            self._policy.reset()
        except Exception:
            pass

        self._active_task = ""
        self._action_queue = None

        logger.info("[ModelMgr] ✅ 待机就绪 (模型保留在 GPU, 可快速恢复)")
        return True

    def _new_action_queue(self):
        """新建 ActionQueue (RTC 模式由配置决定)"""
        from lerobot.policies.rtc.action_queue import ActionQueue
        return ActionQueue(self._rtc_config)

    # ========================================================================
    # 后台推理线程 (RTC)
    # ========================================================================

    def start_inference_thread(self):
        """启动后台推理线程 (含首次 CUDA 预热)"""
        self._start_inference_thread()

    def _start_inference_thread(self):
        """启动后台推理线程 (若未运行)"""
        if self._infer_thread is not None and self._infer_thread.is_alive():
            return
        if self._policy is None:
            logger.debug("[ModelMgr] 模型未加载, 不启动推理线程")
            return
        if self._robot is None or self._camera_manager is None:
            logger.warning("[ModelMgr] 机器人/相机引用未注入, 延迟启动推理线程")
            return

        self._shutdown_event.clear()
        self._thread_warned_dead = False

        # CUDA 预热: 首次推理需 1-2s 算子编译/显存分配, 预热避免开局跳变。
        # 模型不切换, 预热只需做一次; 任务切换重启线程时跳过。
        if not self._warmup_done:
            try:
                self._warmup()
                self._warmup_done = True
            except Exception as e:
                logger.warning(f"[ModelMgr] 预热失败 (可忽略): {e}")

        self._infer_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="PI-RTC-Inference",
        )
        self._infer_thread.start()
        logger.info(
            f"[ModelMgr] 🧵 PI 后台推理线程已启动 "
            f"(任务={self._active_task or '待机'}, RTC={'开' if self._rtc_config.enabled else '关'})"
        )

    def _stop_inference_thread(self):
        """停止后台推理线程"""
        if self._infer_thread is None:
            return
        self._shutdown_event.set()
        self._infer_thread.join(timeout=5.0)
        self._infer_thread = None
        logger.debug("[ModelMgr] PI 后台推理线程已停止")

    def _warmup(self):
        """对模型做 3 次 CUDA 预热推理"""
        if self._policy is None:
            return
        obs = self._prepare_observation()
        if obs is None:
            return
        obs["task"] = [self._get_task_prompt()]
        processed = self._preprocess_fn(obs)
        with torch.no_grad():
            for _ in range(3):
                kwargs = dict(
                    inference_delay=0,
                    execution_horizon=self._rtc_config.execution_horizon,
                )
                if self.config.pi_deterministic_noise:
                    kwargs["noise"] = make_deterministic_noise(
                        self._policy, processed["observation.state"].shape[0], self._device
                    )
                _ = self._policy.predict_action_chunk(processed, **kwargs)
        logger.info("[ModelMgr] 🔥 PI 模型 CUDA 预热完成")

    def _prepare_observation(self) -> Optional[dict]:
        """
        从机器人 + 相机管理器构建观测帧 (供后台推理线程使用)。

        与 inference_pi05_adarms_fix.py 的 prepare_observation 对齐:
            - 关节状态来自 robot.get_observation() 的 *.pos 键
            - 图像来自 camera_manager 的帧缓存 (top / wrist)
            - 图像预处理: HWC→CHW, /255, 扩 batch 维, 转 tensor 到设备
        """
        if self._robot is None or self._camera_manager is None:
            return None

        try:
            joint_obs = self._robot.get_observation()
        except Exception:
            return None

        state_values = []
        for key, value in joint_obs.items():
            if key.endswith('.pos'):
                state_values.append(np.float32(value))

        if len(state_values) < 6:
            logger.debug("[ModelMgr] 关节数据不完整, 跳过本次观测")
            return None

        top_img, wrist_img, _ = self._camera_manager.get_latest_frames()
        if top_img is None or wrist_img is None:
            return None

        frame = {
            'observation.state': np.array(state_values, dtype=np.float32),
            'observation.images.top': top_img,
            'observation.images.wrist': wrist_img,
        }

        # 图像/状态预处理 (对齐 prepare_observation)
        for name in frame:
            if "image" in name:
                obs = frame[name].astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # HWC → CHW
            else:
                obs = frame[name]
            obs = np.expand_dims(obs, axis=0)
            frame[name] = torch.tensor(obs, dtype=torch.float32, device=self._device)

        return frame

    def _inference_loop(self):
        """
        后台推理线程 — 监控 ActionQueue, 队列将耗尽时触发 PI chunk 推理。

        使用 SmartLatencyTracker + 动态阈值 (对齐 inference_pi05_adarms_fix.py):
            threshold = max(预估延迟步数 + execution_horizon/2, 基础阈值)
        """
        try:
            latency_tracker = SmartLatencyTracker(
                alpha=0.15, margin=1.1, initial_estimate=0.200,
            )
            fps = self.config.fps
            time_per_step = 1.0 / fps
            base_threshold = (
                self.config.pi_action_queue_threshold if self._rtc_config.enabled else 0
            )

            infer_count = 0          # 诊断: 推理序号
            last_steps_passed = None # 上次实际消耗步数 (inference_delay 兜底)

            while not self._shutdown_event.is_set():
                policy = self._policy
                queue = self._action_queue

                if policy is None or queue is None:
                    time.sleep(0.02)
                    continue

                # ── 动态阈值: 预估延迟步数 + 执行窗口的一半 ──
                estimated_delay_s = latency_tracker.get_estimate()
                inference_delay_steps = math.ceil(estimated_delay_s / time_per_step)
                # ★ 用上次实际 steps_passed 兜底: EMA 可能低估 (首次/抖动),
                #   而拼接点必须落在 RTC 硬对齐区 (权重=1), 若 est < real 则
                #   对齐点偏早 → chunk 边界跳变。max(EMA, 上次实际+1) 保证不低估。
                if last_steps_passed is not None:
                    inference_delay_steps = max(inference_delay_steps, last_steps_passed + 1)
                dynamic_threshold = inference_delay_steps + (self._rtc_config.execution_horizon // 2)
                threshold = max(dynamic_threshold, base_threshold)

                if queue.qsize() > threshold:
                    time.sleep(0.005)
                    continue

                # ── 步骤1: 捕获当前同步点 ──
                t_start = time.perf_counter()
                idx_at_start = queue.get_action_index()
                prev_chunk_left_over = queue.get_left_over()

                # ── 步骤2: 获取观测 + 预处理 ──
                obs = self._prepare_observation()
                if obs is None:
                    time.sleep(0.02)
                    continue

                # 当前激活任务的训练任务文本 (VLM 调度切换)
                obs["task"] = [self._get_task_prompt()]
                processed_obs = self._preprocess_fn(obs)

                # ── 步骤3: 模型推理 (RTC 引导) ──
                # 必须用 torch.no_grad() 而非 inference_mode: RTC 的 guidance
                # 需要 torch.enable_grad() 内的 autograd.grad
                with torch.no_grad():
                    _infer_kwargs = dict(
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay_steps,
                        execution_horizon=self._rtc_config.execution_horizon,
                    )
                    if self.config.pi_deterministic_noise:
                        _infer_kwargs["noise"] = make_deterministic_noise(
                            policy, processed_obs["observation.state"].shape[0], self._device
                        )
                    actions_chunk = policy.predict_action_chunk(processed_obs, **_infer_kwargs)

                # ── 步骤4: 后处理 ──
                original_actions = actions_chunk.squeeze(0).clone()
                postprocessed_chunk = self._postprocess_fn(actions_chunk)
                postprocessed_chunk = postprocessed_chunk.squeeze(0)

                # ── 步骤5: 更新延迟统计 (EMA 平滑) ──
                actual_latency = time.perf_counter() - t_start
                latency_tracker.update(actual_latency)

                # ── 步骤5.5: 诊断指标 (模型输出是否"有动作") ──
                # joint_span: chunk 内 6 关节极差最大值(rad), 判断模型输出幅度
                #   (<0.05rad≈保持不动; >0.3rad≈有明显动作)
                # gripper_span: chunk 内夹爪极差 (0~1)
                # obs_state: 归一化观测状态 (约 0 = 训练均值附近, 偏离大 = 分布外)
                post_chunk_np = postprocessed_chunk.cpu().numpy()
                joint_span = float(
                    np.abs(post_chunk_np[:, :6].max(0) - post_chunk_np[:, :6].min(0)).max()
                )
                gripper_span = float(
                    np.abs(post_chunk_np[:, 6].max() - post_chunk_np[:, 6].min())
                )
                obs_state_np = processed_obs["observation.state"].cpu().numpy().reshape(-1)[:7]

                # ── 步骤6: 计算实际消耗步数并合并 ──
                idx_at_end = queue.get_action_index()
                steps_passed = idx_at_end - idx_at_start
                last_steps_passed = steps_passed
                queue.merge(
                    original_actions,
                    postprocessed_chunk,
                    steps_passed,
                    idx_at_start,
                )

                # ── 步骤7: 诊断记录 ──
                infer_count += 1
                if self._diag is not None:
                    self._diag.log_infer(
                        infer_count, actual_latency * 1000, latency_tracker.raw_ema * 1000,
                        inference_delay_steps, steps_passed, queue.qsize(), threshold,
                        joint_span, gripper_span, obs_state_np,
                    )

                self._total_inference_steps += 1
                task = self._active_task or "standby"
                self._task_inference_steps[task] = \
                    self._task_inference_steps.get(task, 0) + 1

                logger.debug(
                    f"[ModelMgr] 推理完成 | 耗时:{actual_latency*1000:.0f}ms | "
                    f"EMA延迟:{latency_tracker.raw_ema*1000:.0f}ms | "
                    f"预估步数:{inference_delay_steps} | 实际步数:{steps_passed} | "
                    f"队列剩余:{queue.qsize()} | 阈值:{threshold} | 任务:{task}"
                )

        except Exception:
            logger.error(f"[ModelMgr] ❌ 后台推理线程致命异常:\n{traceback.format_exc()}")
            self._thread_warned_dead = True
            # 不置 shutdown_event, 让主循环通过 watchdog 感知线程死亡

    # ========================================================================
    # 推理接口
    # ========================================================================

    def infer_step(
        self,
        observation_frame: Dict[str, Any],
        task_prompt: str = "",
    ) -> Optional[np.ndarray]:
        """
        从 ActionQueue 取一个动作。

        Args:
            observation_frame: 观测帧 (兼容接口, PI 后台线程自行取观测)
            task_prompt: 任务文本 (同步给后台推理线程, 必须与训练 task 一致)

        Returns:
            numpy action (7,) 已后处理; 队列为空 (后台推理进行中) 返回 None。
            主循环在返回 None 时应保持当前姿态等待下一帧。

        Raises:
            RuntimeError: 模型未加载 / 处于待机
        """
        if self._policy is None:
            raise RuntimeError("PI 模型未加载, 无法推理")
        if not self._active_task:
            raise RuntimeError("当前处于待机状态, 无激活任务")

        # 同步任务 prompt
        if task_prompt:
            self._set_task_prompt(task_prompt)

        # Watchdog: 后台推理线程意外死亡时尝试重启
        if (self._infer_thread is None or not self._infer_thread.is_alive()):
            if not self._shutdown_event.is_set():
                if not self._thread_warned_dead:
                    logger.warning("[ModelMgr] ⚠️ 后台推理线程已停止, 尝试重启...")
                    self._thread_warned_dead = True
                self._start_inference_thread()
            return None

        if self._action_queue is None:
            return None

        action = self._action_queue.get()
        if action is None:
            return None
        return action.cpu().numpy()

    def reset_active_policy(self) -> bool:
        """
        重置模型内部状态 (清空动作队列 + 策略状态)。

        用途: VLM 连拍结束后调用, 确保 PI 从当前观测重新预测 chunk,
        避免回放连拍前基于旧观测预测的陈旧动作。

        Returns:
            True 如果重置成功
        """
        if self._policy is None or not self._active_task:
            return False

        try:
            self._policy.reset()
            # 新建空 ActionQueue: 后台线程会立即基于最新观测重新预测
            self._action_queue = self._new_action_queue()
            logger.info(f"[ModelMgr] 已重置模型内部状态 (清空动作队列, 任务={self._active_task})")
            return True
        except Exception as e:
            logger.debug(f"[ModelMgr] 重置模型内部状态失败: {e}")
            return False

    def is_new_chunk(self) -> bool:
        """
        判断下一次 infer_step 是否会开始一个新的动作 chunk (队列为空)。

        Returns:
            True 如果下一次推理会开始新 chunk
        """
        if self._policy is None or not self._active_task:
            return False
        return self._action_queue is None or self._action_queue.empty()

    # ========================================================================
    # 任务 prompt
    # ========================================================================

    def _set_task_prompt(self, prompt: str):
        with self._task_prompt_lock:
            self._active_task_prompt = prompt

    def _get_task_prompt(self) -> str:
        with self._task_prompt_lock:
            return self._active_task_prompt

    # ========================================================================
    # 预加载 / 状态查询
    # ========================================================================

    def preload_all(self) -> Dict[str, bool]:
        """加载唯一 PI 模型, 激活默认任务, 启动后台推理线程"""
        results = {"model": self.load_model()}
        if not results["model"]:
            logger.error("[ModelMgr] ❌ PI 模型加载失败")
            return results

        # 激活默认任务 (默认 Task_A)
        default_task = self.config.default_model
        if default_task not in TASK_PROMPTS:
            default_task = "Task_A"
        self._set_task_prompt(TASK_PROMPTS.get(default_task, DEFAULT_TASK_PROMPT))
        self._active_task = default_task
        self._action_queue = self._new_action_queue()
        logger.info(
            f"[ModelMgr] 默认任务: {default_task}, "
            f"task_prompt={self._get_task_prompt()!r}"
        )

        # 启动后台推理线程
        self._start_inference_thread()
        return results

    @property
    def active_model(self) -> str:
        """当前任务标签 (Task_A/B/C), 待机时为空字符串"""
        return self._active_task

    @property
    def is_standby(self) -> bool:
        return self._active_task == "" or self._active_task == "standby"

    @property
    def loaded_models(self) -> list:
        return ["pi05"] if self._policy is not None else []

    @property
    def task_prompt(self) -> str:
        return self._get_task_prompt()

    @property
    def rtc_enabled(self) -> bool:
        return self._rtc_config.enabled

    @property
    def queue_size(self) -> int:
        if self._action_queue is None:
            return 0
        return self._action_queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "active_model": self._active_task or "standby",
            "loaded_models": self.loaded_models,
            "model_path": self._model_path,
            "total_switches": self._switch_count,
            "total_inference_steps": self._total_inference_steps,
            "per_task_steps": dict(self._task_inference_steps),
            "rtc_enabled": self._rtc_config.enabled,
            "queue_size": self.queue_size,
            "gpu_allocated_mb": (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available() else 0
            ),
        }

    # ========================================================================
    # 清理
    # ========================================================================

    def cleanup(self):
        """清理模型, 停止推理线程, 释放 GPU 资源"""
        logger.info("[ModelMgr] 清理...")

        self._stop_inference_thread()
        self._shutdown_event.set()

        if self._policy is not None:
            try:
                self._policy.reset()
            except Exception:
                pass
            try:
                del self._policy
            except Exception:
                pass

        self._policy = None
        self._preprocess_fn = None
        self._postprocess_fn = None
        self._policy_config = None
        self._active_task = ""
        self._action_queue = None

        if self._device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
        logger.info("[ModelMgr] ✅ 清理完成")
