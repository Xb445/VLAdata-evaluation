#!/usr/bin/env python3
"""
RTCInspiredSmoother — 借鉴 Real-Time Chunking (RTC) 思想的动作平滑器
=====================================================================

问题背景
--------
SmolVLA 的 `select_action` 每推理一次生成 50 帧动作 chunk 并逐帧执行，
每 50 帧用「新随机噪声 + 新观测」重新推理。由于 flow-matching 采样噪声
与观测变化，新 chunk 与旧 chunk 在边界处不连续，造成机械臂抖动。

解决方案（借鉴 RTC 的重叠时间步 + prefix 修正思想）
---------------------------------------------------
  1. 提前推理 (look-ahead inference)：
     当前 chunk 剩余执行帧数 ≤ execution_horizon 时，在后台线程用最新
     观测推理下一个 chunk，用执行时间掩盖推理延迟（推理约 0.3~0.5s，
     提前 15 帧 @20fps ≈ 0.75s，足够）。
  2. 重叠融合 (prefix blending)：
     新 chunk 的前 blend_steps 帧与旧 chunk 的「未执行残留帧」做加权融合
     （权重从旧到新线性/余弦过渡），等价于 RTC 的 prefix guidance，
     保证 chunk 边界连续无跳变。
  3. 滤波 (filtering)：
     融合后的动作再经指数平滑 (EMA) 抑制高频抖动。

用法
----
  smoother = RTCInspiredSmoother(model=inference, execution_horizon=15,
                                 blend_steps=10, alpha=0.3)
  action = smoother.step(wrist_img, top_img, joint_state, task)  # 每帧调用
  smoother.reset()   # 切换任务 / 重置时调用

依赖: numpy, threading（model 需提供 `predict_action_chunk(wrist,top,state,task)`）
"""

import threading

import numpy as np


def _blend_weights(n: int, schedule: str = "linear") -> np.ndarray:
    """生成 0→1 的融合权重序列（n 帧）。

    Args:
        n: 帧数
        schedule: "linear"（线性）/ "cosine"（余弦，更平滑的起止过渡）

    Returns:
        shape (n,) 的权重数组，从 0 平滑升到 1
    """
    if n <= 1:
        return np.array([1.0], dtype=np.float32)
    if schedule == "cosine":
        return (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)
    return np.linspace(0.0, 1.0, n).astype(np.float32)


class RTCInspiredSmoother:
    """RTC 式 chunk 平滑器：后台预推理 + 重叠融合 + 指数滤波。

    Args:
        model: SmolVLAInference 实例（提供 `predict_action_chunk`）
        execution_horizon: 提前推理触发阈值（帧）。当前 chunk 剩余帧数
            不超过该值时，后台线程开始推理下一个 chunk。
        blend_steps: 边界融合帧数。新 chunk 的前 blend_steps 帧与旧 chunk
            残留做加权融合（应 ≤ execution_horizon）。
        alpha: 指数平滑系数（0~1），越小越平滑。
        action_dim: 动作维度。
        blend_schedule: 融合权重曲线，"linear" 或 "cosine"。
    """

    def __init__(
        self,
        model,
        execution_horizon: int = 15,
        blend_steps: int = 10,
        alpha: float = 0.3,
        action_dim: int = 7,
        blend_schedule: str = "linear",
    ):
        self.model = model
        self.execution_horizon = max(1, int(execution_horizon))
        self.blend_steps = max(1, min(int(blend_steps), self.execution_horizon))
        self.alpha = float(alpha)
        self.action_dim = int(action_dim)
        self.blend_schedule = blend_schedule

        # 当前执行的 chunk 与位置
        self._chunk: np.ndarray | None = None
        self._pos: int = 0
        # 后台推理结果
        self._next_chunk: np.ndarray | None = None
        self._next_ready: bool = False
        self._inferring: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # 滤波状态
        self._last_action: np.ndarray | None = None
        self._chunk_count: int = 0

    # ------------------------------------------------------------------
    #  公共接口
    # ------------------------------------------------------------------

    def reset(self):
        """重置状态（切换任务 / 新 episode 时调用）。会等待后台推理线程结束。"""
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            self._chunk = None
            self._pos = 0
            self._next_chunk = None
            self._next_ready = False
            self._inferring = False
            self._last_action = None
            self._chunk_count = 0
        self._thread = None

    def step(
        self,
        wrist_img: np.ndarray,
        top_img: np.ndarray,
        joint_state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """每帧调用一次，返回平滑后的动作 (action_dim,)。

        Args:
            wrist_img / top_img: RGB 图像 (H, W, 3) uint8
            joint_state: 关节状态 (action_dim,)
            task: 任务描述

        Returns:
            平滑后的动作 (action_dim,)
        """
        # 1) 需要提前推理下一个 chunk？
        if self._chunk is None or self._pos >= len(self._chunk) - self.execution_horizon:
            self._start_inference(wrist_img, top_img, joint_state, task)

        # 2) 新 chunk 就绪且当前 chunk 已到融合边界（或尚无 chunk）→ 切换
        if self._next_ready and (
            self._chunk is None or self._pos >= len(self._chunk) - self.blend_steps
        ):
            self._apply_next_chunk()

        # 3) 取当前帧
        if self._chunk is None or self._pos >= len(self._chunk):
            # 尚无可用 chunk（首次启动 / 刚切换任务）：等待后台推理完成，
            # 保证返回的是真实推理结果而非零向量
            if self._inferring:
                t = self._thread
                if t is not None and t.is_alive():
                    t.join(timeout=3.0)
            if self._next_ready:
                self._apply_next_chunk()
            if self._chunk is not None and self._pos < len(self._chunk):
                action = self._chunk[self._pos].copy()
                self._pos += 1
            else:
                # 推理仍未完成：保持上一动作，避免跳变
                if self._last_action is not None:
                    return self._last_action.copy()
                return np.zeros(self.action_dim, dtype=np.float32)
        else:
            action = self._chunk[self._pos].copy()
            self._pos += 1

        # 4) 指数平滑（滤波）
        if self._last_action is not None:
            action = self.alpha * action + (1.0 - self.alpha) * self._last_action
        self._last_action = action.copy()
        return action

    # ------------------------------------------------------------------
    #  内部实现
    # ------------------------------------------------------------------

    def _start_inference(self, wrist_img, top_img, joint_state, task):
        """在后台线程启动一次 chunk 推理（不阻塞主循环）。"""
        with self._lock:
            if self._inferring:
                return
            self._inferring = True
        self._thread = threading.Thread(
            target=self._infer,
            args=(wrist_img.copy(), top_img.copy(), joint_state.copy(), task),
            daemon=True,
        )
        self._thread.start()

    def _infer(self, wrist_img, top_img, joint_state, task):
        """后台推理线程体。"""
        try:
            chunk = self.model.predict_action_chunk(wrist_img, top_img, joint_state, task)
            with self._lock:
                self._next_chunk = np.asarray(chunk, dtype=np.float32)
                self._next_ready = True
                self._inferring = False
        except Exception as e:  # noqa: BLE001
            print(f"[RTC] 后台推理失败: {e}", flush=True)
            with self._lock:
                self._inferring = False

    def _apply_next_chunk(self):
        """切换到新 chunk，并对边界做重叠融合（类似 RTC prefix guidance）。"""
        with self._lock:
            new = self._next_chunk
            self._next_chunk = None
            self._next_ready = False
        if new is None:
            return

        old = self._chunk
        if old is not None:
            # 旧 chunk 的未执行残留（prev_chunk_left_over）
            n_remain = len(old) - self._pos
            m = min(n_remain, self.blend_steps, len(new))
            if m > 0:
                weights = _blend_weights(m, self.blend_schedule)
                old_remain = old[self._pos : self._pos + m]
                # 加权融合：w=0 时完全用旧残留（保证连续），w→1 过渡到新 chunk
                for i in range(m):
                    new[i] = weights[i] * new[i] + (1.0 - weights[i]) * old_remain[i]

        self._chunk = new
        self._pos = 0
        self._chunk_count += 1

    # ------------------------------------------------------------------
    #  工具
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """返回当前状态（调试用）。"""
        with self._lock:
            return {
                "chunk_count": self._chunk_count,
                "pos": self._pos,
                "chunk_len": len(self._chunk) if self._chunk is not None else 0,
                "inferring": self._inferring,
                "next_ready": self._next_ready,
                "execution_horizon": self.execution_horizon,
                "blend_steps": self.blend_steps,
                "alpha": self.alpha,
            }
