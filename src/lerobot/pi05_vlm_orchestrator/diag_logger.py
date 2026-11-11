"""
推理诊断日志 (DiagLogger) — 对齐 inference_pi05_adarms_fix.py

把推理线程与主循环的关键指标写入 CSV + 终端，便于事后分析。

【用途】定位三个典型问题：
  1. 模型"无动作/幅度小" → chunk_joint_span(rad) 很小 + 归一化观测状态偏离训练均值
  2. "chunk 衔接跳变"     → 主循环 step_jump_deg 在某个时刻突然 >2°（大概率在 chunk 边界）
  3. "流匹配抖动"         → step_jump_deg 频繁超阈值 + 相邻推理 chunk_joint_span 波动大
"""

import os
import time
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class DiagLogger:
    """推理诊断日志: 推理指标 (infer) + 执行指标 (exec) 写入 CSV。"""

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
            logger.info(f"[Diag] 📈 诊断日志已开启 → {self.infer_path} / {self.exec_path}")

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
        try:
            with self.lock:
                with open(self.infer_path, "a") as f:
                    obs = ",".join(f"{v:.4f}" for v in obs_state)
                    f.write(
                        f"{idx},{latency_ms:.1f},{ema_ms:.1f},{est_steps},{real_steps},{qsize},{threshold},"
                        f"{joint_span_rad:.5f},{gripper_span:.4f},{obs}\n"
                    )
        except Exception as e:
            logger.debug(f"[Diag] 写 infer CSV 失败: {e}")

    def log_exec(self, step: int, t_s: float, joint_deg, gripper: int, step_jump_deg: float):
        if not self.enabled:
            return
        try:
            with self.lock:
                with open(self.exec_path, "a") as f:
                    j = ",".join(f"{v:.2f}" for v in joint_deg)
                    f.write(f"{step},{t_s:.3f},{j},{gripper},{step_jump_deg:.3f}\n")
        except Exception as e:
            logger.debug(f"[Diag] 写 exec CSV 失败: {e}")
