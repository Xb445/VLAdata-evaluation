"""
PI0.5 + VLM 多任务联调编排器 (PI-VLM Orchestrator)
===================================================

架构:
    VLM (Qwen3-VL, vLLM) → 场景分类 → 窗口投票 → PI0.5 模型切换 → 机械臂执行

四个场景分类:
    subtask1 (包裹在桌面)          → Task_A  (抓取到条码扫描台)
    subtask2 (包裹在托盘,条码不可见) → Task_B  (翻转包裹)
    subtask3 (包裹在托盘,条码可见)   → Task_C  (放入盒子)
    subtask4 (包裹在盒子)          → 待机等待

每个 PI 模型绑定独立的训练任务文本 (来自 batch5A/B/C_joint_merged 数据集):
    Task_A: "Grab the package and place it on the barcode scanning table."
    Task_B: "Flip the package if the barcode is not facing up."
    Task_C: "Grab the scanned package and place it into the box."

核心机制:
    1. 窗口投票: 5 次 VLM 判断中 ≥3 次一致 → 采用该结果
    2. 静止检测: 机械臂接近静止 2~3s → 触发场景重判
    3. 模型切换: 停止推理线程 + 清 GPU 缓存 + 切换模型与任务文本
    4. PI 推理: 后台线程 + ActionQueue + RTC (对齐 inference_pi05_adarms_fix.py)
"""

from lerobot.pi05_vlm_orchestrator.config import (
    OrchestratorConfig,
    TASK_PROMPTS,
    SUBTASK_TO_MODEL,
    SUBTASK_DESCRIPTIONS,
    DEFAULT_MODEL_PATH,
    LOCAL_TOKENIZER,
)
from lerobot.pi05_vlm_orchestrator.vlm_client import VLMClient, VLMDecision
from lerobot.pi05_vlm_orchestrator.window_voter import WindowVoter
from lerobot.pi05_vlm_orchestrator.pi_model_manager import PIModelManager
from lerobot.pi05_vlm_orchestrator.diag_logger import DiagLogger
from lerobot.pi05_vlm_orchestrator.stationary_detector import StationaryDetector
from lerobot.pi05_vlm_orchestrator.camera_manager import CameraManager
from lerobot.pi05_vlm_orchestrator.orchestrator import PIVLMOrchestrator

__all__ = [
    "OrchestratorConfig",
    "TASK_PROMPTS",
    "SUBTASK_TO_MODEL",
    "SUBTASK_DESCRIPTIONS",
    "DEFAULT_MODEL_PATH",
    "LOCAL_TOKENIZER",
    "VLMClient", "VLMDecision",
    "WindowVoter",
    "PIModelManager",
    "DiagLogger",
    "StationaryDetector",
    "CameraManager",
    "PIVLMOrchestrator",
]
