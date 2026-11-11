"""
ACT + VLM 多任务联调编排器 (ACT-VLM Orchestrator)
=================================================

架构:
    VLM (Qwen3-VL-2B, vLLM) → 场景分类 → 窗口投票 → ACT 模型切换 → 机械臂执行

四个场景分类:
    subtask1 (包裹在桌面)      → Task_A 模型
    subtask2 (包裹在托盘,标签不可见) → Task_B 模型
    subtask3 (包裹在托盘,标签可见)   → Task_C 模型
    subtask4 (包裹在盒子)      → 待机等待

核心机制:
    1. 窗口投票: 5 次 VLM 判断中 ≥3 次一致 → 采用该结果
    2. 静止检测: 机械臂接近静止 3s → 触发场景重判
    3. 模型切换: 切换 ACT 模型时清除 GPU 缓存 + 重新加载,消除累积误差
"""

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig
from lerobot.act_vlm_orchestrator_v2.vlm_client import VLMClient, VLMDecision
from lerobot.act_vlm_orchestrator_v2.window_voter import WindowVoter
from lerobot.act_vlm_orchestrator_v2.act_model_manager import ACTModelManager
from lerobot.act_vlm_orchestrator_v2.stationary_detector import StationaryDetector
from lerobot.act_vlm_orchestrator_v2.camera_manager import CameraManager
from lerobot.act_vlm_orchestrator_v2.orchestrator import ACTVLMOrchestrator

__all__ = [
    "OrchestratorConfig",
    "VLMClient", "VLMDecision",
    "WindowVoter",
    "ACTModelManager",
    "StationaryDetector",
    "CameraManager",
    "ACTVLMOrchestrator",
]
