"""
窗口投票机制 — 滑动窗口内多帧 VLM 判断投票决定最终任务

机制:
    - 维护最近 N 次 VLM 判断结果 (默认 N=5)
    - 当某个 subtask 出现次数 ≥ 阈值 (默认 3), 触发共识
    - 共识后清除窗口, 避免旧结果干扰新判断
    - 支持失败结果的优雅降级处理
"""

import logging
from collections import Counter, deque
from typing import Optional, Tuple

from lerobot.act_vlm_orchestrator_v2.config import OrchestratorConfig, SUBTASK_TO_MODEL
from lerobot.act_vlm_orchestrator_v2.vlm_client import VLMDecision

logger = logging.getLogger(__name__)


class WindowVoter:
    """
    滑动窗口投票器

    工作流程:
        1. add_decision(): 添加 VLM 判断结果到窗口
        2. 窗口满后, 检查是否有分类获得 ≥ threshold 票数
        3. 如有共识 → get_consensus() 返回共识结果
        4. 共识后自动清空窗口, 开始新一轮投票
    """

    def __init__(self, config: OrchestratorConfig):
        self._window_size = config.vote_window_size
        self._threshold = config.vote_consensus_threshold

        # 滑动窗口: 存储 (subtask, target_model) 元组
        self._window: deque = deque(maxlen=self._window_size)

        # 共识状态
        self._consensus_reached: bool = False
        self._consensus_subtask: str = ""
        self._consensus_model: str = ""
        self._consensus_count: int = 0
        self._consensus_ratio: float = 0.0

        # 上一次共识结果 (用于检测变化)
        self._last_consensus_subtask: str = ""

        # 统计
        self._total_decisions: int = 0
        self._failed_decisions: int = 0
        self._consensus_count_total: int = 0

    # ========================================================================
    # 核心接口
    # ========================================================================

    def add_decision(self, decision: VLMDecision) -> bool:
        """
        添加一次 VLM 判断到投票窗口。

        Args:
            decision: VLM 单次场景分类结果

        Returns:
            True 如果本次添加后达成共识, False 否则
        """
        self._total_decisions += 1

        if not decision.success:
            self._failed_decisions += 1
            # 失败结果不参与投票, 但也不清空窗口
            logger.debug(f"[Voter] VLM 调用失败, 跳过投票 "
                         f"(窗口: {len(self._window)}/{self._window_size})")
            return False

        # 添加有效结果到窗口
        self._window.append((decision.subtask, decision.target_model))
        logger.debug(
            f"[Voter] 添加: subtask={decision.subtask}, "
            f"model={decision.target_model}, "
            f"窗口: {len(self._window)}/{self._window_size}"
        )

        # 检查是否达成共识
        return self._check_consensus()

    def _check_consensus(self) -> bool:
        """检查窗口内是否达成共识"""
        if len(self._window) < self._window_size:
            return False

        # 统计窗口内各 subtask 出现次数
        subtask_counts = Counter(s for s, _ in self._window)
        most_common = subtask_counts.most_common(1)

        if not most_common:
            return False

        winner_subtask, winner_count = most_common[0]

        if winner_count >= self._threshold:
            # 达成共识!
            self._consensus_reached = True
            self._consensus_subtask = winner_subtask
            self._consensus_model = SUBTASK_TO_MODEL.get(winner_subtask, "standby")
            self._consensus_count = winner_count
            self._consensus_ratio = winner_count / self._window_size
            self._consensus_count_total += 1

            logger.info(
                f"[Voter] ✅ 达成共识! subtask={winner_subtask} "
                f"→ model={self._consensus_model} "
                f"({winner_count}/{self._window_size} 票, "
                f"比率={self._consensus_ratio:.0%})"
            )

            # 记录窗口详情用于调试
            window_detail = Counter(s for s, _ in self._window)
            logger.debug(f"[Voter] 窗口详情: {dict(window_detail)}")

            return True

        # 未达成共识
        logger.debug(
            f"[Voter] 未达成共识: 最高票={winner_subtask} "
            f"({winner_count}/{self._window_size}), "
            f"需要 ≥{self._threshold}"
        )
        return False

    # ========================================================================
    # 共识结果获取
    # ========================================================================

    def get_consensus(self) -> Optional[Tuple[str, str]]:
        """
        获取并消费共识结果。

        Returns:
            (subtask, model_name) 如果共识已达成, None 否则
            返回后自动清空窗口, 准备下一轮投票
        """
        if not self._consensus_reached:
            return None

        result = (self._consensus_subtask, self._consensus_model)
        self._last_consensus_subtask = self._consensus_subtask

        # 清空窗口和共识状态, 准备下一轮
        self._window.clear()
        self._consensus_reached = False
        self._consensus_subtask = ""
        self._consensus_model = ""
        self._consensus_count = 0
        self._consensus_ratio = 0.0

        logger.info(f"[Voter] 共识已消费: {result}, 窗口已重置")
        return result

    def peek_consensus(self) -> Optional[Tuple[str, str, int]]:
        """
        查看共识结果但不消费 (不清空窗口)。

        Returns:
            (subtask, model_name, count) 如果共识已达成, None 否则
        """
        if not self._consensus_reached:
            return None
        return (self._consensus_subtask, self._consensus_model, self._consensus_count)

    def is_window_full(self) -> bool:
        """检查窗口是否已填满"""
        return len(self._window) >= self._window_size

    def has_consensus(self) -> bool:
        """检查是否有未消费的共识结果"""
        return self._consensus_reached

    # ========================================================================
    # 强制操作
    # ========================================================================

    def force_consensus(self, subtask: str) -> Tuple[str, str]:
        """
        强制设置共识结果 (用于静止检测触发等场景)。

        Args:
            subtask: 强制设置的 subtask

        Returns:
            (subtask, model_name)
        """
        model = SUBTASK_TO_MODEL.get(subtask, "standby")
        logger.info(f"[Voter] 🔧 强制共识: subtask={subtask} → model={model}")

        self._window.clear()
        self._consensus_reached = True
        self._consensus_subtask = subtask
        self._consensus_model = model
        self._consensus_count = self._window_size  # 虚拟全票
        self._consensus_ratio = 1.0

        return self.get_consensus()

    def reset(self):
        """完全重置投票器状态"""
        self._window.clear()
        self._consensus_reached = False
        self._consensus_subtask = ""
        self._consensus_model = ""
        self._consensus_count = 0
        self._consensus_ratio = 0.0
        self._last_consensus_subtask = ""
        logger.info("[Voter] 投票器已完全重置")

    # ========================================================================
    # 状态查询
    # ========================================================================

    @property
    def window_status(self) -> dict:
        """获取当前窗口状态"""
        if not self._window:
            return {"size": 0, "max": self._window_size, "distribution": {}}
        subtask_counts = Counter(s for s, _ in self._window)
        return {
            "size": len(self._window),
            "max": self._window_size,
            "distribution": dict(subtask_counts),
            "most_common": subtask_counts.most_common(1)[0] if subtask_counts else None,
        }

    @property
    def stats(self) -> dict:
        return {
            "total_decisions": self._total_decisions,
            "failed_decisions": self._failed_decisions,
            "success_rate": (self._total_decisions - self._failed_decisions)
                            / max(self._total_decisions, 1),
            "total_consensuses": self._consensus_count_total,
            "last_consensus": self._last_consensus_subtask,
        }
