"""Ranking Agent 的评分权重和相关等级配置。"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RankingConfig:
    """论文综合评分配置，所有权重之和必须等于 1。"""

    semantic_weight: float = 0.35
    method_weight: float = 0.20
    task_weight: float = 0.15
    quality_weight: float = 0.15
    time_weight: float = 0.05
    llm_weight: float = 0.10

    highly_relevant_threshold: float = 0.80
    relevant_threshold: float = 0.60

    def __post_init__(self) -> None:
        weights = self.weights
        if any(not 0.0 <= value <= 1.0 for value in weights.values()):
            raise ValueError("ranking weights must be between 0 and 1")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("ranking weights must sum to 1.0")
        if not 0.0 <= self.relevant_threshold <= self.highly_relevant_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= relevant <= highly_relevant <= 1"
            )

    @property
    def weights(self) -> dict[str, float]:
        return {
            "semantic_score": self.semantic_weight,
            "method_score": self.method_weight,
            "task_score": self.task_weight,
            "quality_score": self.quality_weight,
            "time_score": self.time_weight,
            "llm_score": self.llm_weight,
        }


# 名称兼容：后续 Pipeline 可使用更完整的 Agent 配置名称。
RankingAgentConfig = RankingConfig
DEFAULT_RANKING_CONFIG = RankingConfig()
