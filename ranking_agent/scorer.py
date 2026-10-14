"""计算论文多维特征的加权综合评分。"""

import math
from numbers import Real
from typing import Mapping, Optional

from .config import RankingConfig
from .schema import PaperScore, RelevanceLevel


class RankingScorer:
    """根据固定权重生成 PaperScore 和相关等级。"""

    def __init__(self, config: Optional[RankingConfig] = None) -> None:
        self.config = config or RankingConfig()

    def calculate_score(self, features: Mapping[str, float]) -> PaperScore:
        """校验六项评分、计算 Final Score，并划分相关等级。"""
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping")

        values = {
            name: _feature_value(features, name)
            for name in self.config.weights
        }
        final_score = math.fsum(
            values[name] * weight
            for name, weight in self.config.weights.items()
        )
        # 消除浮点累计误差，并保持 JSON 输出简洁稳定。
        final_score = round(max(0.0, min(1.0, final_score)), 6)

        return PaperScore(
            **values,
            final_score=final_score,
            relevance_level=self._relevance_level(final_score),
        )

    def _relevance_level(self, final_score: float) -> RelevanceLevel:
        if final_score >= self.config.highly_relevant_threshold:
            return "highly_relevant"
        if final_score >= self.config.relevant_threshold:
            return "relevant"
        return "partially_relevant"


def _feature_value(features: Mapping[str, float], name: str) -> float:
    if name not in features:
        raise ValueError(f"missing ranking feature: {name}")
    value = features[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"ranking feature {name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"ranking feature {name} must be between 0 and 1")
    return result
