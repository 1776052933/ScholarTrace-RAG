"""查询向量与论文向量的余弦语义相似度计算。"""

import math
from numbers import Real
from typing import Iterable, List, Sequence


Vector = Sequence[float]


class SimilarityCalculator:
    """计算范围为 0 到 1 的 query-paper 语义相似度。"""

    def calculate(self, query_vector: Vector, paper_vector: Vector) -> float:
        """计算一组查询向量和论文向量的余弦相似度。

        余弦值小于 0 表示向量方向负相关，在论文初筛中按 0 分处理。
        零向量无法定义方向，同样返回 0 分。
        """
        query = _validate_vector(query_vector, "query_vector")
        paper = _validate_vector(paper_vector, "paper_vector")
        return _cosine_score(query, paper)

    def calculate_batch(
        self,
        query_vector: Vector,
        paper_vectors: Iterable[Vector],
    ) -> List[float]:
        """用一个查询向量批量计算多篇论文的相似度。"""
        query = _validate_vector(query_vector, "query_vector")
        return [
            _cosine_score(query, _validate_vector(vector, f"paper_vectors[{index}]"))
            for index, vector in enumerate(paper_vectors)
        ]

    def batch_calculate(
        self,
        query_vector: Vector,
        paper_vectors: Iterable[Vector],
    ) -> List[float]:
        """``calculate_batch`` 的兼容别名。"""
        return self.calculate_batch(query_vector, paper_vectors)


def _cosine_score(query: List[float], paper: List[float]) -> float:
    """对已经校验的向量计算余弦分数。"""
    if len(query) != len(paper):
        raise ValueError(
            "query_vector and paper_vector must have the same dimension "
            f"({len(query)} != {len(paper)})"
        )

    query_norm = math.sqrt(math.fsum(value * value for value in query))
    paper_norm = math.sqrt(math.fsum(value * value for value in paper))
    if query_norm == 0.0 or paper_norm == 0.0:
        return 0.0

    dot_product = math.fsum(left * right for left, right in zip(query, paper))
    cosine = dot_product / (query_norm * paper_norm)
    # 浮点误差可能产生 1.0000000002 之类的结果，因此显式限定范围。
    return float(max(0.0, min(1.0, cosine)))


def _validate_vector(vector: Vector, name: str) -> List[float]:
    if isinstance(vector, (str, bytes)):
        raise TypeError(f"{name} must be a numeric vector")
    try:
        values = list(vector)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable numeric vector") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")

    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain only real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        result.append(number)
    return result
