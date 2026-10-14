"""根据精排分数和引用情况判断论文重要等级。"""

import math
from typing import Any, Mapping, Optional, Protocol, Union

from .schema import PaperImportance, PaperMetadata


class ImportanceLLMClassifier(Protocol):
    """未来 LLM 重要性分类器需要实现的最小接口。"""

    def classify_importance(self, paper_info: Mapping[str, Any]) -> Union[PaperImportance, str]: ...


class PaperImportanceClassifier:
    """使用明确规则分类论文，并预留可注入的 LLM 分类接口。"""

    def __init__(
        self,
        *,
        core_rank_threshold: float = 0.90,
        important_rank_threshold: float = 0.75,
        high_citation_threshold: int = 100,
        highly_relevant_threshold: float = 0.80,
        llm_classifier: Optional[ImportanceLLMClassifier] = None,
    ) -> None:
        if not 0.0 <= important_rank_threshold <= core_rank_threshold <= 1.0:
            raise ValueError("rank thresholds must satisfy 0 <= important <= core <= 1")
        if not 0.0 <= highly_relevant_threshold <= 1.0:
            raise ValueError("highly_relevant_threshold must be between 0 and 1")
        if high_citation_threshold < 0:
            raise ValueError("high_citation_threshold must not be negative")
        self.core_rank_threshold = core_rank_threshold
        self.important_rank_threshold = important_rank_threshold
        self.high_citation_threshold = high_citation_threshold
        self.highly_relevant_threshold = highly_relevant_threshold
        self.llm_classifier = llm_classifier

    def classify(
        self,
        citation_count: int,
        rank_score: float,
        venue: str = "",
        year: Optional[int] = None,
        *,
        use_llm: bool = False,
    ) -> PaperImportance:
        """返回 CORE、IMPORTANT 或 NORMAL。"""
        info = _paper_info(citation_count, rank_score, venue, year)
        if use_llm and self.llm_classifier is not None:
            return PaperImportance(self.llm_classifier.classify_importance(info))
        return self.classify_by_rules(
            citation_count=info["citation_count"],
            rank_score=info["rank_score"],
        )

    def classify_paper(
        self,
        paper: PaperMetadata,
        *,
        use_llm: bool = False,
    ) -> PaperImportance:
        """直接分类 PaperMetadata 的便捷接口。"""
        if not isinstance(paper, PaperMetadata):
            raise TypeError("paper must be a PaperMetadata")
        return self.classify(
            citation_count=paper.citation_count,
            rank_score=paper.rank_score,
            venue=paper.venue,
            year=paper.year,
            use_llm=use_llm,
        )

    def classify_by_rules(self, citation_count: int, rank_score: float) -> PaperImportance:
        """执行不依赖外部服务的论文重要性分类规则。"""
        _validate_numbers(citation_count, rank_score)
        if rank_score > self.core_rank_threshold:
            return PaperImportance.CORE
        if (
            citation_count >= self.high_citation_threshold
            and rank_score >= self.highly_relevant_threshold
        ):
            return PaperImportance.CORE
        if rank_score > self.important_rank_threshold:
            return PaperImportance.IMPORTANT
        return PaperImportance.NORMAL


def _paper_info(
    citation_count: int,
    rank_score: float,
    venue: str,
    year: Optional[int],
) -> dict[str, Any]:
    _validate_numbers(citation_count, rank_score)
    if not isinstance(venue, str):
        raise TypeError("venue must be a string")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        raise TypeError("year must be an integer or None")
    return {
        "citation_count": citation_count,
        "rank_score": float(rank_score),
        "venue": venue.strip(),
        "year": year,
    }


def _validate_numbers(citation_count: int, rank_score: float) -> None:
    if isinstance(citation_count, bool) or not isinstance(citation_count, int):
        raise TypeError("citation_count must be an integer")
    if citation_count < 0:
        raise ValueError("citation_count must not be negative")
    if isinstance(rank_score, bool) or not isinstance(rank_score, (int, float)):
        raise TypeError("rank_score must be a real number")
    if not math.isfinite(float(rank_score)) or not 0.0 <= float(rank_score) <= 1.0:
        raise ValueError("rank_score must be between 0 and 1")
