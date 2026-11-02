"""从论文元数据提取年份、引用、摘要和来源质量特征。"""

from datetime import datetime
import math
import re
from typing import Dict, Optional

from .schema import Paper


class PaperFeatureExtractor:
    """将 Paper 元数据转换为范围为 0 到 1 的精排特征。"""

    def __init__(
        self,
        *,
        current_year: Optional[int] = None,
        year_decay: float = 5.0,
        citation_reference: int = 1000,
        complete_abstract_chars: int = 400,
    ) -> None:
        self.current_year = current_year or datetime.now().year
        self.year_decay = float(year_decay)
        self.citation_reference = citation_reference
        self.complete_abstract_chars = complete_abstract_chars
        if self.current_year < 1900:
            raise ValueError("current_year must be at least 1900")
        if self.year_decay <= 0:
            raise ValueError("year_decay must be greater than zero")
        if self.citation_reference < 1:
            raise ValueError("citation_reference must be a positive integer")
        if self.complete_abstract_chars < 1:
            raise ValueError("complete_abstract_chars must be a positive integer")

    def extract(self, paper: Paper) -> Dict[str, float]:
        """提取单篇论文的四项基础质量特征。"""
        if not isinstance(paper, Paper):
            raise TypeError("paper must be an instance of ranking_agent.schema.Paper")
        return {
            "year_score": self.year_score(paper.year),
            "citation_score": self.citation_score(paper.citation_count),
            "abstract_score": self.abstract_score(paper.abstract),
            "venue_score": self.venue_score(paper.venue),
        }

    def year_score(self, year: Optional[int]) -> float:
        """论文越接近当前年份得分越高，未知年份得 0 分。"""
        if year is None or year < 1900:
            return 0.0
        age = max(0, self.current_year - year)
        return _round(math.exp(-age / self.year_decay))

    def citation_score(self, citation_count: int) -> float:
        """使用对数归一化引用数，降低极高引用次数的边际影响。"""
        citations = max(0, citation_count)
        score = math.log1p(citations) / math.log1p(self.citation_reference)
        return _round(min(1.0, score))

    def abstract_score(self, abstract: str) -> float:
        """根据去除空白后的摘要长度评价摘要完整度。"""
        if not abstract or not abstract.strip():
            return 0.0
        effective_length = len(re.sub(r"\s+", "", abstract))
        return _round(min(1.0, effective_length / self.complete_abstract_chars))

    def venue_score(self, venue: str) -> float:
        """根据 venue 是否为高质量正式出版来源进行分层评分。"""
        normalized = re.sub(r"[^a-z0-9]+", " ", (venue or "").casefold()).strip()
        if not normalized:
            return 0.0
        if _matches_top_venue(normalized):
            return 1.0
        if any(marker in normalized for marker in _FORMAL_VENUE_MARKERS):
            return 0.8
        if any(marker in normalized for marker in _PREPRINT_MARKERS):
            return 0.5
        return 0.65


_TOP_VENUES = {
    "aaai", "acl", "cvpr", "eccv", "emnlp", "iccv", "iclr", "icml",
    "ijcai", "jmlr", "kdd", "miccai", "naacl", "neurips", "nips",
    "sigir", "tpami",
}

_TOP_VENUE_PHRASES = {
    "medical image analysis",
    "nature medicine",
    "nature machine intelligence",
    "transactions on pattern analysis and machine intelligence",
}

_FORMAL_VENUE_MARKERS = (
    "conference", "journal", "proceedings", "transactions", "symposium",
    "workshop", "association", "ieee", "acm", "springer", "elsevier",
)

_PREPRINT_MARKERS = ("arxiv", "biorxiv", "medrxiv", "preprint")


def _matches_top_venue(normalized: str) -> bool:
    tokens = set(normalized.split())
    return bool(tokens & _TOP_VENUES) or any(phrase in normalized for phrase in _TOP_VENUE_PHRASES)


def _round(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)
