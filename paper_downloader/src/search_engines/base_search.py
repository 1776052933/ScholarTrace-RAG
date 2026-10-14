"""
base_search.py — 搜索引擎抽象基类

定义所有搜索引擎必须实现的统一接口，以及结果标准化工具。
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SearchEngine(ABC):
    """搜索引擎抽象基类。

    所有具体搜索引擎（ArxivSearch, CrossrefSearch, GoogleScholarSearch）
    必须实现 ``search()`` 和 ``get_paper_info()``。

    标准化结果格式::

        {
            "title":          str,
            "authors":        list[str],
            "year":           str | None,
            "abstract":       str | None,
            "doi":            str | None,
            "arxiv_id":       str | None,
            "pdf_url":        str | None,
            "url":            str,
            "source":         str,    # "arxiv" | "crossref" | "google_scholar"
            "citation_count": int | None,
            "journal":        str | None,
        }
    """

    # 搜索引擎名称，子类覆盖
    ENGINE_NAME: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化搜索引擎。

        Args:
            config: 配置字典。若为 None 则使用空配置。
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("%s 引擎初始化完成", self.ENGINE_NAME)

    # ── 抽象接口 ──────────────────────────────────────────────────

    @abstractmethod
    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        """搜索论文。

        Args:
            query:       搜索关键词或标题。
            max_results: 最大返回数量。
            **kwargs:    引擎特定的额外参数。

        Returns:
            标准化论文信息列表。
        """
        ...

    @abstractmethod
    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """根据标识符获取单篇论文的详细信息。

        Args:
            identifier: DOI / arXiv ID / URL。
            **kwargs:   引擎特定的额外参数。

        Returns:
            标准化论文信息，未找到返回 None。
        """
        ...

    # ── 结果标准化 ────────────────────────────────────────────────

    def normalize_results(
        self,
        raw_results: List[Dict[str, Any]],
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """将引擎原始结果转换为统一的标准化格式。

        子类应调用此方法处理搜索结果，确保所有字段存在且类型正确。

        Args:
            raw_results: 引擎返回的原始结果列表。
            source:      来源标识，默认使用 self.ENGINE_NAME。

        Returns:
            标准化后的论文信息列表。
        """
        src = source or self.ENGINE_NAME
        normalized = []
        for item in raw_results:
            try:
                norm = self._normalize_single(item, src)
                if norm and norm.get("title"):
                    normalized.append(norm)
            except Exception as exc:
                self.logger.warning("标准化单条结果失败: %s — 条目: %s", exc, str(item)[:200])
        return normalized

    @staticmethod
    def _normalize_single(raw: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """标准化单条原始结果。

        Args:
            raw:    原始结果 dict。
            source: 来源标识。

        Returns:
            标准化 dict，字段缺失时填充默认值。
        """
        # 提取/清理标题
        title = SearchEngine._extract_title(raw.get("title", ""))

        # 标准化作者列表
        authors = SearchEngine._normalize_authors(raw.get("authors", []))

        # 年份
        year = SearchEngine._extract_year(raw.get("year"))

        return {
            "title":          title,
            "authors":        authors,
            "year":           year,
            "abstract":       raw.get("abstract") or raw.get("summary") or None,
            "doi":            raw.get("doi") or None,
            "arxiv_id":       raw.get("arxiv_id") or None,
            "pdf_url":        raw.get("pdf_url") or None,
            "url":            raw.get("url") or "",
            "source":         source,
            "citation_count": raw.get("citation_count") or None,
            "journal":        raw.get("journal") or raw.get("journal_title") or None,
        }

    @staticmethod
    def _extract_title(title_raw: Any) -> str:
        """提取并清理标题文本。

        处理 CrossRef 返回的多行标题、列表格式等。
        """
        if isinstance(title_raw, list):
            title_raw = title_raw[0] if title_raw else ""
        title = str(title_raw).strip()
        # 移除 HTML 标签（某些源可能返回 HTML）
        title = re.sub(r"<[^>]+>", "", title)
        # 规范化空白
        title = re.sub(r"\s+", " ", title)
        return title

    @staticmethod
    def _normalize_authors(authors_raw: Any) -> List[str]:
        """将各种作者格式统一为字符串列表。

        支持:
            - 字符串: "John Smith; Jane Doe"
            - 字符串列表: ["John Smith", "Jane Doe"]
            - dict 列表: [{"name": "John Smith"}, ...]
            - 对象列表: 有 .name 属性的对象
        """
        if not authors_raw:
            return []

        if isinstance(authors_raw, str):
            # 按 ; 或 , 分割（注意逗号可能同时是姓名分隔符）
            if ";" in authors_raw:
                return [a.strip() for a in authors_raw.split(";") if a.strip()]
            return [authors_raw.strip()]

        if isinstance(authors_raw, list):
            result = []
            for a in authors_raw:
                if isinstance(a, str):
                    result.append(a.strip())
                elif isinstance(a, dict):
                    name = a.get("name") or a.get("given") or ""
                    family = a.get("family") or ""
                    if family:
                        name = f"{name} {family}".strip()
                    if name:
                        result.append(name)
                elif hasattr(a, "name"):
                    result.append(str(a.name).strip())
                else:
                    result.append(str(a).strip())
            return [r for r in result if r]

        return []

    @staticmethod
    def _extract_year(year_raw: Any) -> Optional[str]:
        """从各种年份格式中提取四位年份字符串。"""
        if not year_raw:
            return None
        if isinstance(year_raw, (int, float)):
            return str(int(year_raw))
        year_str = str(year_raw).strip()
        # 尝试提取四位数字
        match = re.search(r"\b(19|20)\d{2}\b", year_str)
        if match:
            return match.group(0)
        return year_str[:4] if len(year_str) >= 4 else year_str

    # ── 工具方法 ──────────────────────────────────────────────────

    @staticmethod
    def _clean_query(query: str) -> str:
        """清理搜索查询字符串。"""
        # 移除多余空白
        query = re.sub(r"\s+", " ", query.strip())
        # 移除可能干扰搜索的特殊字符（保留基本标点）
        return query

    @staticmethod
    def make_blank_result(source: str = "") -> Dict[str, Any]:
        """创建一个空白的标准化结果模板。

        Args:
            source: 来源标识。

        Returns:
            包含所有字段默认值的空白模板。
        """
        return {
            "title": "",
            "authors": [],
            "year": None,
            "abstract": None,
            "doi": None,
            "arxiv_id": None,
            "pdf_url": None,
            "url": "",
            "source": source,
            "citation_count": None,
            "journal": None,
        }
