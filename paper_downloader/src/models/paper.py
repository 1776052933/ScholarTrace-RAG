"""
paper.py — 论文数据模型

提供标准化的论文信息数据结构，支持序列化与反序列化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    """标准化学术论文数据模型。

    所有字段均可选，提供 ``from_dict()`` / ``to_dict()`` / ``to_json()``
    用于跨模块传递和持久化。

    Usage::

        >>> p = Paper(title="Attention Is All You Need", year="2017")
        >>> p.to_dict()
        {'title': 'Attention Is All You Need', 'year': '2017', ...}
        >>> Paper.from_dict({"title": "GPT-4", "doi": "10.xxx"})
        Paper(title='GPT-4', ...)
    """

    # ── 核心元数据 ──────────────────────────────────────────────

    title: str = ""
    """论文标题。"""

    authors: List[str] = field(default_factory=list)
    """作者列表，每个元素为完整姓名。"""

    year: Optional[str] = None
    """发表年份（四位数字字符串）。"""

    abstract: Optional[str] = None
    """摘要文本。"""

    doi: Optional[str] = None
    """DOI 标识符，如 10.1038/nature14539。"""

    arxiv_id: Optional[str] = None
    """arXiv ID，如 2401.00001v2。"""

    # ── URL ────────────────────────────────────────────────────

    pdf_url: Optional[str] = None
    """PDF 在线地址。"""

    url: Optional[str] = None
    """论文主页 URL。"""

    # ── 来源 ───────────────────────────────────────────────────

    source: Optional[str] = None
    """数据来源，如 arxiv / crossref / google_scholar。"""

    citation_count: Optional[int] = None
    """被引次数。"""

    journal: Optional[str] = None
    """期刊/会议名称。"""

    # ── 本地状态 ───────────────────────────────────────────────

    pdf_path: Optional[str] = None
    """本地 PDF 文件路径（下载后填充）。"""

    file_size: int = 0
    """本地 PDF 文件大小（字节）。"""

    downloaded_at: Optional[str] = None
    """下载时间（ISO 8601 格式）。"""

    # ── 构造方法 ──────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """从字典创建 Paper 实例。

        未知字段将被忽略，不触发错误。

        Args:
            data: 包含论文信息的字典。

        Returns:
            Paper 实例。
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        return cls(**kwargs)

    @classmethod
    def from_search_result(cls, result: Dict[str, Any]) -> "Paper":
        """从搜索引擎标准化结果创建 Paper 实例。

        Args:
            result: SearchEngine.normalize_results() 输出的 dict。

        Returns:
            Paper 实例。
        """
        authors = result.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(";")]

        return cls(
            title=result.get("title", ""),
            authors=authors if isinstance(authors, list) else [],
            year=result.get("year"),
            abstract=result.get("abstract"),
            doi=result.get("doi"),
            arxiv_id=result.get("arxiv_id"),
            pdf_url=result.get("pdf_url"),
            url=result.get("url"),
            source=result.get("source"),
            citation_count=result.get("citation_count"),
            journal=result.get("journal"),
            pdf_path=result.get("pdf_path"),
            file_size=result.get("file_size", 0),
        )

    # ── 序列化 ──────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """将 Paper 转为字典（包含所有字段）。"""
        return asdict(self)

    def to_dict_compact(self) -> Dict[str, Any]:
        """转为紧凑字典，省略 None 和默认值字段。"""
        d = self.to_dict()
        return {
            k: v for k, v in d.items()
            if v not in (None, "", [], 0)
        }

    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """序列化为 JSON 字符串。

        Args:
            indent:       缩进空格数。
            ensure_ascii: 是否转义非 ASCII 字符。

        Returns:
            JSON 字符串。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def to_bibtex(self) -> str:
        """生成 BibTeX 引用条目。

        Returns:
            BibTeX 格式字符串。
        """
        if self.authors:
            first = self.authors[0].split()[-1] if self.authors[0].split() else "unknown"
            key = first.lower()
        else:
            key = self.title.split()[0].lower() if self.title else "unknown"
        key += self.year or ""

        authors_bib = " and ".join(self.authors) if self.authors else ""

        lines = [
            f"@article{{{key},",
            f"  title = {{{self.title}}},",
            f"  author = {{{authors_bib}}},",
            f"  year = {{{self.year or ''}}},",
        ]
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.journal:
            lines.append(f"  journal = {{{self.journal}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    # ── 属性 ────────────────────────────────────────────────────

    @property
    def has_pdf(self) -> bool:
        """是否已有本地 PDF 文件。"""
        if not self.pdf_path:
            return False
        return Path(self.pdf_path).exists()

    @property
    def identifier(self) -> Optional[str]:
        """返回最佳标识符（优先级: DOI > arXiv ID > URL）。"""
        return self.doi or self.arxiv_id or self.url

    @property
    def first_author(self) -> str:
        """返回第一作者姓名。"""
        return self.authors[0] if self.authors else ""

    @property
    def first_author_surname(self) -> str:
        """返回第一作者姓氏。"""
        if not self.authors:
            return "unknown"
        parts = self.authors[0].split()
        return parts[-1] if parts else "unknown"

    @property
    def citation(self) -> str:
        """生成简短引文，如 "Vaswani et al. (2017)"。"""
        surname = self.first_author_surname
        if len(self.authors) > 1:
            surname += " et al."
        year = f" ({self.year})" if self.year else ""
        return f"{surname}{year}"

    def __str__(self) -> str:
        title_preview = f"{self.title[:60]}{'...' if len(self.title) > 60 else ''}"
        return f"Paper({self.citation}: {title_preview})"

    def __repr__(self) -> str:
        return self.__str__()
