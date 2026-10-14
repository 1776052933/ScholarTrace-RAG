"""
paper_downloader.src.search_engines — 多源论文搜索引擎集成.

提供的搜索引擎:
    - ArxivSearch      : arXiv 官方 API (需要: pip install arxiv)
    - CrossrefSearch   : CrossRef REST API (需要: pip install requests)
    - GoogleScholarSearch : Google Scholar (需要: pip install scholarly)
    - SearchFactory    : 多引擎联合搜索 & 结果合并去重
"""

from paper_downloader.src.search_engines.base_search import SearchEngine

# 可选导入 — 缺少依赖时对应的类仍可被引用但初始化时会警告
try:
    from paper_downloader.src.search_engines.arxiv_search import ArxivSearch
except ImportError:
    ArxivSearch = None  # type: ignore[assignment]

try:
    from paper_downloader.src.search_engines.crossref_search import CrossrefSearch
except ImportError:
    CrossrefSearch = None  # type: ignore[assignment]

try:
    from paper_downloader.src.search_engines.google_scholar_search import GoogleScholarSearch
except ImportError:
    GoogleScholarSearch = None  # type: ignore[assignment]

from paper_downloader.src.search_engines.search_factory import SearchFactory

__all__ = [
    "SearchEngine",
    "ArxivSearch",
    "CrossrefSearch",
    "GoogleScholarSearch",
    "SearchFactory",
]
from paper_downloader.src.search_engines.openalex_search import OpenAlexSearch
from paper_downloader.src.search_engines.semantic_scholar_search import SemanticScholarSearch

__all__ = ["OpenAlexSearch", "SemanticScholarSearch"]
