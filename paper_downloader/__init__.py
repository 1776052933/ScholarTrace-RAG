"""
paper_downloader — 学术论文自动下载器
======================================

一键搜索并下载学术论文 PDF，支持 arXiv、CrossRef、Google Scholar 多源检索。

可作为独立工具使用，也可作为 AI Agent / LLM 项目的论文获取组件。

便捷导入::

    from paper_downloader import download_paper, search_papers

    # 下载单篇论文
    paper = download_paper("Attention Is All You Need")
    print(paper.pdf_path)

    # 批量下载
    papers = download_papers(["GPT-4 Technical Report", "BERT: Pre-training"])

    # 搜索
    results = search_papers("diffusion models", max_results=5)

版本: 0.2.0
"""

__version__ = "0.2.0"

# ── 公共 API ──────────────────────────────────────────────────────

# 极简统一接口（最推荐 AI 项目使用）
from paper_downloader.src.interface import (
    download_pdf,
    batch_download_pdf,
    search_papers as search_papers_dict,
)

# 便捷函数（返回 Paper 对象）
from paper_downloader.src.api import (
    download_paper,
    download_papers,
    search_papers,
    get_paper_info,
    set_config,
    reset_downloader,
)

# 核心类
from paper_downloader.src.core.downloader import PaperDownloader

# 异步支持
try:
    from paper_downloader.src.core.async_downloader import AsyncPaperDownloader
except ImportError:
    AsyncPaperDownloader = None  # type: ignore[assignment]

# 数据模型
from paper_downloader.src.models.paper import Paper

# 异常类
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
    ValidationError,
    ConfigError,
    SearchError,
)

# ── 辅助组件（按需导入）───────────────────────────────────────────

# 搜索引擎
from paper_downloader.src.search_engines.search_factory import SearchFactory
from paper_downloader.src.search_engines.base_search import SearchEngine

# 配置管理
from paper_downloader.src.config.config_manager import ConfigManager

# 缓存（当前源码包可能未包含 cache 子模块，保持可选）
try:
    from paper_downloader.src.cache.cache_manager import CacheManager
    from paper_downloader.src.cache.cache_decorator import cached
except ImportError:
    CacheManager = None  # type: ignore[assignment]

    def cached(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ImportError("paper_downloader.src.cache is not available in this source tree")

# 下载管理
from paper_downloader.src.downloaders.download_manager import DownloadManager

# 批量处理
from paper_downloader.src.core.batch_processor import BatchProcessor
from paper_downloader.src.utils.report_generator import ReportGenerator

# 监控
from paper_downloader.src.monitoring.metrics import MetricsCollector
from paper_downloader.src.monitoring.health_check import HealthChecker

# ── 公开导出 ──────────────────────────────────────────────────────

__all__ = [
    # 极简统一接口（最推荐 AI 项目使用）
    "download_pdf",
    "batch_download_pdf",
    "search_papers_dict",
    # 便捷 API（返回 Paper 对象）
    "download_paper",
    "download_papers",
    "search_papers",
    "get_paper_info",
    "set_config",
    "reset_downloader",
    # 核心类
    "PaperDownloader",
    "AsyncPaperDownloader",
    # 数据模型
    "Paper",
    # 异常
    "PaperDownloaderError",
    "PaperNotFoundError",
    "DownloadError",
    "ValidationError",
    "ConfigError",
    "SearchError",
    # 引擎
    "SearchFactory",
    "SearchEngine",
    # 配置与缓存
    "ConfigManager",
    "CacheManager",
    "cached",
    # 下载与处理
    "DownloadManager",
    "BatchProcessor",
    "ReportGenerator",
    # 监控
    "MetricsCollector",
    "HealthChecker",
]
