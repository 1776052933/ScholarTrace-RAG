"""
paper_downloader.src.downloaders — 论文 PDF 下载与处理模块.

提供的功能:
    - Downloader         : 下载器抽象基类
    - HTTPDownloader     : 通用 HTTP 下载器（断点续传 + 进度条）
    - ArxivPDFDownloader : arXiv 专用下载器（API 优先 + HTTP 回退）
    - PDFProcessor       : PDF 元数据提取 / 重命名 / 损坏检测
    - DownloadManager    : 并发下载管理器（队列 / 重试 / 历史记录）
"""

from paper_downloader.src.downloaders.base_downloader import Downloader
from paper_downloader.src.downloaders.http_downloader import HTTPDownloader
from paper_downloader.src.downloaders.arxiv_downloader import ArxivPDFDownloader
from paper_downloader.src.downloaders.pdf_processor import PDFProcessor
from paper_downloader.src.downloaders.download_manager import DownloadManager

__all__ = [
    "Downloader",
    "HTTPDownloader",
    "ArxivPDFDownloader",
    "PDFProcessor",
    "DownloadManager",
]
from paper_downloader.src.downloaders.source_resolver import (
    DownloadCandidate,
    DownloadSourceResolver,
)

__all__ = ["DownloadCandidate", "DownloadSourceResolver"]
