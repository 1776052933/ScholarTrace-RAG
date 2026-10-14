"""paper_downloader.src.core — 核心下载器与辅助组件."""

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.core.callback_manager import CallbackManager, CallbackEvent
from paper_downloader.src.core.progress_tracker import ProgressTracker
from paper_downloader.src.core.batch_processor import BatchProcessor

__all__ = [
    "PaperDownloader",
    "CallbackManager",
    "CallbackEvent",
    "ProgressTracker",
    "BatchProcessor",
]
