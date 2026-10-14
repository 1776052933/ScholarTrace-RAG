"""
paper_downloader.src.exceptions — 自定义异常与错误处理.

注意: exceptions/ 目录会覆盖旧的 exceptions.py 文件，
因此在此重新导出所有异常类型以确保向后兼容。
"""

from __future__ import annotations
from typing import Any, Optional

from paper_downloader.src.exceptions.error_handler import ErrorHandler, retry_on_error


# ═══════════════════════════════════════════════════════════════════
# 异常类（保持在 exceptions/__init__.py 中以兼容旧导入路径）
# ═══════════════════════════════════════════════════════════════════

class PaperDownloaderError(Exception):
    """paper_downloader 所有异常的基类。"""

    def __init__(self, message: str = "", *, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        base = self.message or self.__class__.__name__
        if self.details:
            base += f" (details: {self.details})"
        return base


class PaperNotFoundError(PaperDownloaderError):
    """论文未找到。

    当搜索结果为空时抛出。
    """

    def __init__(self, message: str = "论文未找到", *, query: str = "",
                 engines: Optional[list] = None, details: Optional[Any] = None):
        combined = details or {}
        if query:
            combined["query"] = query  # type: ignore[index]
        if engines:
            combined["engines"] = engines  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.query = query
        self.engines = engines or []


class DownloadError(PaperDownloaderError):
    """下载失败。"""

    def __init__(self, message: str = "PDF 下载失败", *, url: str = "",
                 attempt_count: int = 0, last_error: str = "",
                 details: Optional[Any] = None):
        combined = details or {}
        if url:
            combined["url"] = url  # type: ignore[index]
        if attempt_count:
            combined["attempt_count"] = attempt_count  # type: ignore[index]
        if last_error:
            combined["last_error"] = last_error  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.url = url
        self.attempt_count = attempt_count
        self.last_error = last_error


class ValidationError(PaperDownloaderError):
    """数据校验失败。"""

    def __init__(self, message: str = "数据校验失败", *, file_path: str = "",
                 reason: str = "", details: Optional[Any] = None):
        combined = details or {}
        if file_path:
            combined["file_path"] = file_path  # type: ignore[index]
        if reason:
            combined["reason"] = reason  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.file_path = file_path
        self.reason = reason


class ConfigError(PaperDownloaderError):
    """配置错误。"""

    def __init__(self, message: str = "配置错误", *, config_path: str = "",
                 missing_key: str = "", details: Optional[Any] = None):
        combined = details or {}
        if config_path:
            combined["config_path"] = config_path  # type: ignore[index]
        if missing_key:
            combined["missing_key"] = missing_key  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.config_path = config_path
        self.missing_key = missing_key


class SearchError(PaperDownloaderError):
    """搜索执行错误。"""

    def __init__(self, message: str = "搜索执行失败", *, query: str = "",
                 engine_errors: Optional[dict] = None, details: Optional[Any] = None):
        combined = details or {}
        if query:
            combined["query"] = query  # type: ignore[index]
        if engine_errors:
            combined["engine_errors"] = engine_errors  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.query = query
        self.engine_errors = engine_errors or {}


__all__ = [
    "ErrorHandler",
    "retry_on_error",
    "PaperDownloaderError",
    "PaperNotFoundError",
    "DownloadError",
    "ValidationError",
    "ConfigError",
    "SearchError",
]
