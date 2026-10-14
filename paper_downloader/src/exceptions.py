"""
exceptions.py — 自定义异常体系

提供 paper_downloader 模块的专用异常类，
方便调用方精确捕获和处理不同类型的错误。
"""

from typing import Any, List, Optional


class PaperDownloaderError(Exception):
    """paper_downloader 模块所有异常的基类。

    所有自定义异常均继承自此，方便调用方统一捕获。
    """

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

    当通过标题、DOI 或 arXiv ID 搜索不到目标论文时抛出。

    Example::

        raise PaperNotFoundError(
            f"未找到论文: {title}",
            details={"query": title, "engines": ["arxiv", "crossref"]},
        )
    """

    def __init__(
        self,
        message: str = "论文未找到",
        *,
        query: str = "",
        engines: Optional[List[str]] = None,
        details: Optional[Any] = None,
    ):
        combined = details or {}
        if query:
            combined["query"] = query  # type: ignore[index]
        if engines:
            combined["engines"] = engines  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.query = query
        self.engines = engines or []


class DownloadError(PaperDownloaderError):
    """下载失败。

    当 PDF 下载过程中出现网络错误、超时、服务器拒绝等时抛出。

    Example::

        raise DownloadError(
            f"下载 PDF 失败: {url}",
            details={"url": url, "attempt_count": 3, "last_error": str(exc)},
        )
    """

    def __init__(
        self,
        message: str = "PDF 下载失败",
        *,
        url: str = "",
        attempt_count: int = 0,
        last_error: str = "",
        details: Optional[Any] = None,
    ):
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
    """数据校验失败。

    当输入参数无效、论文数据不完整或 PDF 文件损坏时抛出。

    Example::

        raise ValidationError(
            "PDF 文件无效",
            details={"file_path": str(path), "reason": "corrupted"},
        )
    """

    def __init__(
        self,
        message: str = "数据校验失败",
        *,
        file_path: str = "",
        reason: str = "",
        details: Optional[Any] = None,
    ):
        combined = details or {}
        if file_path:
            combined["file_path"] = file_path  # type: ignore[index]
        if reason:
            combined["reason"] = reason  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.file_path = file_path
        self.reason = reason


class ConfigError(PaperDownloaderError):
    """配置错误。

    当配置文件缺失、格式错误或参数不合法时抛出。

    Example::

        raise ConfigError(
            f"未知的搜索引擎: {name}",
            details={"engine_name": name, "available": ["arxiv", "crossref"]},
        )
    """

    def __init__(
        self,
        message: str = "配置错误",
        *,
        config_path: str = "",
        missing_key: str = "",
        details: Optional[Any] = None,
    ):
        combined = details or {}
        if config_path:
            combined["config_path"] = config_path  # type: ignore[index]
        if missing_key:
            combined["missing_key"] = missing_key  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.config_path = config_path
        self.missing_key = missing_key


class SearchError(PaperDownloaderError):
    """搜索执行错误。

    当所有搜索引擎均返回错误时抛出。

    Example::

        raise SearchError(
            f"搜索全部引擎失败: {query}",
            details={"query": query, "engine_errors": {...}},
        )
    """

    def __init__(
        self,
        message: str = "搜索执行失败",
        *,
        query: str = "",
        engine_errors: Optional[dict] = None,
        details: Optional[Any] = None,
    ):
        combined = details or {}
        if query:
            combined["query"] = query  # type: ignore[index]
        if engine_errors:
            combined["engine_errors"] = engine_errors  # type: ignore[index]
        super().__init__(message, details=combined or None)
        self.query = query
        self.engine_errors = engine_errors or {}
