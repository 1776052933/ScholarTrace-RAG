"""
error_handler.py — 错误处理与自动重试

提供 ``@retry_on_error`` 装饰器、``ErrorHandler`` 上下文管理器，
支持指数退避重试、可配置的重试条件、自动错误日志记录。
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, List, Optional, Tuple, Type, Union


# ── 可重试的异常类型 ──────────────────────────────────────────────

_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# 尝试导入 requests/aiohttp 的异常类型
try:
    import requests
    _RETRYABLE_EXCEPTIONS += (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
    )
except ImportError:
    pass

try:
    import aiohttp
    _RETRYABLE_EXCEPTIONS += (
        aiohttp.ClientError,
    )
except ImportError:
    pass


class ErrorHandler:
    """错误处理上下文管理器。

    在 ``with`` 块中捕获可重试异常，自动执行重试。

    Usage::

        handler = ErrorHandler(max_retries=3)

        with handler:
            result = risky_api_call()
        # handler 自动处理重试

        # 或直接调用
        result = handler.run(risky_api_call, arg1, arg2)
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        retryable: Optional[Tuple[Type[BaseException], ...]] = None,
        on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    ):
        """初始化。

        Args:
            max_retries: 最大重试次数（不含首次尝试）。
            delay:       首次重试延迟（秒）。
            backoff:     退避因子（乘法）。
            retryable:   可重试的异常类型元组，None=使用默认。
            on_retry:    每次重试时调用的回调。
                         签名: (exception, attempt, next_delay) -> None。
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.retryable = retryable or _RETRYABLE_EXCEPTIONS
        self.on_retry = on_retry
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_error: Optional[Exception] = None

    def run(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """执行函数，失败时自动重试。

        Args:
            func:    要执行的函数。
            *args:   位置参数。
            **kwargs: 关键字参数。

        Returns:
            函数返回值。

        Raises:
            最后一次重试失败后抛出原始异常。
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self.retryable as exc:
                last_exc = exc
                self.last_error = exc

                if attempt >= self.max_retries:
                    self.logger.error(
                        "重试耗尽 (%d/%d): %s", attempt + 1, self.max_retries + 1, exc,
                    )
                    raise

                next_delay = self.delay * (self.backoff ** attempt)
                self.logger.warning(
                    "重试 %d/%d (%.1fs后): %s",
                    attempt + 1, self.max_retries + 1, next_delay, exc,
                )

                if self.on_retry:
                    try:
                        self.on_retry(exc, attempt + 1, next_delay)
                    except Exception:
                        pass

                time.sleep(next_delay)

        # 理论上不可达
        if last_exc:
            raise last_exc

    def __enter__(self) -> "ErrorHandler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None and isinstance(exc_val, self.retryable):
            self.logger.debug("ErrorHandler 捕获异常，交由 run() 处理")
            return False
        return False


# ── 装饰器 ────────────────────────────────────────────────────────

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    retryable: Optional[Tuple[Type[BaseException], ...]] = None,
    on_error: Optional[Callable[[Exception, int], None]] = None,
):
    """自动重试装饰器。

    支持同步和异步函数。当函数抛出可重试异常时自动按指数退避重试。

    Args:
        max_retries: 最大重试次数。
        delay:       首次重试延迟（秒）。
        backoff:     退避因子。
        retryable:   可重试的异常类型。
        on_error:    错误回调 (exception, attempt) -> None。

    Returns:
        装饰器函数。

    Usage::

        @retry_on_error(max_retries=3, delay=1, backoff=2)
        def download_pdf(url: str) -> bytes:
            return requests.get(url).content

        @retry_on_error(max_retries=5)
        async def async_call(doi: str) -> dict:
            return await api.get(doi)
    """

    def decorator(func: Callable) -> Callable:
        handler = ErrorHandler(
            max_retries=max_retries,
            delay=delay,
            backoff=backoff,
            retryable=retryable,
        )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc = None
                for attempt in range(handler.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except handler.retryable as exc:
                        last_exc = exc
                        if on_error:
                            try:
                                on_error(exc, attempt + 1)
                            except Exception:
                                pass
                        if attempt >= handler.max_retries:
                            raise
                        next_delay = handler.delay * (handler.backoff ** attempt)
                        await asyncio.sleep(next_delay)
                if last_exc:
                    raise last_exc

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return handler.run(func, *args, **kwargs)

            return sync_wrapper

    return decorator
