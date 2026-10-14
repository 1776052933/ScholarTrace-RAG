"""
callback_manager.py — 回调事件管理器

提供多事件类型的回调注册与触发机制，
支持为搜索、下载、错误等生命周期事件注册多个回调函数。
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from paper_downloader.src.models.paper import Paper


# ── 事件类型 ──────────────────────────────────────────────────────

class CallbackEvent(str, Enum):
    """回调事件枚举。"""

    # 搜索阶段
    ON_SEARCH_START     = "on_search_start"      # 开始搜索, 参数: (query: str)
    ON_SEARCH_COMPLETE  = "on_search_complete"   # 搜索完成, 参数: (papers: List[Paper])
    ON_SEARCH_ERROR     = "on_search_error"      # 搜索出错, 参数: (query: str, error: Exception)

    # 下载阶段
    ON_DOWNLOAD_START   = "on_download_start"    # 开始下载, 参数: (paper: Paper)
    ON_DOWNLOAD_COMPLETE = "on_download_complete" # 下载完成, 参数: (paper: Paper)
    ON_DOWNLOAD_ERROR   = "on_download_error"    # 下载失败, 参数: (paper: Paper, error: Exception)

    # 批量阶段
    ON_BATCH_START      = "on_batch_start"       # 批量开始, 参数: (total: int)
    ON_BATCH_PROGRESS   = "on_batch_progress"    # 单篇完成, 参数: (current: int, total: int, paper: Paper)
    ON_BATCH_COMPLETE   = "on_batch_complete"    # 批量完成, 参数: (results: List[Paper])
    ON_BATCH_ERROR      = "on_batch_error"       # 批量错误, 参数: (error: Exception)

    # 通用
    ON_ERROR            = "on_error"             # 任何未分类错误, 参数: (error: Exception)
    ON_ALL_COMPLETE     = "on_all_complete"      # 所有操作完成, 参数: (summary: Dict)


# 回调函数类型别名
Callback = Callable[..., None]


class CallbackManager:
    """回调事件管理器。

    支持为多种生命周期事件注册和触发回调函数。
    每个事件可注册多个回调，按注册顺序依次调用。

    Usage::

        mgr = CallbackManager()

        @mgr.on(CallbackEvent.ON_DOWNLOAD_COMPLETE)
        def on_done(paper: Paper):
            print(f"Downloaded: {paper.title}")

        mgr.trigger(CallbackEvent.ON_DOWNLOAD_COMPLETE, paper)
    """

    def __init__(self):
        self._callbacks: Dict[CallbackEvent, List[Callback]] = {
            event: [] for event in CallbackEvent
        }
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── 注册 ──────────────────────────────────────────────────────

    def register(self, event: CallbackEvent, callback: Callback) -> None:
        """注册一个回调函数到指定事件。

        Args:
            event:    事件类型。
            callback: 回调函数，签名需与事件匹配。

        Raises:
            ValueError: callback 不可调用。
        """
        if not callable(callback):
            raise ValueError(f"callback 必须可调用，收到: {type(callback)}")
        self._callbacks[event].append(callback)
        self.logger.debug("注册回调: %s → %s", event.value, callback.__name__)

    def unregister(self, event: CallbackEvent, callback: Callback) -> bool:
        """取消注册一个回调。

        Args:
            event:    事件类型。
            callback: 要移除的回调函数。

        Returns:
            True 表示成功移除。
        """
        try:
            self._callbacks[event].remove(callback)
            return True
        except ValueError:
            return False

    def clear(self, event: Optional[CallbackEvent] = None) -> None:
        """清空指定事件的所有回调（或全部事件）。

        Args:
            event: 要清空的事件，None 表示清空所有。
        """
        if event is None:
            for evt in CallbackEvent:
                self._callbacks[evt].clear()
        else:
            self._callbacks[event].clear()

    # ── 装饰器 ────────────────────────────────────────────────────

    def on(self, event: CallbackEvent) -> Callable[[Callback], Callback]:
        """装饰器语法注册回调。

        Args:
            event: 事件类型。

        Returns:
            装饰器函数。

        Example::

            mgr = CallbackManager()

            @mgr.on(CallbackEvent.ON_SEARCH_COMPLETE)
            def handle_results(papers):
                ...
        """
        def decorator(fn: Callback) -> Callback:
            self.register(event, fn)
            return fn
        return decorator

    # ── 触发 ──────────────────────────────────────────────────────

    def trigger(self, event: CallbackEvent, *args: Any, **kwargs: Any) -> None:
        """触发指定事件的所有回调。

        任一回调抛出的异常会被捕获并记录，不会中断后续回调。

        Args:
            event: 要触发的事件。
            *args: 传递给回调的位置参数。
            **kwargs: 传递给回调的关键字参数。
        """
        for callback in self._callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                self.logger.warning(
                    "回调 %s 触发 %s 时异常: %s",
                    callback.__name__, event.value, exc,
                )

    def trigger_safe(self, event: CallbackEvent, *args: Any, **kwargs: Any) -> List[Exception]:
        """触发事件并收集所有异常（不记录日志）。

        Args:
            event: 要触发的事件。
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            所有被捕获的异常列表（空列表 = 全部成功）。
        """
        errors: List[Exception] = []
        for callback in self._callbacks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
        return errors

    # ── 属性 ──────────────────────────────────────────────────────

    def count(self, event: Optional[CallbackEvent] = None) -> int:
        """返回已注册的回调数量。

        Args:
            event: 指定事件，None 返回总数。
        """
        if event is None:
            return sum(len(v) for v in self._callbacks.values())
        return len(self._callbacks[event])

    def list_callbacks(self, event: Optional[CallbackEvent] = None) -> Dict[str, List[str]]:
        """列出所有已注册的回调函数名。

        Args:
            event: 指定事件，None 返回所有。

        Returns:
            {event_name: [callback_name, ...]}
        """
        events = [event] if event else list(CallbackEvent)
        return {
            e.value: [fn.__name__ for fn in self._callbacks[e]]
            for e in events
        }
