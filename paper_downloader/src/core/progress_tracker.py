"""
progress_tracker.py — 进度跟踪器

提供批量操作的进度追踪，支持多回调通知，
适用于 GUI 进度条、日志记录、状态推送等场景。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from paper_downloader.src.models.paper import Paper


class ProgressTracker:
    """批量处理进度跟踪器。

    线程安全，支持注册多个进度回调函数。

    Usage::

        tracker = ProgressTracker(total=50, description="下载论文")

        @tracker.on_update
        def show_progress(current: int, total: int, message: str):
            pct = current / total * 100
            print(f"[{pct:.0f}%] {message}")

        for paper in papers:
            download(paper)
            tracker.update(message=paper.title)
    """

    def __init__(
        self,
        total: int = 0,
        description: str = "Processing",
    ):
        """初始化进度跟踪器。

        Args:
            total:       总任务数。
            description: 任务描述。
        """
        self._lock = threading.RLock()
        self._total = max(total, 0)
        self._current = 0
        self._failed = 0
        self._skipped = 0
        self._description = description
        self._start_time: Optional[float] = None
        self._callbacks: List[Callable[[int, int, str], None]] = []
        self._error_callbacks: List[Callable[[str, Exception], None]] = []
        self._logger = logging.getLogger(self.__class__.__name__)

    # ── 属性 ──────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """总任务数。"""
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        with self._lock:
            self._total = max(value, 0)

    @property
    def current(self) -> int:
        """已完成数。"""
        return self._current

    @property
    def failed(self) -> int:
        """失败数。"""
        return self._failed

    @property
    def skipped(self) -> int:
        """跳过数。"""
        return self._skipped

    @property
    def progress(self) -> float:
        """进度 0.0 ~ 1.0。"""
        if self._total == 0:
            return 1.0
        return min(self._current / self._total, 1.0)

    @property
    def percentage(self) -> float:
        """百分比 0.0 ~ 100.0。"""
        return self.progress * 100.0

    @property
    def elapsed(self) -> float:
        """已耗时（秒）。"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def eta(self) -> Optional[float]:
        """预估剩余时间（秒）。"""
        if self._current == 0 or self._start_time is None:
            return None
        elapsed = self.elapsed
        rate = self._current / elapsed if elapsed > 0 else 0
        remaining = self._total - self._current
        return remaining / rate if rate > 0 else None

    # ── 进度更新 ──────────────────────────────────────────────────

    def start(self, total: Optional[int] = None) -> None:
        """标记开始追踪。

        Args:
            total: 设置总任务数（覆盖构造函数中的值）。
        """
        with self._lock:
            if total is not None:
                self._total = max(total, 0)
            self._current = 0
            self._failed = 0
            self._skipped = 0
            self._start_time = time.time()
        self._logger.info("开始: %s (total=%d)", self._description, self._total)

    def update(
        self,
        increment: int = 1,
        success: bool = True,
        message: str = "",
        paper: Optional[Paper] = None,
    ) -> None:
        """更新进度。

        Args:
            increment: 进度增量。
            success:   是否成功（决定 failed 计数是否增加）。
            message:   当前任务的描述信息。
            paper:     关联的 Paper 对象（传递给回调）。
        """
        with self._lock:
            if success:
                self._current += increment
            else:
                self._failed += increment

        info = message
        if paper:
            info = info or paper.title[:60]

        # 触发进度回调
        for cb in self._callbacks:
            try:
                cb(self._current, self._total, info)
            except Exception as exc:
                self._logger.warning("进度回调异常: %s", exc)

    def skip(self, count: int = 1, reason: str = "") -> None:
        """标记跳过。

        Args:
            count:  跳过数量。
            reason: 跳过原因。
        """
        with self._lock:
            self._skipped += count
            self._current += count
        if reason:
            self._logger.debug("跳过: %s", reason)

    def report_error(self, message: str, error: Optional[Exception] = None) -> None:
        """报告错误但不中断进度。

        Args:
            message: 错误描述。
            error:   异常对象。
        """
        with self._lock:
            self._failed += 1
        for cb in self._error_callbacks:
            try:
                cb(message, error or Exception(message))
            except Exception as exc:
                self._logger.warning("错误回调异常: %s", exc)

    def finish(self) -> Dict[str, Any]:
        """标记完成，返回汇总信息。

        Returns:
            包含 total / completed / failed / skipped / elapsed / eta 的 dict。
        """
        et = self.elapsed
        summary = {
            "description": self._description,
            "total":        self._total,
            "completed":    self._current,
            "failed":       self._failed,
            "skipped":      self._skipped,
            "elapsed_sec":  round(et, 1),
            "rate_per_sec": round(self._current / et, 2) if et > 0 else 0,
        }
        self._logger.info(
            "完成: %s — %d/%d (失败:%d, 跳过:%d, 耗时:%.1fs)",
            self._description, self._current, self._total,
            self._failed, self._skipped, et,
        )
        return summary

    # ── 回调注册 ──────────────────────────────────────────────────

    def on_update(self, callback: Callable[[int, int, str], None]) -> Callable[[int, int, str], None]:
        """注册进度更新回调（支持装饰器语法）。

        Args:
            callback: 回调函数，签名为 (current: int, total: int, message: str) -> None。
        """
        self._callbacks.append(callback)
        return callback

    def on_error(
        self, callback: Callable[[str, Exception], None],
    ) -> Callable[[str, Exception], None]:
        """注册错误报告回调。

        Args:
            callback: 回调函数，签名为 (message: str, error: Exception) -> None。
        """
        self._error_callbacks.append(callback)
        return callback

    def remove_callback(self, callback: Callable) -> bool:
        """移除一个回调（从 update 和 error 列表中均尝试移除）。"""
        removed = False
        for lst in (self._callbacks, self._error_callbacks):
            try:
                lst.remove(callback)
                removed = True
            except ValueError:
                pass
        return removed

    def clear_callbacks(self) -> None:
        """移除所有回调。"""
        self._callbacks.clear()
        self._error_callbacks.clear()

    # ── 显示 ──────────────────────────────────────────────────────

    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度的快照。

        Returns:
            current / total / failed / skipped / percent / elapsed / eta 字段。
        """
        return {
            "current":   self._current,
            "total":     self._total,
            "failed":    self._failed,
            "skipped":   self._skipped,
            "percent":   round(self.percentage, 1),
            "elapsed":   round(self.elapsed, 1),
            "eta":       round(self.eta, 1) if self.eta is not None else None,
        }

    def __repr__(self) -> str:
        p = self.get_progress()
        return (
            f"ProgressTracker({p['current']}/{p['total']}, "
            f"{p['percent']}%, failed={p['failed']})"
        )
