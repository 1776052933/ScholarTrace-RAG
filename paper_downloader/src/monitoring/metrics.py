"""
metrics.py — 监控指标收集

线程安全的运行时指标收集器，追踪搜索、下载、API 调用的次数与耗时。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """运行时指标收集器（线程安全）。

    记录会话级别的统计信息：搜索次数、下载成功/失败数、
    平均下载速度、API 调用次数等。

    Usage::

        metrics = MetricsCollector()

        with metrics.timer("search"):
            results = engine.search(query)

        metrics.record_search()
        metrics.record_download(success=True, size_bytes=1024000)

        stats = metrics.get_stats()
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._started_at = time.time()

        # 搜索
        self._search_count = 0
        self._search_total_time = 0.0

        # 下载
        self._download_success = 0
        self._download_failed = 0
        self._download_total_bytes = 0
        self._download_total_time = 0.0

        # API 调用
        self._api_calls: Dict[str, int] = {}  # engine → count
        self._api_errors: Dict[str, int] = {}  # engine → errors

        # 错误
        self._error_count = 0

        # 缓存
        self._cache_hits = 0
        self._cache_misses = 0

    # ── 记录 ──────────────────────────────────────────────────────

    def record_search(self, duration: float = 0.0) -> None:
        """记录一次搜索。

        Args:
            duration: 搜索耗时（秒）。
        """
        with self._lock:
            self._search_count += 1
            self._search_total_time += duration

    def record_download(
        self,
        success: bool = True,
        size_bytes: int = 0,
        duration: float = 0.0,
    ) -> None:
        """记录一次下载。

        Args:
            success:    是否成功。
            size_bytes: 下载文件大小（字节）。
            duration:   下载耗时（秒）。
        """
        with self._lock:
            if success:
                self._download_success += 1
                self._download_total_bytes += size_bytes
            else:
                self._download_failed += 1
            self._download_total_time += duration

    def record_api_call(self, engine: str, success: bool = True) -> None:
        """记录一次 API 调用。

        Args:
            engine:  搜索引擎名称。
            success: 是否成功。
        """
        with self._lock:
            self._api_calls[engine] = self._api_calls.get(engine, 0) + 1
            if not success:
                self._api_errors[engine] = self._api_errors.get(engine, 0) + 1

    def record_error(self) -> None:
        """记录一次错误。"""
        with self._lock:
            self._error_count += 1

    def record_cache(self, hit: bool = True) -> None:
        """记录一次缓存查询。

        Args:
            hit: True=命中，False=未命中。
        """
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    # ── 计时器 ────────────────────────────────────────────────────

    class _Timer:
        """上下文计时器。"""
        def __init__(self, collector: "MetricsCollector", label: str):
            self._collector = collector
            self._label = label
            self._start = 0.0

        def __enter__(self):
            self._start = time.time()
            return self

        def __exit__(self, *args):
            elapsed = time.time() - self._start
            if self._label == "search":
                self._collector.record_search(duration=elapsed)
            elif self._label == "download":
                self._collector.record_download(duration=elapsed)

    def timer(self, label: str = "search") -> _Timer:
        """创建上下文计时器。

        Usage::

            with metrics.timer("search"):
                results = engine.search(query)
        """
        return self._Timer(self, label)

    # ── 统计 ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计快照。

        Returns:
            包含所有指标的字典。
        """
        with self._lock:
            uptime = time.time() - self._started_at
            total_downloads = self._download_success + self._download_failed

            avg_speed = 0.0
            if self._download_total_time > 0:
                avg_speed = self._download_total_bytes / self._download_total_time

            download_success_rate = 0.0
            if total_downloads > 0:
                download_success_rate = self._download_success / total_downloads * 100

            cache_hit_rate = 0.0
            total_cache = self._cache_hits + self._cache_misses
            if total_cache > 0:
                cache_hit_rate = self._cache_hits / total_cache * 100

            return {
                "uptime_seconds": round(uptime, 1),
                "started_at": datetime.fromtimestamp(self._started_at).isoformat(),
                "search": {
                    "count": self._search_count,
                    "total_time_sec": round(self._search_total_time, 2),
                    "avg_time_sec": round(
                        self._search_total_time / max(self._search_count, 1), 2,
                    ),
                },
                "download": {
                    "success": self._download_success,
                    "failed": self._download_failed,
                    "total": total_downloads,
                    "success_rate_pct": round(download_success_rate, 1),
                    "total_bytes": self._download_total_bytes,
                    "total_mb": round(self._download_total_bytes / (1024 * 1024), 2),
                    "total_time_sec": round(self._download_total_time, 2),
                    "avg_speed_kbps": round(avg_speed / 1024, 1),
                },
                "api": {
                    "calls": dict(self._api_calls),
                    "errors": dict(self._api_errors),
                    "total_calls": sum(self._api_calls.values()),
                    "total_errors": sum(self._api_errors.values()),
                },
                "cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate_pct": round(cache_hit_rate, 1),
                },
                "error_count": self._error_count,
            }

    def reset_stats(self) -> None:
        """重置所有统计。"""
        with self._lock:
            self._started_at = time.time()
            self._search_count = 0
            self._search_total_time = 0.0
            self._download_success = 0
            self._download_failed = 0
            self._download_total_bytes = 0
            self._download_total_time = 0.0
            self._api_calls.clear()
            self._api_errors.clear()
            self._error_count = 0
            self._cache_hits = 0
            self._cache_misses = 0

    def export_json(self, indent: int = 2) -> str:
        """导出统计数据为 JSON 字符串。"""
        return json.dumps(self.get_stats(), ensure_ascii=False, indent=indent)

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"Metrics("
            f"searches={s['search']['count']}, "
            f"downloads={s['download']['total']} "
            f"({s['download']['success_rate_pct']}%), "
            f"api_calls={s['api']['total_calls']}, "
            f"cache_hit={s['cache']['hit_rate_pct']}%"
            f")"
        )
