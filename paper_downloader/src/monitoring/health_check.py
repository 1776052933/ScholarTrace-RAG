"""
health_check.py — 健康检查

检查各搜索引擎 API 可用性、磁盘空间、网络连接状况。
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class HealthChecker:
    """系统健康检查器。

    检查搜索 API、磁盘空间、网络连接等运行环境状态。

    Usage::

        checker = HealthChecker()
        report = checker.run_full_check()
        if not report["healthy"]:
            print("警告：某些组件不可用")
    """

    # 检查连接的目标（用于网络可用性判断）
    _CONNECTIVITY_TARGETS = [
        ("api.crossref.org", 443),
        ("arxiv.org", 443),
        ("scholar.google.com", 443),
        ("8.8.8.8", 53),
    ]

    # 磁盘空间警告阈值（字节）
    _DISK_WARN_MB = 500  # 少于 500MB 警告
    _DISK_CRITICAL_MB = 100  # 少于 100MB 严重

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化健康检查器。

        Args:
            config: 配置字典（用于获取搜索引擎列表等）。
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── 搜索引擎检查 ──────────────────────────────────────────────

    def check_engines(self, engines: Optional[List[str]] = None) -> Dict[str, Any]:
        """检查各搜索引擎 API 是否可达。

        通过发送 HEAD/GET 请求到各 API 的根路径判定。

        Args:
            engines: 要检查的引擎名称列表，None 使用配置中的引擎。

        Returns:
            {engine_name: {"available": bool, "latency_ms": float, "error": str|null}}
        """
        import requests

        engine_config = engines or self.config.get("engines", {}).get(
            "priority", ["arxiv", "crossref"]
        )

        engine_urls = {
            "arxiv":           "https://export.arxiv.org/api/query?search_query=all:test&max_results=1",
            "crossref":        "https://api.crossref.org/works?rows=1",
            "google_scholar":  "https://scholar.google.com",
        }

        results: Dict[str, Any] = {}
        for engine_name in engine_config:
            url = engine_urls.get(engine_name)
            if not url:
                results[engine_name] = {"available": False, "error": "未知引擎"}
                continue

            try:
                start = time.monotonic()
                resp = requests.get(
                    url, timeout=10, allow_redirects=True,
                    headers={"User-Agent": "AutoPaper-HealthCheck/0.1"},
                )
                latency = (time.monotonic() - start) * 1000

                results[engine_name] = {
                    "available": resp.status_code < 500,
                    "latency_ms": round(latency, 1),
                    "status_code": resp.status_code,
                    "error": None if resp.status_code < 500 else f"HTTP {resp.status_code}",
                }
            except requests.exceptions.Timeout:
                results[engine_name] = {"available": False, "latency_ms": None, "error": "超时"}
            except requests.exceptions.ConnectionError:
                results[engine_name] = {"available": False, "latency_ms": None, "error": "连接失败"}
            except Exception as exc:
                results[engine_name] = {"available": False, "latency_ms": None, "error": str(exc)[:200]}

        return results

    # ── 磁盘空间检查 ──────────────────────────────────────────────

    def check_disk_space(self, path: Optional[str] = None) -> Dict[str, Any]:
        """检查磁盘空间。

        Args:
            path: 检查的路径，None 使用默认下载目录。

        Returns:
            {free_mb, total_mb, used_mb, percent_free, status, warning}。
        """
        target = path or self.config.get("output", {}).get("default_dir", ".")
        target = os.path.abspath(target)

        # 确保路径存在
        Path(target).mkdir(parents=True, exist_ok=True)

        try:
            usage = shutil.disk_usage(target)
            free_mb = usage.free / (1024 * 1024)
            total_mb = usage.total / (1024 * 1024)

            status = "ok"
            warning = None
            if free_mb < self._DISK_CRITICAL_MB:
                status = "critical"
                warning = f"磁盘空间严重不足：仅剩 {free_mb:.1f} MB"
            elif free_mb < self._DISK_WARN_MB:
                status = "warning"
                warning = f"磁盘空间偏低：剩余 {free_mb:.1f} MB"

            return {
                "path": target,
                "free_mb": round(free_mb, 1),
                "total_mb": round(total_mb, 1),
                "used_mb": round((usage.total - usage.free) / (1024 * 1024), 1),
                "percent_free": round(free_mb / total_mb * 100, 1) if total_mb > 0 else 0,
                "status": status,
                "warning": warning,
            }
        except OSError as exc:
            return {
                "path": target,
                "error": str(exc),
                "status": "error",
            }

    # ── 网络连接检查 ──────────────────────────────────────────────

    def check_network(self) -> Dict[str, Any]:
        """检查网络连通性。

        通过 TCP 连接多个目标端口测试。

        Returns:
            {connected: bool, targets: {...}, details: str}。
        """
        results: Dict[str, Any] = {"connected": False, "targets": {}}
        success_count = 0

        for host, port in self._CONNECTIVITY_TARGETS:
            key = f"{host}:{port}"
            try:
                sock = socket.create_connection((host, port), timeout=5)
                sock.close()
                results["targets"][key] = {"reachable": True, "latency_ms": None}
                success_count += 1
            except (socket.timeout, socket.error, OSError) as exc:
                results["targets"][key] = {"reachable": False, "error": str(exc)[:100]}

        results["connected"] = success_count >= 1
        if success_count >= len(self._CONNECTIVITY_TARGETS):
            results["details"] = "所有目标可达"
        elif success_count >= 1:
            results["details"] = f"部分连通（{success_count}/{len(self._CONNECTIVITY_TARGETS)}）"
        else:
            results["details"] = "无网络连接"

        return results

    # ── 综合健康报告 ──────────────────────────────────────────────

    def run_full_check(
        self,
        engines: Optional[List[str]] = None,
        download_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行完整健康检查。

        Args:
            engines:       要检查的引擎列表。
            download_path: 下载目录路径。

        Returns:
            完整健康报告 dict，包含 `healthy` 布尔字段。
        """
        network = self.check_network()
        disk = self.check_disk_space(download_path)
        engine_status = self.check_engines(engines)

        engine_healthy = all(
            e.get("available", False) for e in engine_status.values()
        )

        healthy = (
            network["connected"] and
            disk.get("status", "error") != "critical" and
            engine_healthy
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "healthy": healthy,
            "network": network,
            "disk": disk,
            "engines": engine_status,
            "issues": [],
        }

        # 收集问题列表
        if not network["connected"]:
            report["issues"].append("无网络连接")
        if disk.get("status") == "critical":
            report["issues"].append(disk.get("warning", "磁盘空间严重不足"))
        elif disk.get("status") == "warning":
            report["issues"].append(disk.get("warning", "磁盘空间偏低"))
        for name, status in engine_status.items():
            if not status.get("available", False):
                report["issues"].append(
                    f"搜索引擎 {name} 不可用: {status.get('error', '未知')}"
                )

        self.logger.info(
            "健康检查完成: %s (问题: %d)",
            "健康" if healthy else "不健康",
            len(report["issues"]),
        )
        return report
