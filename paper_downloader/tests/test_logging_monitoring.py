"""
test_logging_monitoring.py — 日志/监控/错误处理/报告测试.

覆盖 logger、MetricsCollector、HealthChecker、ErrorHandler、
retry_on_error、ReportGenerator 以及 core/downloader 的集成。
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import requests

from paper_downloader.src.utils.logger import setup_logging, get_logger
from paper_downloader.src.monitoring.metrics import MetricsCollector
from paper_downloader.src.monitoring.health_check import HealthChecker
from paper_downloader.src.exceptions.error_handler import ErrorHandler, retry_on_error
from paper_downloader.src.utils.report_generator import ReportGenerator
from paper_downloader.src.models.paper import Paper


# ═══════════════════════════════════════════════════════════════════
# Logger 测试
# ═══════════════════════════════════════════════════════════════════

class TestLogger:
    """日志系统测试."""

    def test_get_logger(self):
        """获取日志器."""
        logger = get_logger("test.module")
        assert logger is not None
        assert logger.level is not None

    def test_same_logger_cached(self):
        """同名日志器缓存."""
        a = get_logger("test.cached")
        b = get_logger("test.cached")
        assert a is b

    def test_setup_logging_with_file(self, tmp_path):
        """配置文件输出."""
        log_file = tmp_path / "test.log"
        setup_logging(level="DEBUG", log_file=str(log_file), color=False)
        logger = get_logger("test.file")
        logger.debug("test message")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "test message" in content

    def test_setup_logging_levels(self):
        """日志级别."""
        setup_logging(level="ERROR", color=False)
        logger = get_logger("test.level")
        assert logger.level == 40  # ERROR

    def test_custom_format(self, tmp_path):
        """自定义格式."""
        log_file = tmp_path / "fmt.log"
        setup_logging(
            level="INFO",
            log_file=str(log_file),
            log_format="[%(levelname)s] %(name)s: %(message)s",
            color=False,
        )
        logger = get_logger("test.fmt")
        logger.info("formatted")
        assert log_file.exists()
        assert "[INFO]" in log_file.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════
# MetricsCollector 测试
# ═══════════════════════════════════════════════════════════════════

class TestMetricsCollector:
    """指标收集器测试."""

    def test_record_search(self):
        """记录搜索."""
        mc = MetricsCollector()
        mc.record_search(duration=1.5)
        mc.record_search(duration=0.8)
        stats = mc.get_stats()
        assert stats["search"]["count"] == 2
        assert stats["search"]["total_time_sec"] == 2.3

    def test_record_download(self):
        """记录下载."""
        mc = MetricsCollector()
        mc.record_download(success=True, size_bytes=1024000, duration=2.0)
        mc.record_download(success=False)
        stats = mc.get_stats()
        assert stats["download"]["success"] == 1
        assert stats["download"]["failed"] == 1
        assert stats["download"]["total_mb"] > 0.9

    def test_record_api_call(self):
        """记录 API 调用."""
        mc = MetricsCollector()
        mc.record_api_call("arxiv")
        mc.record_api_call("arxiv")
        mc.record_api_call("crossref")
        mc.record_api_call("arxiv", success=False)
        stats = mc.get_stats()
        assert stats["api"]["calls"]["arxiv"] == 3
        assert stats["api"]["errors"]["arxiv"] == 1

    def test_record_cache(self):
        """记录缓存."""
        mc = MetricsCollector()
        mc.record_cache(hit=True)
        mc.record_cache(hit=True)
        mc.record_cache(hit=False)
        stats = mc.get_stats()
        assert stats["cache"]["hits"] == 2
        assert stats["cache"]["misses"] == 1
        assert abs(stats["cache"]["hit_rate_pct"] - 66.7) < 1.0

    def test_record_error(self):
        """记录错误."""
        mc = MetricsCollector()
        mc.record_error()
        mc.record_error()
        assert mc.get_stats()["error_count"] == 2

    def test_timer_context(self):
        """上下文计时器."""
        mc = MetricsCollector()
        with mc.timer("search"):
            time.sleep(0.01)
        stats = mc.get_stats()
        assert stats["search"]["count"] == 1
        assert stats["search"]["total_time_sec"] > 0

    def test_reset_stats(self):
        """重置统计."""
        mc = MetricsCollector()
        mc.record_search()
        mc.record_download()
        mc.reset_stats()
        stats = mc.get_stats()
        assert stats["search"]["count"] == 0
        assert stats["download"]["total"] == 0

    def test_export_json(self):
        """导出 JSON."""
        mc = MetricsCollector()
        mc.record_search()
        j = mc.export_json()
        data = json.loads(j)
        assert data["search"]["count"] == 1

    def test_thread_safety(self):
        """线程安全."""
        import threading
        mc = MetricsCollector()

        def record():
            for _ in range(100):
                mc.record_search()
                mc.record_download(success=True, size_bytes=100)

        threads = [threading.Thread(target=record) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = mc.get_stats()
        assert stats["search"]["count"] == 500


# ═══════════════════════════════════════════════════════════════════
# HealthChecker 测试
# ═══════════════════════════════════════════════════════════════════

class TestHealthChecker:
    """健康检查器测试."""

    def test_check_disk_space(self, tmp_path):
        """磁盘空间检查."""
        checker = HealthChecker()
        result = checker.check_disk_space(str(tmp_path))
        assert "free_mb" in result
        assert result["path"] == str(tmp_path)
        assert result["status"] in ("ok", "warning", "critical")

    def test_check_disk_space_nonexistent(self, tmp_path):
        """不存在的路径自动创建."""
        checker = HealthChecker()
        sub = tmp_path / "subdir"
        result = checker.check_disk_space(str(sub))
        assert result["status"] != "error"

    def test_check_network(self):
        """网络检查."""
        checker = HealthChecker()
        result = checker.check_network()
        assert "connected" in result
        assert "targets" in result
        assert isinstance(result["connected"], bool)

    def test_check_engines(self):
        """引擎可用性检查 — mock HTTP."""
        checker = HealthChecker()
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            result = checker.check_engines(["arxiv"])
            assert "arxiv" in result
            assert result["arxiv"]["available"] is True

    def test_check_engines_timeout(self):
        """引擎超时."""
        checker = HealthChecker()
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            result = checker.check_engines(["crossref"])
            assert result["crossref"]["available"] is False
            assert result["crossref"]["error"] is not None

    def test_run_full_check(self, tmp_path):
        """综合健康报告."""
        checker = HealthChecker()
        with patch.object(checker, "check_network", return_value={"connected": True}):
            with patch.object(checker, "check_engines", return_value={
                "arxiv": {"available": True, "latency_ms": 100, "error": None},
            }):
                report = checker.run_full_check(download_path=str(tmp_path))
                assert "healthy" in report
                assert "timestamp" in report
                assert "issues" in report


# ═══════════════════════════════════════════════════════════════════
# ErrorHandler 测试
# ═══════════════════════════════════════════════════════════════════

class TestErrorHandler:
    """错误处理与重试测试."""

    def test_no_error(self):
        """无错误时正常返回."""
        handler = ErrorHandler(max_retries=2, delay=0.01)
        result = handler.run(lambda x: x * 2, 21)
        assert result == 42

    def test_retry_then_succeed(self):
        """重试后成功."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("fail")
            return "ok"

        handler = ErrorHandler(max_retries=3, delay=0.01, backoff=1)
        result = handler.run(flaky)
        assert result == "ok"
        assert call_count[0] == 3

    def test_exhausted_retries(self):
        """重试耗尽."""
        handler = ErrorHandler(max_retries=2, delay=0.01)

        def always_fail():
            raise ConnectionError("always")

        with pytest.raises(ConnectionError):
            handler.run(always_fail)

    def test_non_retryable_not_caught(self):
        """不可重试异常直接抛出."""
        handler = ErrorHandler(max_retries=3, delay=0.01)

        def value_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            handler.run(value_error)

    def test_retry_decorator_sync(self):
        """同步 retry_on_error 装饰器."""
        call_count = [0]

        @retry_on_error(max_retries=3, delay=0.01, backoff=1)
        def decorated():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "done"

        result = decorated()
        assert result == "done"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_async(self):
        """异步 retry_on_error 装饰器."""
        call_count = [0]

        @retry_on_error(max_retries=3, delay=0.01, backoff=1)
        async def async_decorated():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "async_done"

        result = await async_decorated()
        assert result == "async_done"
        assert call_count[0] == 2

    def test_retry_decorator_exhausted(self):
        """装饰器 — 重试耗尽."""
        @retry_on_error(max_retries=1, delay=0.01)
        def always_fails():
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            always_fails()

    def test_on_retry_callback(self):
        """重试回调."""
        attempts: List[int] = []

        def on_retry(exc, attempt, delay):
            attempts.append(attempt)

        handler = ErrorHandler(max_retries=2, delay=0.01, on_retry=on_retry)

        def fails_twice():
            if len(attempts) < 2:
                raise ConnectionError()
            return "ok"

        handler.run(fails_twice)
        assert len(attempts) == 2


# ═══════════════════════════════════════════════════════════════════
# ReportGenerator 测试
# ═══════════════════════════════════════════════════════════════════

class TestReportGenerator:
    """报告生成器测试."""

    @pytest.fixture
    def sample_papers(self, tmp_path):
        """测试用 Paper 列表."""
        real_pdf = tmp_path / "a.pdf"
        real_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        return [
            Paper(title="Paper Alpha", authors=["Alice"], year="2024",
                  doi="10.a", source="arxiv", pdf_path=str(real_pdf), file_size=1024),
            Paper(title="Paper Beta", authors=["Bob"], year="2023",
                  doi="10.b", source="crossref"),
            Paper(title="Paper Gamma", authors=["Charlie", "Dana"], year="2024",
                  source="arxiv"),
        ]

    def test_generate_summary(self, sample_papers):
        """生成摘要."""
        gen = ReportGenerator()
        summary = gen.generate_summary(sample_papers)
        assert summary["total"] == 3
        assert summary["succeeded"] == 1  # 只有 Alpha 有真实 pdf
        assert summary["failed"] == 2
        assert "arxiv" in summary["by_source"]

    def test_export_json(self, sample_papers, tmp_path):
        """导出 JSON."""
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.export_json(sample_papers)
        assert os.path.exists(path)
        data = json.load(open(path))
        assert data["total"] == 3

    def test_export_json_string(self, sample_papers):
        """导出 JSON 字符串."""
        gen = ReportGenerator()
        s = gen.export_json_string(sample_papers)
        data = json.loads(s)
        assert data["total"] == 3

    def test_export_csv(self, sample_papers, tmp_path):
        """导出 CSV."""
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.export_csv(sample_papers)
        assert os.path.exists(path)
        content = Path(path).read_text(encoding="utf-8-sig")
        assert "Paper Alpha" in content
        assert "success" in content

    def test_export_csv_string(self, sample_papers):
        """导出 CSV 字符串."""
        gen = ReportGenerator()
        s = gen.export_csv_string(sample_papers)
        assert "Paper Alpha" in s

    def test_generate_markdown(self, sample_papers):
        """生成 Markdown."""
        gen = ReportGenerator()
        md = gen.generate_markdown(sample_papers)
        assert "# 论文下载报告" in md
        assert "Paper Alpha" in md
        assert "arxiv" in md
        assert "✅" in md or "success" in md

    def test_export_markdown(self, sample_papers, tmp_path):
        """导出 Markdown 文件."""
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.export_markdown(sample_papers)
        assert os.path.exists(path)
        assert Path(path).read_text(encoding="utf-8").startswith("#")

    def test_export_jsonl(self, sample_papers, tmp_path):
        """导出 JSONL."""
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.export_jsonl(sample_papers)
        assert os.path.exists(path)
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 3
        json.loads(lines[0])  # 每行合法 JSON

    def test_export_bibtex(self, sample_papers, tmp_path):
        """导出 BibTeX."""
        gen = ReportGenerator(output_dir=str(tmp_path))
        path = gen.export_bibtex(sample_papers)
        assert os.path.exists(path)
        content = Path(path).read_text()
        assert "@article" in content
        assert "Paper Alpha" in content


# ═══════════════════════════════════════════════════════════════════
# Core Downloader 集成测试
# ═══════════════════════════════════════════════════════════════════

class TestCoreIntegration:
    """core/downloader 的 error_handler + metrics 集成."""

    def test_metrics_collected_during_operations(self, tmp_path):
        """搜索和下载后指标更新."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "timeout": {"search": 5, "download": 10},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        # Mock 搜索
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {"title": "Test", "authors": ["A"], "year": "2024", "source": "arxiv"}
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        papers = dl.search("test")
        stats = dl.metrics.get_stats()
        assert stats["search"]["count"] == 1

    def test_retry_decorator_integration(self):
        """retry_on_error 装饰器正常工作."""
        call_count = [0]

        @retry_on_error(max_retries=3, delay=0.001, backoff=1)
        def might_fail():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return call_count[0]

        result = might_fail()
        assert result == 3

    def test_error_handler_context_manager(self):
        """ErrorHandler 上下文管理器."""
        handler = ErrorHandler(max_retries=2, delay=0.01)
        call_count = [0]

        def inside_handler():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("retry me")

        # 通过 run() 使用
        result = handler.run(inside_handler)
        assert result is None  # 函数返回 None
        assert call_count[0] == 2

    def test_metrics_export_json(self):
        """指标导出 JSON."""
        mc = MetricsCollector()
        mc.record_search(duration=1.0)
        mc.record_download(success=True, size_bytes=5000)
        j = mc.export_json()
        assert json.loads(j)["search"]["count"] == 1


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
