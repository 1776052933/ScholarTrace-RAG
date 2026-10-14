"""
test_async_batch.py — 异步/批量处理模块测试.

覆盖 CallbackManager、ProgressTracker、AsyncPaperDownloader、
BatchProcessor 和增强的 batch_download。
"""

import asyncio
import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from paper_downloader.src.core.callback_manager import CallbackManager, CallbackEvent
from paper_downloader.src.core.progress_tracker import ProgressTracker
from paper_downloader.src.core.batch_processor import BatchProcessor
from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import PaperNotFoundError


# ═══════════════════════════════════════════════════════════════════
# CallbackManager 测试
# ═══════════════════════════════════════════════════════════════════

class TestCallbackManager:
    """回调管理器测试."""

    def test_register_and_trigger(self):
        """注册并触发回调."""
        mgr = CallbackManager()
        results: List[str] = []

        mgr.register(CallbackEvent.ON_SEARCH_COMPLETE, lambda papers: results.extend(papers))
        mgr.trigger(CallbackEvent.ON_SEARCH_COMPLETE, ["paper1", "paper2"])

        assert results == ["paper1", "paper2"]

    def test_decorator_syntax(self):
        """装饰器注册."""
        mgr = CallbackManager()
        called: List[str] = []

        @mgr.on(CallbackEvent.ON_DOWNLOAD_COMPLETE)
        def handler(paper):
            called.append(paper)

        mgr.trigger(CallbackEvent.ON_DOWNLOAD_COMPLETE, "test_paper")
        assert called == ["test_paper"]

    def test_multiple_callbacks(self):
        """同一事件多个回调."""
        mgr = CallbackManager()
        acc: List[int] = []

        mgr.register(CallbackEvent.ON_BATCH_START, lambda total: acc.append(total))
        mgr.register(CallbackEvent.ON_BATCH_START, lambda total: acc.append(total * 2))
        mgr.trigger(CallbackEvent.ON_BATCH_START, 10)

        assert acc == [10, 20]

    def test_callback_error_not_propagated(self):
        """回调异常不中断后续回调."""
        mgr = CallbackManager()
        acc: List[str] = []

        def bad(_): raise RuntimeError("Boom!")

        def good(_): acc.append("ok")

        mgr.register(CallbackEvent.ON_ERROR, bad)
        mgr.register(CallbackEvent.ON_ERROR, good)
        mgr.trigger(CallbackEvent.ON_ERROR, "test")

        assert acc == ["ok"]

    def test_unregister(self):
        """取消注册."""
        mgr = CallbackManager()
        cb = lambda x: None
        mgr.register(CallbackEvent.ON_SEARCH_START, cb)
        assert mgr.count(CallbackEvent.ON_SEARCH_START) == 1

        mgr.unregister(CallbackEvent.ON_SEARCH_START, cb)
        assert mgr.count(CallbackEvent.ON_SEARCH_START) == 0

    def test_clear(self):
        """清空回调."""
        mgr = CallbackManager()
        mgr.register(CallbackEvent.ON_ERROR, lambda x: None)
        mgr.register(CallbackEvent.ON_ERROR, lambda x: None)
        mgr.clear(CallbackEvent.ON_ERROR)
        assert mgr.count(CallbackEvent.ON_ERROR) == 0

    def test_clear_all(self):
        """清空所有."""
        mgr = CallbackManager()
        mgr.register(CallbackEvent.ON_ERROR, lambda x: None)
        mgr.register(CallbackEvent.ON_SEARCH_START, lambda x: None)
        mgr.clear()
        assert mgr.count() == 0

    def test_count(self):
        """计数."""
        mgr = CallbackManager()
        mgr.register(CallbackEvent.ON_ERROR, lambda x: None)
        mgr.register(CallbackEvent.ON_ERROR, lambda x: None)
        assert mgr.count(CallbackEvent.ON_ERROR) == 2

    def test_register_not_callable(self):
        """注册非可调用对象报错."""
        mgr = CallbackManager()
        with pytest.raises(ValueError):
            mgr.register(CallbackEvent.ON_ERROR, "not_callable")  # type: ignore[arg-type]

    def test_trigger_safe_collects_errors(self):
        """trigger_safe 收集异常."""
        mgr = CallbackManager()

        def bad(_): raise ValueError("err")

        mgr.register(CallbackEvent.ON_ERROR, bad)
        errors = mgr.trigger_safe(CallbackEvent.ON_ERROR, "x")
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

    def test_list_callbacks(self):
        """列出回调."""
        mgr = CallbackManager()

        def my_handler(_): pass

        mgr.register(CallbackEvent.ON_SEARCH_COMPLETE, my_handler)
        info = mgr.list_callbacks(CallbackEvent.ON_SEARCH_COMPLETE)
        assert "my_handler" in info["on_search_complete"]


# ═══════════════════════════════════════════════════════════════════
# ProgressTracker 测试
# ═══════════════════════════════════════════════════════════════════

class TestProgressTracker:
    """进度跟踪器测试."""

    def test_basic_progress(self):
        """基本进度追踪."""
        tracker = ProgressTracker(total=10)
        tracker.start()
        for _ in range(5):
            tracker.update(message="task")
        assert tracker.current == 5
        assert tracker.progress == 0.5
        assert tracker.percentage == 50.0

    def test_failed_count(self):
        """失败计数."""
        tracker = ProgressTracker(total=5)
        tracker.start()
        tracker.update(message="ok")
        tracker.update(success=False, message="fail")
        assert tracker.current == 1
        assert tracker.failed == 1

    def test_skip(self):
        """跳过计数."""
        tracker = ProgressTracker(total=10)
        tracker.start()
        tracker.skip(count=3, reason="already done")
        assert tracker.skipped == 3
        assert tracker.current == 3

    def test_elapsed_and_eta(self):
        """耗时和ETA."""
        tracker = ProgressTracker(total=100)
        tracker.start()
        tracker._current = 50
        # ETA 需要 elapsed > 0
        import time
        tracker._start_time = time.time() - 10  # 假装已经过了10秒
        assert tracker.elapsed >= 10
        eta = tracker.eta
        assert eta is not None and eta > 0

    def test_progress_callback(self):
        """进度更新回调."""
        tracker = ProgressTracker(total=3)
        progress_data: List[tuple] = []

        @tracker.on_update
        def cb(current, total, msg):
            progress_data.append((current, total, msg))

        tracker.start()
        tracker.update(message="A")
        tracker.update(message="B")

        assert len(progress_data) == 2
        assert progress_data[0][0] == 1  # current

    def test_error_callback(self):
        """错误报告回调."""
        tracker = ProgressTracker(total=3)
        errors: List[str] = []

        @tracker.on_error
        def cb(msg, err):
            errors.append(msg)

        tracker.report_error("something broke")
        assert len(errors) == 1
        assert "something broke" in errors[0]

    def test_finish_summary(self):
        """完成汇总."""
        tracker = ProgressTracker(total=5, description="Test")
        tracker.start()
        tracker.update(message="1")
        tracker.update(message="2")
        tracker.update(success=False, message="3")
        tracker.update(message="4")
        tracker.update(message="5")

        summary = tracker.finish()
        assert summary["total"] == 5
        assert summary["completed"] == 4
        assert summary["failed"] == 1
        assert "elapsed_sec" in summary

    def test_remove_callback(self):
        """移除回调."""
        tracker = ProgressTracker(total=5)
        called: List[int] = []

        def cb(c, t, m): called.append(c)

        tracker.on_update(cb)
        tracker.start()
        tracker.update(message="x")
        assert len(called) == 1

        tracker.remove_callback(cb)
        tracker.update(message="y")
        assert len(called) == 1  # 未再增加

    def test_clear_callbacks(self):
        """清空所有回调."""
        tracker = ProgressTracker(total=5)

        @tracker.on_update
        def cb(c, t, m): pass

        tracker.clear_callbacks()
        tracker.start()
        tracker.update(message="x")  # 不应触发任何回调


# ═══════════════════════════════════════════════════════════════════
# BatchProcessor 测试
# ═══════════════════════════════════════════════════════════════════

class TestBatchProcessor:
    """批量处理器测试."""

    def setup_method(self):
        self.dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 5},
            "download": {"path": "./papers"},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "timeout": {"search": 5, "download": 10, "connection": 3},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })
        self.bp = BatchProcessor(self.dl)

    # ── TXT ─────────────────────────────────────────────────

    def test_from_txt(self, tmp_path):
        """从 TXT 加载."""
        f = tmp_path / "titles.txt"
        f.write_text("Paper A\nPaper B\n# comment\n\nPaper C\n")
        titles = self.bp.from_txt(str(f))
        assert titles == ["Paper A", "Paper B", "Paper C"]

    # ── CSV ─────────────────────────────────────────────────

    def test_from_csv_with_column(self, tmp_path):
        """从 CSV 指定列加载."""
        f = tmp_path / "papers.csv"
        f.write_text("title,author,year\nBERT,Devlin,2019\nGPT-4,OpenAI,2024\n")
        titles = self.bp.from_csv(str(f), column="title")
        assert titles == ["BERT", "GPT-4"]

    def test_from_csv_auto_detect(self, tmp_path):
        """CSV 自动检测标题列."""
        f = tmp_path / "auto.csv"
        f.write_text("id,Title,year\n1,Paper One,2023\n2,Paper Two,2024\n")
        titles = self.bp.from_csv(str(f))
        assert "Paper One" in titles
        assert "Paper Two" in titles

    def test_from_csv_missing_column(self, tmp_path):
        """CSV 列名不存在."""
        f = tmp_path / "bad.csv"
        f.write_text("col1,col2\n1,2\n")
        with pytest.raises(Exception):
            self.bp.from_csv(str(f), column="nonexistent")

    # ── JSON ────────────────────────────────────────────────

    def test_from_json_string_array(self, tmp_path):
        """JSON 字符串数组."""
        f = tmp_path / "list.json"
        json.dump(["Paper X", "Paper Y", "Paper Z"], f.open("w"))
        titles = self.bp.from_json(str(f))
        assert len(titles) == 3
        assert "Paper X" in titles

    def test_from_json_dict_array(self, tmp_path):
        """JSON 对象数组."""
        f = tmp_path / "dicts.json"
        json.dump([
            {"title": "Alpha", "year": 2024},
            {"title": "Beta", "year": 2023},
        ], f.open("w"))
        titles = self.bp.from_json(str(f))
        assert titles == ["Alpha", "Beta"]

    def test_from_json_nested(self, tmp_path):
        """JSON 嵌套格式."""
        f = tmp_path / "nested.json"
        json.dump({"papers": [{"title": "Paper 1"}, {"title": "Paper 2"}]}, f.open("w"))
        titles = self.bp.from_json(str(f))
        assert titles == ["Paper 1", "Paper 2"]

    # ── BibTeX ──────────────────────────────────────────────

    def test_from_bibtex(self, tmp_path):
        """BibTeX 解析."""
        f = tmp_path / "refs.bib"
        f.write_text("""
@article{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam},
  year = {2017},
  doi = {10.xxx/attention},
  journal = {NeurIPS},
}
@inproceedings{devlin2019bert,
  title = {BERT: Pre-training of Deep Bidirectional Transformers},
  author = {Devlin, Jacob and Chang, Ming-Wei},
  year = {2019},
  doi = {10.18653/v1/N19-1423},
}
        """)
        papers = self.bp.from_bibtex(str(f))
        assert len(papers) == 2
        assert papers[0].title == "Attention Is All You Need"
        assert any("Vaswani" in a for a in papers[0].authors)
        assert papers[1].doi == "10.18653/v1/N19-1423"

    # ── from_file 分发 ──────────────────────────────────────

    def test_from_file_auto_dispatch(self, tmp_path):
        """自动根据扩展名分发."""
        f = tmp_path / "list.txt"
        f.write_text("A\nB\n")
        titles = self.bp.from_file(str(f))
        assert titles == ["A", "B"]

    def test_from_file_nonexistent(self):
        """文件不存在."""
        with pytest.raises(Exception):
            self.bp.from_file("/nonexistent/paper_list.txt")

    # ── 报告生成 ────────────────────────────────────────────

    def test_generate_json_report(self, tmp_path):
        """生成 JSON 报告."""
        real_pdf = tmp_path / "a.pdf"
        real_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        papers = [
            Paper(title="Paper A", pdf_path=str(real_pdf), file_size=100),
            Paper(title="Paper B"),
        ]
        files = self.bp.generate_report(papers, output_dir=str(tmp_path), formats=["json"])
        assert "json" in files
        report = json.load(open(files["json"]))
        assert report["summary"]["total"] == 2
        assert report["summary"]["succeeded"] == 1

    def test_generate_csv_report(self, tmp_path):
        """生成 CSV 报告."""
        real_pdf = tmp_path / "b.pdf"
        real_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        papers = [
            Paper(title="Paper A", pdf_path=str(real_pdf)),
            Paper(title="Paper B"),
        ]
        files = self.bp.generate_report(papers, output_dir=str(tmp_path), formats=["csv"])
        assert "csv" in files
        with open(files["csv"], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["status"] == "success"


# ═══════════════════════════════════════════════════════════════════
# AsyncPaperDownloader 测试
# ═══════════════════════════════════════════════════════════════════

class TestAsyncDownloader:
    """异步下载器测试."""

    @pytest.fixture
    def async_dl(self):
        """异步下载器实例."""
        from paper_downloader.src.core.async_downloader import AsyncPaperDownloader
        return AsyncPaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 5},
            "download": {"path": "./papers"},
            "concurrency": {"max_downloads": 2, "request_delay": 0},
            "timeout": {"search": 5, "download": 10, "connection": 3},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

    def test_init(self, async_dl):
        """初始化."""
        assert async_dl._search_semaphore is not None
        assert async_dl._download_semaphore is not None
        assert async_dl.callbacks is not None

    def test_callback_property(self, async_dl):
        """回调管理器可访问."""
        mgr = async_dl.callbacks
        assert isinstance(mgr, CallbackManager)

    def test_make_filename(self, async_dl):
        """文件名生成."""
        p = Paper(title="BERT", authors=["Jacob Devlin"], year="2019")
        name = async_dl._make_filename(p)
        assert "Devlin" in name
        assert "2019" in name
        assert "BERT" in name

    def test_url_to_filename(self, async_dl):
        """URL 提取文件名."""
        assert "paper.pdf" in async_dl._url_to_filename("https://x.org/path/paper.pdf")
        assert async_dl._url_to_filename("https://x.org/noext").endswith(".pdf")

    def test_async_context_manager(self, async_dl):
        """异步上下文管理器."""
        import asyncio

        async def run():
            async with async_dl as dl:
                assert dl is async_dl
            await asyncio.sleep(0)

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_search_async_basic(self, async_dl):
        """异步搜索基本流程."""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {"title": "Test", "authors": ["A"], "year": "2024", "source": "arxiv"}
        ]
        mock_factory.available_engines = ["arxiv"]
        async_dl._search_factory = mock_factory

        papers = await async_dl.search_async("test", max_results=3)
        assert len(papers) == 1
        assert isinstance(papers[0], Paper)

    @pytest.mark.asyncio
    async def test_batch_download_async(self, async_dl, tmp_path):
        """异步批量下载."""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {
                "title": "Paper X",
                "authors": ["Author A"],
                "year": "2024",
                "doi": "10.x",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "source": "arxiv",
            }
        ]
        mock_factory.available_engines = ["arxiv"]
        async_dl._search_factory = mock_factory

        # Mock aiohttp session
        try:
            import aiohttp
            mock_session = AsyncMock()
            async_dl._aio_session = mock_session
        except ImportError:
            pytest.skip("aiohttp not available")

        papers = await async_dl.batch_download_async(
            ["Paper X"],
            output_dir=str(tmp_path),
        )
        assert len(papers) >= 1

    @pytest.mark.asyncio
    async def test_search_with_fallback(self, async_dl):
        """搜索回退 — 失败返回空列表."""
        async def mock_search(*args, **kwargs):
            raise PaperNotFoundError("not found")

        async_dl.search_async = mock_search  # type: ignore[method-assign]
        result = await async_dl._search_with_fallback("xyz", 5, None)
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# 增强 batch_download 测试
# ═══════════════════════════════════════════════════════════════════

class TestEnhancedBatchDownload:
    """增强 batch_download 测试."""

    def test_concurrent_search(self, tmp_path):
        """并发搜索 + 下载."""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "timeout": {"search": 5, "download": 10, "connection": 3},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {
                "title": "Test Paper",
                "authors": ["A"],
                "year": "2024",
                "doi": "10.t",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "source": "arxiv",
                "url": "https://arxiv.org/abs/2401.00001",
            }
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        mock_manager = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Test Paper"
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        mock_task.pdf_path = str(tmp_path / "test.pdf")
        mock_task.file_size = 1024
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_manager.run_all.return_value = [mock_task]
        dl._download_manager = mock_manager

        # 创建测试 PDF
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4\n%%EOF")

        papers = dl.batch_download(
            ["Test Paper"],
            output_dir=str(tmp_path),
            concurrent_searches=1,
            skip_existing=True,
        )
        assert len(papers) == 1

    def test_skip_existing_pdf(self, tmp_path):
        """跳过已有 PDF."""
        # 预创建一个 PDF
        existing = tmp_path / "Test_Paper2024_existing.pdf"
        existing.write_bytes(b"%PDF-1.4\n%%EOF")

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "timeout": {"search": 5, "download": 10},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        # Mock search factory
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {
                "title": "Test Paper",
                "authors": ["A"],
                "year": "2024",
                "doi": "10.t",
                "pdf_url": "https://arxiv.org/pdf/2401.00001",
                "source": "arxiv",
            }
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # 即使已有匹配的 PDF，search 也不会被调用（skip_existing=True）
        papers = dl.batch_download(
            ["Test Paper"],
            output_dir=str(tmp_path),
            skip_existing=True,
        )
        assert len(papers) >= 1


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
