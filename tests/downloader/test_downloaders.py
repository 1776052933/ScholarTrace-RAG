"""
test_downloaders.py — 下载器与 PDF 处理模块单元测试.

覆盖 base_downloader、http_downloader、arxiv_downloader、
pdf_processor 和 download_manager。
"""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from paper_downloader.src.downloaders.base_downloader import Downloader
from paper_downloader.src.downloaders.http_downloader import HTTPDownloader
from paper_downloader.src.downloaders.arxiv_downloader import ArxivPDFDownloader
from paper_downloader.src.downloaders.pdf_processor import PDFProcessor
from paper_downloader.src.downloaders.download_manager import (
    DownloadManager,
    DownloadTask,
    DownloadStatus,
)

# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

SAMPLE_CONFIG: Dict[str, Any] = {
    "concurrency": {"max_downloads": 2, "request_delay": 0.0},
    "timeout": {"search": 5, "download": 10, "connection": 3},
    "proxy": {"enabled": False},
    "retry": {"max_attempts": 2, "backoff_factor": 1},
    "download": {"path": "./papers"},
    "logging": {"level": "WARNING", "file": "/dev/null"},
}


def make_minimal_pdf(path: Path) -> Path:
    """创建一个最小的合法 PDF 文件用于测试."""
    # 最小 PDF 结构：header + objects + xref + trailer + %%EOF
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n"
        b"%%EOF\n"
    )
    path.write_bytes(content)
    return path


class _StubDownloader(Downloader):
    """用于测试抽象基类的具体实现."""
    DOWNLOADER_NAME = "stub"

    def download(self, url, save_path="./papers", filename=None, **kwargs):
        p = Path(save_path) / (filename or "stub.pdf")
        p.parent.mkdir(parents=True, exist_ok=True)
        make_minimal_pdf(p)
        return p


# ═══════════════════════════════════════════════════════════════════
# base_downloader 测试
# ═══════════════════════════════════════════════════════════════════

class TestBaseDownloader:
    """Downloader 抽象基类测试."""

    def test_validate_pdf_valid(self, tmp_path):
        """有效 PDF 文件."""
        pdf = make_minimal_pdf(tmp_path / "test.pdf")
        assert Downloader.validate_pdf(pdf) is True

    def test_validate_pdf_not_exists(self):
        """文件不存在."""
        assert Downloader.validate_pdf("/nonexistent/file.pdf") is False

    def test_validate_pdf_empty_file(self, tmp_path):
        """空文件."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        assert Downloader.validate_pdf(empty) is False

    def test_validate_pdf_not_a_pdf(self, tmp_path):
        """非 PDF 文件."""
        txt = tmp_path / "not.pdf"
        txt.write_text("Just a text file")
        assert Downloader.validate_pdf(txt) is False

    def test_validate_pdf_deep(self, tmp_path):
        """深度校验（头部+尾部）."""
        pdf = make_minimal_pdf(tmp_path / "valid.pdf")
        assert Downloader.validate_pdf_deep(pdf) is True

    def test_validate_pdf_deep_no_eof(self, tmp_path):
        """缺少 %%EOF 的文件."""
        corrupted = tmp_path / "corrupted.pdf"
        corrupted.write_bytes(b"%PDF-1.4\nincomplete content\n")
        assert Downloader.validate_pdf_deep(corrupted) is False

    def test_extract_filename_from_url(self):
        """从 URL 推断文件名."""
        assert Downloader._extract_filename_from_url("https://x.org/path/paper.pdf") == "paper.pdf"
        assert Downloader._extract_filename_from_url("https://x.org/noext") == "paper.pdf"

    def test_sanitize_path(self, tmp_path):
        """路径确保父目录存在."""
        p = Downloader._sanitize_path(tmp_path / "sub" / "file.pdf")
        assert p.parent.exists()

    def test_cannot_instantiate_abstract(self):
        """抽象类无法直接实例化."""
        with pytest.raises(TypeError):
            Downloader()  # type: ignore[abstract]

    def test_stub_implements_download(self, tmp_path):
        """实现了抽象方法后可正常使用."""
        d = _StubDownloader()
        p = d.download("http://x.org/test.pdf", save_path=str(tmp_path))
        assert p is not None
        assert p.exists()
        assert Downloader.validate_pdf(p)


# ═══════════════════════════════════════════════════════════════════
# HTTPDownloader 测试
# ═══════════════════════════════════════════════════════════════════

class TestHTTPDownloader:
    """HTTPDownloader 测试."""

    def setup_method(self):
        self.downloader = HTTPDownloader(SAMPLE_CONFIG)

    def teardown_method(self):
        self.downloader.close()

    def test_check_url(self):
        """HEAD 预检."""
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": "12345",
            "Content-Disposition": 'attachment; filename="paper.pdf"',
        }
        mock_resp.url = "https://final.example.org/paper.pdf"
        mock_resp.raise_for_status.return_value = None
        mock_session.head.return_value = mock_resp
        self.downloader._session = mock_session

        info = self.downloader.check_url("https://example.org/paper")
        assert info is not None
        assert info["content_length"] == 12345
        assert info["is_pdf"] is True
        assert "paper.pdf" in info["filename"]  # type: ignore[operator]

    def test_check_url_failure(self):
        """HEAD 失败返回 None."""
        mock_session = MagicMock()
        mock_session.head.side_effect = requests.exceptions.ConnectionError("No route")
        self.downloader._session = mock_session

        info = self.downloader.check_url("https://dead.example.org")
        assert info is None

    def test_download_success(self, tmp_path):
        """成功下载 PDF."""
        pdf_bytes = make_minimal_pdf(tmp_path / "source.pdf").read_bytes()

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/pdf", "Content-Length": str(len(pdf_bytes))}
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [pdf_bytes]
        mock_session.get.return_value = mock_resp
        self.downloader._session = mock_session

        result = self.downloader.download("https://example.org/test.pdf", save_path=str(tmp_path))
        assert result is not None
        assert result.exists()
        assert Downloader.validate_pdf(result)

    def test_download_http_error(self, tmp_path):
        """HTTP 错误."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        self.downloader._session = mock_session

        result = self.downloader.download("https://example.org/missing.pdf", save_path=str(tmp_path))
        assert result is None

    def test_download_timeout(self, tmp_path):
        """下载超时."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Read timeout")
        self.downloader._session = mock_session

        result = self.downloader.download("https://slow.example.org/big.pdf", save_path=str(tmp_path))
        assert result is None

    def test_resume_download(self, tmp_path):
        """断点续传 — Range 头."""
        partial = tmp_path / "partial.pdf"
        partial.write_bytes(b"%PDF-1.4\npartial content\n")

        pdf_bytes = make_minimal_pdf(tmp_path / "full.pdf").read_bytes()
        remaining = pdf_bytes[len(b"%PDF-1.4\npartial content\n"):]

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/pdf", "Content-Range": f"bytes 22-{len(pdf_bytes)-1}/{len(pdf_bytes)}"}
        mock_resp.status_code = 206
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [remaining]
        mock_session.get.return_value = mock_resp
        self.downloader._session = mock_session

        result = self.downloader.download("https://example.org/resume.pdf", save_path=str(tmp_path),
                                           filename="partial", resume=True)
        assert result is not None

    def test_context_manager(self):
        """上下文管理器自动关闭 session."""
        dl = HTTPDownloader(SAMPLE_CONFIG)
        with dl as d:
            assert d is dl
        # 退出后 session 已关闭

    @staticmethod
    def test_get_content_length():
        """解析 Content-Length."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": "2048"}
        assert HTTPDownloader._get_content_length(mock_resp) == 2048

    @staticmethod
    def test_get_content_length_from_range():
        """从 Content-Range 解析."""
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Range": "bytes 0-1023/10240"}
        assert HTTPDownloader._get_content_length(mock_resp) == 10240


# ═══════════════════════════════════════════════════════════════════
# ArxivPDFDownloader 测试
# ═══════════════════════════════════════════════════════════════════

class TestArxivPDFDownloader:
    """ArxivPDFDownloader 测试."""

    def setup_method(self):
        self.downloader = ArxivPDFDownloader(SAMPLE_CONFIG)

    def teardown_method(self):
        self.downloader.close()

    # ── ID 提取 ──────────────────────────────────────────────

    def test_extract_id_pure(self):
        """纯 arXiv ID."""
        assert ArxivPDFDownloader._extract_arxiv_id("2401.00001") == "2401.00001"

    def test_extract_id_url(self):
        """从 arXiv URL."""
        url = "https://arxiv.org/abs/2401.00001"
        assert ArxivPDFDownloader._extract_arxiv_id(url) == "2401.00001"

    def test_extract_id_pdf_url(self):
        """从 PDF URL."""
        url = "https://arxiv.org/pdf/2401.00001.pdf"
        assert ArxivPDFDownloader._extract_arxiv_id(url) == "2401.00001"

    def test_extract_id_with_version(self):
        """带版本号."""
        assert ArxivPDFDownloader._extract_arxiv_id("2401.00001v3") == "2401.00001v3"

    def test_extract_id_invalid(self):
        """无效输入."""
        assert ArxivPDFDownloader._extract_arxiv_id("not an arxiv id") is None
        assert ArxivPDFDownloader._extract_arxiv_id("") is None

    # ── 下载 ─────────────────────────────────────────────────

    def test_download_via_http_fallback(self, tmp_path):
        """API 回退到 HTTP 下载."""
        # Mock HTTP downloader
        mock_http = MagicMock()
        pdf = make_minimal_pdf(tmp_path / "downloaded.pdf")
        mock_http.download.return_value = pdf
        self.downloader._http_downloader = mock_http

        result = self.downloader.download("2401.00001", save_path=str(tmp_path), use_api=False)
        assert result is not None
        mock_http.download.assert_called_once()

    def test_download_invalid_id(self):
        """无效 ID."""
        result = self.downloader.download("invalid text", save_path="./papers")
        assert result is None

    def test_context_manager(self):
        """上下文管理器."""
        dl = ArxivPDFDownloader(SAMPLE_CONFIG)
        with dl as d:
            assert d is dl


# ═══════════════════════════════════════════════════════════════════
# PDFProcessor 测试
# ═══════════════════════════════════════════════════════════════════

class TestPDFProcessor:
    """PDFProcessor 工具集测试."""

    def test_extract_metadata_valid(self, tmp_path):
        """有效 PDF 元数据."""
        pdf = make_minimal_pdf(tmp_path / "test.pdf")
        meta = PDFProcessor.extract_metadata(pdf)
        assert meta["is_valid_pdf"] is True
        assert meta["file_size"] > 0
        assert meta["pages"] >= 0  # 0 if PyPDF2 not available

    def test_extract_metadata_missing_file(self):
        """文件不存在."""
        meta = PDFProcessor.extract_metadata("/nonexistent.pdf")
        assert meta["is_valid_pdf"] is False

    def test_rename_pdf(self, tmp_path):
        """按模板重命名."""
        pdf = make_minimal_pdf(tmp_path / "old.pdf")
        new_path = PDFProcessor.rename_pdf(
            pdf, title="Attention Is All You Need",
            authors="Ashish Vaswani; Noam Shazeer",
            year="2017",
            keep_original=True,
        )
        assert new_path is not None
        assert "Vaswani" in str(new_path)
        assert "2017" in str(new_path)
        assert new_path.exists()
        assert pdf.exists()  # 原文件保留

    def test_rename_pdf_move_mode(self, tmp_path):
        """重命名并移动（不保留原文件）."""
        pdf = make_minimal_pdf(tmp_path / "old2.pdf")
        new_path = PDFProcessor.rename_pdf(
            pdf, title="Deep Learning", authors="Geoffrey Hinton",
            year="2015", keep_original=False,
        )
        assert new_path is not None
        assert not Path(tmp_path / "old2.pdf").exists()  # 原文件已移动

    def test_rename_pdf_missing_file(self):
        """文件不存在."""
        result = PDFProcessor.rename_pdf("/nonexistent.pdf", title="X")
        assert result is None

    def test_check_corrupted_valid(self, tmp_path):
        """有效 PDF 不应被判为损坏."""
        pdf = make_minimal_pdf(tmp_path / "valid.pdf")
        assert PDFProcessor.check_corrupted(pdf) is False

    def test_check_corrupted_empty(self, tmp_path):
        """空文件判为损坏."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        assert PDFProcessor.check_corrupted(empty) is True

    def test_check_corrupted_missing(self):
        """缺失文件判为损坏."""
        assert PDFProcessor.check_corrupted("/nonexistent.pdf") is True

    def test_check_corrupted_bad_header(self, tmp_path):
        """坏的头部."""
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"Not a PDF file at all!")
        assert PDFProcessor.check_corrupted(bad) is True

    def test_get_page_count(self, tmp_path):
        """获取页数."""
        pdf = make_minimal_pdf(tmp_path / "pages.pdf")
        count = PDFProcessor.get_page_count(pdf)
        # count 可能是 int (PyPDF2 存在) 或 None (PyPDF2 不存在)
        if count is not None:
            assert count >= 1

    def test_extract_text(self, tmp_path):
        """提取首页文本."""
        pdf = make_minimal_pdf(tmp_path / "text.pdf")
        text = PDFProcessor.extract_text(pdf, max_pages=1)
        # PyPDF2 不存在时返回 None
        if text is not None:
            assert isinstance(text, str)

    def test_sanitize_name(self):
        """文件名清理."""
        assert PDFProcessor._sanitize_name('bad:name<test>.pdf') == "bad_name_test.pdf"


# ═══════════════════════════════════════════════════════════════════
# DownloadManager 测试
# ═══════════════════════════════════════════════════════════════════

class TestDownloadManager:
    """DownloadManager 并发下载管理器测试."""

    def setup_method(self):
        self.manager = DownloadManager(SAMPLE_CONFIG, max_workers=2)

    def teardown_method(self):
        self.manager.close()

    # ── 任务管理 ─────────────────────────────────────────────

    def test_add_task(self):
        """添加单个任务."""
        key = self.manager.add_task("https://example.org/paper.pdf", title="Test Paper")
        assert self.manager.total_tasks == 1
        task = self.manager.get_task(key)
        assert task is not None
        assert task.title == "Test Paper"
        assert task.status == DownloadStatus.PENDING

    def test_add_tasks_batch(self):
        """批量添加任务."""
        items = [
            {"url": "https://x.org/a.pdf", "title": "Paper A", "doi": "10.a"},
            {"url": "https://x.org/b.pdf", "title": "Paper B", "doi": "10.b"},
            {"url": "https://x.org/c.pdf", "title": "Paper C"},
        ]
        keys = self.manager.add_tasks_batch(items)
        assert len(keys) == 3
        assert self.manager.total_tasks == 3

    def test_add_duplicate_task(self):
        """重复任务警告但不重复添加."""
        self.manager.add_task("https://x.org/dup.pdf", title="First")
        self.manager.add_task("https://x.org/dup.pdf", title="Second")
        # 仍然是同一个任务
        assert self.manager.total_tasks == 1

    def test_get_all_tasks(self):
        """获取全部任务."""
        self.manager.add_task("https://a.org/1.pdf")
        self.manager.add_task("https://b.org/2.pdf")
        tasks = self.manager.get_all_tasks()
        assert len(tasks) == 2

    # ── 执行下载 ─────────────────────────────────────────────

    def test_run_all_success(self, tmp_path):
        """并发下载全部成功."""
        # Mock HTTPDownloader.download
        self.manager._downloader = MagicMock()
        pdf = make_minimal_pdf(tmp_path / "downloaded.pdf")

        def fake_download(url, save_path="./papers", filename=None, **kw):
            out = tmp_path / f"{filename or 'result'}.pdf"
            make_minimal_pdf(out)
            return out

        self.manager._downloader.download.side_effect = fake_download

        self.manager.add_task("https://x.org/a.pdf", title="Paper A", save_dir=str(tmp_path), filename="a")
        self.manager.add_task("https://x.org/b.pdf", title="Paper B", save_dir=str(tmp_path), filename="b")

        results = self.manager.run_all()
        assert len(results) == 2
        assert all(t.status == DownloadStatus.COMPLETED for t in results)
        assert self.manager.completed_count == 2
        assert self.manager.failed_count == 0

    def test_run_all_with_failures(self, tmp_path):
        """部分下载失败."""
        self.manager._downloader = MagicMock()
        self.manager._max_retries = 1  # 不重试，直接失败

        def flaky_download(url, save_path="./papers", filename=None, **kw):
            if filename == "b":  # 任务 B 始终失败
                return None
            out = tmp_path / f"{filename or 'result'}.pdf"
            make_minimal_pdf(out)
            return out

        self.manager._downloader.download.side_effect = flaky_download

        self.manager.add_task("https://x.org/a.pdf", title="A", save_dir=str(tmp_path), filename="a")
        self.manager.add_task("https://x.org/b.pdf", title="B", save_dir=str(tmp_path), filename="b")
        self.manager.add_task("https://x.org/c.pdf", title="C", save_dir=str(tmp_path), filename="c")

        results = self.manager.run_all()
        completed = [t for t in results if t.status == DownloadStatus.COMPLETED]
        failed = [t for t in results if t.status == DownloadStatus.FAILED]
        assert len(completed) == 2
        assert len(failed) == 1

    def test_run_all_with_retries(self, tmp_path):
        """重试后成功."""
        self.manager._downloader = MagicMock()
        self.manager._max_retries = 3
        self.manager._backoff_factor = 0  # 不真实延迟

        call_count = {"n": 0}

        def retry_then_succeed(url, save_path="./papers", filename=None, **kw):
            call_count["n"] += 1
            if call_count["n"] < 3:  # 前两次失败
                raise requests.exceptions.Timeout("Timeout")
            out = tmp_path / f"{filename or 'result'}.pdf"
            make_minimal_pdf(out)
            return out

        self.manager._downloader.download.side_effect = retry_then_succeed

        self.manager.add_task("https://x.org/retry.pdf", title="Retry", save_dir=str(tmp_path), filename="retry")
        results = self.manager.run_all()
        task = results[0]
        assert task.status == DownloadStatus.COMPLETED
        assert task.attempt_count >= 3  # type: ignore[operator]

    def test_run_all_exhausted_retries(self, tmp_path):
        """耗尽重试次数."""
        self.manager._downloader = MagicMock()
        self.manager._max_retries = 2
        self.manager._backoff_factor = 0

        self.manager._downloader.download.side_effect = requests.exceptions.ConnectionError("Fail")

        self.manager.add_task("https://x.org/fail.pdf", title="Fail", save_dir=str(tmp_path), filename="f")
        results = self.manager.run_all()
        assert results[0].status == DownloadStatus.FAILED
        assert results[0].attempt_count == 2

    # ── 历史记录 ─────────────────────────────────────────────

    def test_save_and_load_history(self, tmp_path):
        """保存和加载下载历史."""
        history_path = tmp_path / "history.json"
        mgr = DownloadManager(SAMPLE_CONFIG, history_file=str(history_path))
        mgr.add_task("https://x.org/a.pdf", title="Paper A")
        mgr._save_history()

        history = mgr.get_history()
        assert len(history) == 1
        assert history[0]["title"] == "Paper A"

        mgr.close()

    def test_get_history_empty(self):
        """无历史文件."""
        mgr = DownloadManager(SAMPLE_CONFIG, history_file="/nonexistent/path/history.json")
        assert mgr.get_history() == []
        mgr.close()

    def test_clear_history(self, tmp_path):
        """清空历史."""
        history_path = tmp_path / "to_clear.json"
        mgr = DownloadManager(SAMPLE_CONFIG, history_file=str(history_path))
        mgr.add_task("https://x.org/del.pdf")
        mgr._save_history()
        assert len(mgr.get_history()) == 1
        mgr.clear_history()
        assert mgr.get_history() == []
        mgr.close()

    # ── 统计 ─────────────────────────────────────────────────

    def test_get_stats(self, tmp_path):
        """下载统计."""
        self.manager._downloader = MagicMock()

        def fake_dl(url, save_path="./papers", filename=None, **kw):
            out = tmp_path / f"{filename or 'r'}.pdf"
            make_minimal_pdf(out)
            return out

        self.manager._downloader.download.side_effect = fake_dl

        self.manager.add_task("https://x.org/s1.pdf", title="S1", save_dir=str(tmp_path), filename="s1")
        self.manager.add_task("https://x.org/s2.pdf", title="S2", save_dir=str(tmp_path), filename="s2")
        self.manager.run_all()

        stats = self.manager.get_stats()
        assert stats["total"] == 2
        assert stats["completed"] == 2
        assert stats["failed"] == 0
        assert stats["total_bytes"] > 0

    # ── 进度回调 ─────────────────────────────────────────────

    def test_progress_callback(self, tmp_path):
        """进度回调在每个任务完成时触发."""
        self.manager._downloader = MagicMock()

        def fake_dl(url, save_path="./papers", filename=None, **kw):
            out = tmp_path / f"{filename or 'r'}.pdf"
            make_minimal_pdf(out)
            return out

        self.manager._downloader.download.side_effect = fake_dl

        completed: List[DownloadTask] = []
        self.manager.set_progress_callback(lambda t: completed.append(t))

        self.manager.add_task("https://x.org/c1.pdf", title="C1", save_dir=str(tmp_path), filename="c1")
        self.manager.add_task("https://x.org/c2.pdf", title="C2", save_dir=str(tmp_path), filename="c2")
        self.manager.run_all()

        assert len(completed) == 2

    # ── 上下文管理器 ─────────────────────────────────────────

    def test_context_manager(self, tmp_path):
        """上下文管理器自动保存和关闭."""
        history_path = tmp_path / "ctx_history.json"
        with DownloadManager(SAMPLE_CONFIG, history_file=str(history_path)) as mgr:
            mgr.add_task("https://x.org/ctx.pdf")
            mgr._save_history()
        # 退出后历史已保存
        assert history_path.exists()

    # ── 任务超时 ─────────────────────────────────────────────

    def test_task_timeout_calculation(self):
        """超时计算含重试缓冲."""
        timeout = self.manager._get_task_timeout()
        assert timeout > int(SAMPLE_CONFIG["timeout"]["download"])


# ═══════════════════════════════════════════════════════════════════
# DownloadTask 数据结构
# ═══════════════════════════════════════════════════════════════════

class TestDownloadTask:
    """DownloadTask dataclass 测试."""

    def test_default_values(self):
        """默认值."""
        t = DownloadTask(url="https://x.org/test.pdf")
        assert t.status == DownloadStatus.PENDING
        assert t.attempt_count == 0
        assert t.created_at is not None

    def test_full_construction(self):
        """完整构造."""
        t = DownloadTask(
            url="https://x.org/p.pdf",
            title="My Paper",
            authors="John Smith",
            year="2024",
            doi="10.1234/test",
            arxiv_id="2401.00001",
            status=DownloadStatus.COMPLETED,
            pdf_path="/papers/my_paper.pdf",
            file_size=1024000,
        )
        assert t.title == "My Paper"
        assert t.doi == "10.1234/test"
        assert t.file_size == 1024000


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
