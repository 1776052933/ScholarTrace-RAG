"""
test_interface.py — 统一极简接口测试.

覆盖 download_pdf、batch_download_pdf、search_papers 三个接口函数.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from paper_downloader.src.interface import (
    download_pdf,
    batch_download_pdf,
    search_papers,
    _paper_to_dict,
    _resolve_engines,
    _get_downloader,
)
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import PaperNotFoundError


# ═══════════════════════════════════════════════════════════════════
# 工具函数测试
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:
    """工具函数测试."""

    def test_resolve_engines_auto(self):
        """auto → 全引擎."""
        assert _resolve_engines("auto") == [
            "arxiv", "openalex", "crossref", "google_scholar",
        ]

    def test_resolve_engines_specific(self):
        """指定引擎."""
        assert _resolve_engines("arxiv") == ["arxiv"]
        assert _resolve_engines("crossref") == ["crossref"]
        assert _resolve_engines("scholar") == ["google_scholar"]
        assert _resolve_engines("openalex") == ["openalex"]

    def test_resolve_engines_unknown(self):
        """未知引擎回退 auto."""
        assert _resolve_engines("unknown") == [
            "arxiv", "openalex", "crossref", "google_scholar",
        ]

    def test_paper_to_dict(self):
        """Paper → dict 转换."""
        p = Paper(
            title="Test", authors=["Alice", "Bob"], year="2024",
            doi="10.1234/test", arxiv_id="2401.00001",
            abstract="An abstract.", pdf_url="https://arxiv.org/pdf/2401.00001",
            source="arxiv", citation_count=42, journal="Nature",
            pdf_path="/tmp/test.pdf", file_size=1024,
        )
        d = _paper_to_dict(p)
        assert d["title"] == "Test"
        assert d["authors"] == ["Alice", "Bob"]
        assert d["first_author"] == "Alice"
        assert d["doi"] == "10.1234/test"
        assert d["citation_count"] == 42
        assert d["file_path"] == "/tmp/test.pdf"


# ═══════════════════════════════════════════════════════════════════
# download_pdf 测试
# ═══════════════════════════════════════════════════════════════════

def make_minimal_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<</Size 1>>\nstartxref\n0\n%%EOF\n")
    return path


class TestDownloadPdf:
    """download_pdf 接口测试."""

    def setup_method(self):
        import paper_downloader.src.interface as iface
        iface._downloader = None

    def test_empty_title_returns_error(self):
        """空标题返回错误."""
        result = download_pdf("")
        assert result["success"] is False
        assert "不能为空" in result["error"]

    def test_success_flow(self, tmp_path):
        """成功下载流程."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        # Mock 搜索
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [{
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"], "year": "2017",
            "doi": "10.5555/3295222", "arxiv_id": "1706.03762",
            "pdf_url": "https://arxiv.org/pdf/1706.03762",
            "source": "arxiv", "citation_count": 100000,
            "url": "https://arxiv.org/abs/1706.03762",
        }]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # Mock 下载
        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Attention Is All You Need"
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        pdf = tmp_path / "attention.pdf"
        make_minimal_pdf(pdf)
        mock_task.pdf_path = str(pdf)
        mock_task.file_size = pdf.stat().st_size
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        result = download_pdf("Attention Is All You Need", output_dir=str(tmp_path))
        assert result["success"] is True
        assert result["file_path"] is not None
        assert "attention" in str(result["file_path"])
        assert result["paper_info"] is not None
        assert result["paper_info"]["title"] == "Attention Is All You Need"
        assert result["engine_used"] == "arxiv"

    def test_callback_invoked(self, tmp_path):
        """进度回调被调用."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [{
            "title": "Test", "authors": ["A"], "year": "2024",
            "pdf_url": "https://x.org/test.pdf", "source": "arxiv",
            "url": "https://x.org/test",
        }]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Test"
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        pdf = tmp_path / "test.pdf"
        make_minimal_pdf(pdf)
        mock_task.pdf_path = str(pdf)
        mock_task.file_size = pdf.stat().st_size
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        progress_records: List[tuple] = []

        def cb(progress, msg):
            progress_records.append((progress, msg))

        download_pdf("Test", output_dir=str(tmp_path), callback=cb)
        assert len(progress_records) >= 2  # 至少搜索和下载两次回调

    def test_engine_fallback(self, tmp_path):
        """引擎回退 — arxiv 失败后用 crossref."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv", "crossref"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        # 第一次搜索返回空（arxiv），第二次成功（crossref）
        call_count = [0]
        original_search = dl.search

        def search_side_effect(title, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PaperNotFoundError("arxiv not found")
            return [Paper(title=title, source="crossref", pdf_url="https://x.org/pdf")]

        dl.search = search_side_effect

        # Mock download — task title 匹配 Paper(title=query)
        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Test Paper"  # 匹配搜索返回的 Paper 标题
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        pdf = tmp_path / "fallback.pdf"
        make_minimal_pdf(pdf)
        mock_task.pdf_path = str(pdf)
        mock_task.file_size = pdf.stat().st_size
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        result = download_pdf("Test Paper", output_dir=str(tmp_path))
        assert result["success"] is True


# ═══════════════════════════════════════════════════════════════════
# batch_download_pdf 测试
# ═══════════════════════════════════════════════════════════════════

class TestBatchDownloadPdf:
    """batch_download_pdf 接口测试."""

    def setup_method(self):
        import paper_downloader.src.interface as iface
        iface._downloader = None

    def test_empty_titles(self):
        """空列表."""
        assert batch_download_pdf([]) == []

    def test_batch_download(self, tmp_path):
        """批量下载."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 1},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 2, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        # Mock 搜索 — 每次都返回同一结果
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [{
            "title": "Test Paper", "authors": ["A"],
            "year": "2024", "pdf_url": "https://x.org/test.pdf",
            "source": "arxiv", "url": "https://x.org/test",
        }]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Test Paper"
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"

        def make_pdf_task():
            pdf = tmp_path / f"batch_{id(mock_task)}.pdf"
            make_minimal_pdf(pdf)
            mock_task.pdf_path = str(pdf)
            mock_task.file_size = 100
            mock_task.completed_at = "2024-01-01T00:00:00"
            return mock_task

        mock_mgr.run_all.return_value = [make_pdf_task()]
        dl._download_manager = mock_mgr

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        progress: List[tuple] = []

        def cb(cur, tot, msg):
            progress.append((cur, tot, msg))

        results = batch_download_pdf(
            ["Paper A", "Paper B", "Paper C"],
            output_dir=str(tmp_path),
            max_concurrent=2,
            callback=cb,
        )
        assert len(results) == 3
        assert len(progress) > 0


# ═══════════════════════════════════════════════════════════════════
# search_papers 测试
# ═══════════════════════════════════════════════════════════════════

class TestSearchPapers:
    """search_papers 接口测试."""

    def setup_method(self):
        import paper_downloader.src.interface as iface
        iface._downloader = None

    def test_search_returns_dicts(self):
        """返回纯字典列表."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 5},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {"title": "Paper 1", "authors": ["A"], "year": "2024", "source": "arxiv"},
            {"title": "Paper 2", "authors": ["B"], "year": "2023", "source": "arxiv"},
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        results = search_papers("test", max_results=5)
        assert len(results) == 2
        assert isinstance(results[0], dict)
        assert results[0]["title"] == "Paper 1"
        assert results[0]["authors"] == ["A"]

    def test_search_empty_results(self):
        """空结果."""
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 5},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = []
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        import paper_downloader.src.interface as iface
        iface._downloader = dl

        results = search_papers("nonexistent_xyz")
        assert results == []


# ═══════════════════════════════════════════════════════════════════
# 顶层导入测试
# ═══════════════════════════════════════════════════════════════════

class TestTopLevelImports:
    """验证 __init__.py 中的顶层导入."""

    def test_download_pdf_importable(self):
        """download_pdf 可从顶层导入."""
        from paper_downloader import download_pdf
        assert callable(download_pdf)

    def test_batch_download_pdf_importable(self):
        """batch_download_pdf 可从顶层导入."""
        from paper_downloader import batch_download_pdf
        assert callable(batch_download_pdf)

    def test_search_papers_dict_importable(self):
        """search_papers_dict 可从顶层导入."""
        from paper_downloader import search_papers_dict
        assert callable(search_papers_dict)


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
