"""
test_core.py — 核心模块测试.

覆盖 Paper 模型、自定义异常、Core PaperDownloader、API 函数。
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
    ValidationError,
    ConfigError,
    SearchError,
)
from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src import api

# ═══════════════════════════════════════════════════════════════════
# Paper 模型测试
# ═══════════════════════════════════════════════════════════════════

class TestPaperModel:
    """Paper 数据模型测试."""

    def test_default_construction(self):
        """默认构造."""
        p = Paper()
        assert p.title == ""
        assert p.authors == []
        assert p.year is None
        assert p.doi is None

    def test_full_construction(self):
        """完整构造."""
        p = Paper(
            title="Test Paper",
            authors=["Alice", "Bob"],
            year="2024",
            doi="10.1234/test",
            arxiv_id="2401.00001",
            abstract="An abstract.",
            pdf_url="https://arxiv.org/pdf/2401.00001",
            url="https://arxiv.org/abs/2401.00001",
            source="arxiv",
            citation_count=42,
            journal="Nature",
            pdf_path="/tmp/test.pdf",
            file_size=1024,
        )
        assert p.title == "Test Paper"
        assert p.citation_count == 42

    def test_from_dict(self):
        """从字典创建."""
        data = {
            "title": "GPT-4",
            "authors": ["OpenAI"],
            "year": "2024",
            "doi": "10.xxx",
            "extra_field": "should be ignored",
        }
        p = Paper.from_dict(data)
        assert p.title == "GPT-4"
        assert p.doi == "10.xxx"
        # extra_field 不应出现在 Paper 属性中
        assert not hasattr(p, "extra_field")

    def test_from_search_result(self):
        """从搜索标准化结果创建."""
        result = {
            "title": "BERT",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": "2019",
            "abstract": "We introduce BERT...",
            "doi": "10.18653/v1/N19-1423",
            "arxiv_id": "1810.04805",
            "pdf_url": "https://arxiv.org/pdf/1810.04805",
            "url": "https://arxiv.org/abs/1810.04805",
            "source": "arxiv",
            "citation_count": 100000,
            "journal": "NAACL",
            "pdf_path": "/tmp/bert.pdf",
            "file_size": 500000,
        }
        p = Paper.from_search_result(result)
        assert p.title == "BERT"
        assert len(p.authors) == 2
        assert p.doi == "10.18653/v1/N19-1423"

    def test_from_search_result_authors_string(self):
        """作者为分号分隔字符串."""
        result = {
            "title": "Test",
            "authors": "Alice; Bob; Charlie",
            "year": "2024",
        }
        p = Paper.from_search_result(result)
        assert p.authors == ["Alice", "Bob", "Charlie"]

    def test_from_search_result_empty(self):
        """空搜索结果."""
        p = Paper.from_search_result({})
        assert p.title == ""
        assert p.authors == []

    def test_to_dict(self):
        """序列化为字典."""
        p = Paper(title="T", authors=["A"], year="2024")
        d = p.to_dict()
        assert d["title"] == "T"
        assert d["authors"] == ["A"]
        assert d["doi"] is None

    def test_to_dict_compact(self):
        """紧凑序列化省略空值."""
        p = Paper(title="T", authors=["A"], year="2024")
        d = p.to_dict_compact()
        assert "doi" not in d
        assert "abstract" not in d
        assert d["title"] == "T"

    def test_to_json(self):
        """JSON 序列化."""
        p = Paper(title="JSON Test", year="2024")
        j = p.to_json()
        assert "JSON Test" in j
        assert json.loads(j)["title"] == "JSON Test"

    def test_to_bibtex(self):
        """BibTeX 输出."""
        p = Paper(
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            year="2017",
            doi="10.xxx/attention",
            journal="NeurIPS",
        )
        bib = p.to_bibtex()
        assert "@article" in bib
        assert "Vaswani" in bib
        assert "10.xxx/attention" in bib

    # ── 属性 ───────────────────────────────────────────────

    def test_has_pdf_true(self, tmp_path):
        """有本地 PDF."""
        pdf = tmp_path / "real.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        p = Paper(pdf_path=str(pdf))
        assert p.has_pdf is True

    def test_has_pdf_false(self):
        """无 PDF."""
        p = Paper(pdf_path="/nonexistent/file.pdf")
        assert p.has_pdf is False

    def test_identifier_priority(self):
        """标识符优先级: DOI > arXiv > URL."""
        p = Paper(doi="10.a", arxiv_id="2401.x", url="https://x.org")
        assert p.identifier == "10.a"

        p2 = Paper(arxiv_id="2401.x", url="https://x.org")
        assert p2.identifier == "2401.x"

        p3 = Paper(url="https://x.org")
        assert p3.identifier == "https://x.org"

    def test_first_author_surname(self):
        """第一作者姓氏."""
        p = Paper(authors=["Geoffrey Hinton", "Yann LeCun"])
        assert p.first_author_surname == "Hinton"

    def test_first_author_surname_empty(self):
        """无作者."""
        p = Paper(authors=[])
        assert p.first_author_surname == "unknown"

    def test_citation_format(self):
        """引文格式."""
        p = Paper(authors=["Ashish Vaswani", "Noam Shazeer"], year="2017")
        assert "Vaswani" in p.citation
        assert "et al." in p.citation
        assert "2017" in p.citation

    def test_citation_single_author(self):
        """单一作者."""
        p = Paper(authors=["John Smith"], year="2023")
        assert "et al." not in p.citation
        assert "Smith" in p.citation

    def test_str_repr(self):
        """字符串表示."""
        p = Paper(title="A" * 70, authors=["Alice"], year="2024")
        s = str(p)
        assert "Paper(" in s
        assert "..." in s  # 标题过长被截断


# ═══════════════════════════════════════════════════════════════════
# 异常测试
# ═══════════════════════════════════════════════════════════════════

class TestExceptions:
    """自定义异常测试."""

    def test_base_error(self):
        """基类异常."""
        exc = PaperDownloaderError("test", details={"key": "val"})
        assert "test" in str(exc)
        assert exc.details == {"key": "val"}

    def test_paper_not_found(self):
        """PaperNotFoundError."""
        exc = PaperNotFoundError("未找到", query="test query", engines=["arxiv"])
        assert exc.query == "test query"
        assert exc.engines == ["arxiv"]
        assert "arxiv" in str(exc)

    def test_download_error(self):
        """DownloadError."""
        exc = DownloadError(
            "下载失败",
            url="https://x.org/bad.pdf",
            attempt_count=3,
            last_error="timeout",
        )
        assert exc.url == "https://x.org/bad.pdf"
        assert exc.attempt_count == 3
        assert "timeout" in str(exc)

    def test_validation_error(self):
        """ValidationError."""
        exc = ValidationError("无效", file_path="/tmp/x.pdf", reason="corrupted")
        assert exc.file_path == "/tmp/x.pdf"
        assert exc.reason == "corrupted"

    def test_config_error(self):
        """ConfigError."""
        exc = ConfigError("配置错", config_path="/tmp/cfg.yaml", missing_key="engines")
        assert exc.config_path == "/tmp/cfg.yaml"
        assert exc.missing_key == "engines"

    def test_search_error(self):
        """SearchError."""
        exc = SearchError("搜索失败", query="xyz", engine_errors={"arxiv": "timeout"})
        assert exc.query == "xyz"
        assert exc.engine_errors == {"arxiv": "timeout"}

    def test_exception_inheritance(self):
        """继承关系."""
        assert issubclass(PaperNotFoundError, PaperDownloaderError)
        assert issubclass(DownloadError, PaperDownloaderError)
        assert issubclass(ValidationError, PaperDownloaderError)
        assert issubclass(ConfigError, PaperDownloaderError)
        assert issubclass(SearchError, PaperDownloaderError)

    def test_catch_all(self):
        """统一捕获."""
        for exc_cls in [PaperNotFoundError, DownloadError, ValidationError, ConfigError, SearchError]:
            try:
                raise exc_cls("msg")
            except PaperDownloaderError:
                pass  # 全部可被基类捕获


# ═══════════════════════════════════════════════════════════════════
# PaperDownloader 核心类测试
# ═══════════════════════════════════════════════════════════════════

class TestCoreDownloader:
    """PaperDownloader 核心类测试."""

    @pytest.fixture
    def dl(self) -> PaperDownloader:
        """不依赖外部配置的下载器."""
        return PaperDownloader(
            config={
                "search": {"engines": ["arxiv"], "max_results": 5, "sort_by": "relevance"},
                "download": {"path": "./papers", "filename_template": "{first_author}_{year}_{title}"},
                "concurrency": {"max_downloads": 1, "request_delay": 0},
                "timeout": {"search": 5, "download": 10, "connection": 3},
                "proxy": {"enabled": False},
                "retry": {"max_attempts": 1, "backoff_factor": 0},
                "logging": {"level": "ERROR"},
            }
        )

    def test_init_with_config_dict(self):
        """使用配置字典初始化."""
        dl = PaperDownloader(config={"search": {"engines": ["arxiv"]}})
        assert dl._config["search"]["engines"] == ["arxiv"]

    def test_init_with_kwargs(self):
        """kwargs 覆盖配置."""
        dl = PaperDownloader(engines=["crossref"], max_results=15)
        assert dl._config["search"]["engines"] == ["crossref"]
        assert dl._config["search"]["max_results"] == 15

    def test_config_property(self):
        """config 属性返回只读副本."""
        dl = PaperDownloader()
        cfg = dl.config
        cfg["search"] = {}  # 修改副本不影响原始
        assert isinstance(dl._config["search"], dict)

    def test_search_basic(self, dl):
        """搜索返回 Paper 列表."""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {"title": "Test Paper", "authors": ["A"], "year": "2024", "doi": "10.t", "source": "arxiv"}
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        papers = dl.search("test", max_results=3)
        assert len(papers) == 1
        assert papers[0].title == "Test Paper"
        assert isinstance(papers[0], Paper)

    def test_search_no_results(self, dl):
        """搜索无结果抛出 PaperNotFoundError."""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = []
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        with pytest.raises(PaperNotFoundError) as exc_info:
            dl.search("nonexistent paper xyz")
        assert "nonexistent" in str(exc_info.value)

    def test_normalize_paper_input_single_paper(self, dl):
        """输入单个 Paper."""
        p = Paper(title="T")
        result = dl._normalize_paper_input(p)
        assert len(result) == 1
        assert result[0].title == "T"

    def test_normalize_paper_input_paper_list(self, dl):
        """输入 Paper 列表."""
        papers = [Paper(title="A"), Paper(title="B")]
        result = dl._normalize_paper_input(papers)
        assert len(result) == 2

    def test_normalize_paper_input_dict_list(self, dl):
        """输入 dict 列表."""
        papers = [{"title": "A", "authors": [], "year": "2024"}, {"title": "B", "authors": [], "year": "2023"}]
        result = dl._normalize_paper_input(papers)
        assert len(result) == 2
        assert all(isinstance(p, Paper) for p in result)

    def test_normalize_paper_input_invalid(self, dl):
        """无效输入."""
        with pytest.raises(ValidationError):
            dl._normalize_paper_input("not a paper")  # type: ignore[arg-type]

    def test_download_by_title(self, dl):
        """端到端单篇下载."""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [
            {
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": "2017",
                "doi": "10.1234/attention",
                "arxiv_id": "1706.03762",
                "pdf_url": "https://arxiv.org/pdf/1706.03762",
                "source": "arxiv",
                "citation_count": 100000,
                "url": "https://arxiv.org/abs/1706.03762",
            }
        ]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # Mock download manager
        mock_manager = MagicMock()
        mock_task = MagicMock()
        mock_task.title = "Attention Is All You Need"
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        mock_task.pdf_path = "/tmp/attention.pdf"
        mock_task.file_size = 1024
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_manager.run_all.return_value = [mock_task]
        dl._download_manager = mock_manager

        paper = dl.download_by_title("Attention Is All You Need")
        assert paper.title == "Attention Is All You Need"

    def test_download_empty_input(self, dl):
        """空输入抛异常."""
        with pytest.raises(ValidationError):
            dl.download([])

    def test_set_progress_callback(self, dl):
        """设置进度回调."""
        called: List[Paper] = []

        def cb(p: Paper) -> None:
            called.append(p)

        dl.set_progress_callback(cb)
        assert dl._progress_callback is not None

    def test_context_manager(self):
        """上下文管理器."""
        with PaperDownloader() as dl:
            assert dl is not None


# ═══════════════════════════════════════════════════════════════════
# API 函数测试
# ═══════════════════════════════════════════════════════════════════

class TestAPI:
    """对外 API 函数测试."""

    def setup_method(self):
        api.reset_downloader()

    def teardown_method(self):
        api.reset_downloader()

    def test_download_paper(self):
        """api.download_paper."""
        with patch.object(PaperDownloader, "download_by_title") as mock_dl:
            mock_paper = Paper(title="Test", pdf_path="/tmp/test.pdf")
            mock_dl.return_value = mock_paper

            paper = api.download_paper("Test Paper", output_dir="./out")
            assert paper.title == "Test"
            mock_dl.assert_called_once()

    def test_download_papers(self):
        """api.download_papers."""
        with patch.object(PaperDownloader, "batch_download") as mock_dl:
            mock_dl.return_value = [Paper(title="A"), Paper(title="B")]

            papers = api.download_papers(["A", "B"])
            assert len(papers) == 2
            mock_dl.assert_called_once()

    def test_search_papers(self):
        """api.search_papers."""
        with patch.object(PaperDownloader, "search") as mock_dl:
            mock_dl.return_value = [Paper(title="Result")]
            papers = api.search_papers("query", max_results=5)
            assert len(papers) == 1
            mock_dl.assert_called_once_with("query", max_results=5, engines=None)

    def test_get_paper_info(self):
        """api.get_paper_info."""
        with patch.object(PaperDownloader, "get_paper_info") as mock_dl:
            mock_paper = Paper(title="Nature Paper", doi="10.1038/nature14539")
            mock_dl.return_value = mock_paper

            paper = api.get_paper_info("10.1038/nature14539")
            assert paper is not None
            assert paper.doi == "10.1038/nature14539"

    def test_set_config(self):
        """api.set_config."""
        api.set_config(engines=["crossref"])
        dl = api._get_downloader()
        assert dl._config["search"]["engines"] == ["crossref"]

    def test_reset_downloader(self):
        """api.reset_downloader."""
        api._downloader = PaperDownloader()
        api.reset_downloader()
        assert api._downloader is None


# ═══════════════════════════════════════════════════════════════════
# main.py CLI 测试
# ═══════════════════════════════════════════════════════════════════

class TestCLI:
    """CLI 入口测试."""

    def test_parser_title_argument(self):
        """--title 参数."""
        from paper_downloader.src.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--title", "Test Paper"])
        assert args.title == "Test Paper"

    def test_parser_search_download(self):
        """--search --download 参数."""
        from paper_downloader.src.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--search", "BERT", "--download", "--max", "3"])
        assert args.search == "BERT"
        assert args.download is True
        assert args.max == 3

    def test_parser_engines(self):
        """--engines 多值参数."""
        from paper_downloader.src.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["--title", "Test", "--engines", "arxiv", "crossref"])
        assert args.engines == ["arxiv", "crossref"]

    def test_parser_mutually_exclusive(self):
        """--title 和 --search 互斥."""
        from paper_downloader.src.main import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--title", "A", "--search", "B"])

    def test_read_titles_from_file(self, tmp_path):
        """从文件读取标题."""
        from paper_downloader.src.main import read_titles_from_file
        f = tmp_path / "titles.txt"
        f.write_text("Paper A\nPaper B\n# comment\n\nPaper C\n")

        titles = read_titles_from_file(str(f))
        assert titles == ["Paper A", "Paper B", "Paper C"]

    def test_read_titles_nonexistent_file(self):
        """文件不存在."""
        from paper_downloader.src.main import read_titles_from_file
        with pytest.raises(SystemExit):
            read_titles_from_file("/nonexistent/titles.txt")


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
