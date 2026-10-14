"""
test_paper_downloader.py — PaperDownloader 基类的单元测试.

测试抽象接口、配置加载、文件命名、代理构建等核心功能.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import mock_open, patch

import pytest
import yaml

from paper_downloader.src.paper_downloader import PaperDownloader


# ── 具体实现 stub（用于测试抽象基类）──────────────────────────────

class _StubDownloader(PaperDownloader):
    """用于测试的 PaperDownloader 具体实现."""

    def search_by_title(self, title: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "title": "Test Paper",
                "authors": ["Jane Smith", "John Doe"],
                "year": "2024",
                "doi": "10.1234/example",
                "url": "https://example.org/paper",
                "source": "test",
            }
        ]

    def download_paper(
        self, identifier: str, save_path: Optional[str] = None, **kwargs: Any
    ) -> Optional[str]:
        return f"{save_path or './papers'}/test_paper.pdf"


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """返回一份最小有效配置."""
    return {
        "download": {
            "path": "./papers",
            "organize_by": "flat",
            "filename_template": "{first_author}_{year}_{title}",
        },
        "search": {
            "engines": ["arxiv"],
            "max_results": 10,
            "sort_by": "relevance",
            "min_year": None,
            "language": "en",
        },
        "concurrency": {
            "max_downloads": 3,
            "request_delay": 0.0,
        },
        "timeout": {
            "search": 30,
            "download": 120,
            "connection": 10,
        },
        "proxy": {
            "enabled": False,
            "http": "",
            "https": "",
            "username": "",
            "password": "",
        },
        "logging": {
            "level": "DEBUG",
            "file": "./logs/test.log",
            "max_bytes": 10485760,
            "backup_count": 3,
        },
        "retry": {
            "max_attempts": 3,
            "backoff_factor": 2,
        },
    }


@pytest.fixture
def config_file(sample_config, tmp_path):
    """向临时目录写入一份 YAML 配置文件."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.fixture
def downloader(config_file):
    """返回使用临时配置的 StubDownloader 实例."""
    return _StubDownloader(config_path=config_file)


# ── 配置加载 ──────────────────────────────────────────────────────

class TestConfigLoading:
    """配置加载相关测试."""

    def test_load_valid_config(self, downloader, sample_config):
        """加载有效 YAML 配置."""
        assert downloader.config["search"]["max_results"] == 10
        assert downloader.config["download"]["path"] == "./papers"

    def test_load_nonexistent_config(self):
        """加载不存在的配置文件应抛出 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _StubDownloader(config_path="/nonexistent/path/config.yaml")

    def test_load_default_config_path(self, sample_config):
        """未提供 config_path 时使用默认路径."""
        with patch("builtins.open", mock_open(read_data=yaml.dump(sample_config))):
            d = _StubDownloader()
            assert d.config is not None


# ── 文件名处理 ────────────────────────────────────────────────────

class TestFilenameSanitization:
    """文件名清理 & 生成测试."""

    def test_remove_illegal_chars(self, downloader):
        """非法字符应替换为下划线."""
        assert downloader._sanitize_filename('test:file<name>.pdf') == "test_file_name.pdf"

    def test_trim_long_filename(self, downloader):
        """超长文件名应截断."""
        long_name = "x" * 300
        result = downloader._sanitize_filename(long_name, max_len=100)
        assert len(result) <= 100

    def test_build_filename_default_template(self, downloader):
        """默认模板生成文件名."""
        name = downloader._build_filename(
            title="Attention Is All You Need",
            authors="Ashish Vaswani",
            year="2017",
        )
        assert "Vaswani" in name
        assert "2017" in name
        assert "Attention" in name

    def test_build_filename_author_list(self, downloader):
        """作者为 list 时正确提取第一作者."""
        name = downloader._build_filename(
            title="Deep Learning",
            authors=["Geoffrey Hinton", "Yann LeCun"],
            year="2015",
        )
        assert "Hinton" in name


# ── 代理 ──────────────────────────────────────────────────────────

class TestProxy:
    """代理配置测试."""

    def test_proxy_disabled(self, downloader):
        """代理未启用时返回 None."""
        assert downloader._get_proxies() is None

    def test_proxy_enabled(self, sample_config, config_file):
        """代理启用时返回正确字典."""
        sample_config["proxy"]["enabled"] = True
        sample_config["proxy"]["http"] = "http://127.0.0.1:8080"
        sample_config["proxy"]["https"] = "https://127.0.0.1:8443"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_config, f)

        d = _StubDownloader(config_path=config_file)
        proxies = d._get_proxies()
        assert proxies is not None
        assert "http" in proxies
        assert "127.0.0.1:8080" in proxies["http"]

    def test_proxy_with_auth(self, sample_config, config_file):
        """代理认证信息应拼接到 URL 中."""
        sample_config["proxy"]["enabled"] = True
        sample_config["proxy"]["http"] = "http://proxy.example.com:3128"
        sample_config["proxy"]["username"] = "alice"
        sample_config["proxy"]["password"] = "s3cret"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_config, f)

        d = _StubDownloader(config_path=config_file)
        proxies = d._get_proxies()
        assert "alice:s3cret" in proxies["http"]


# ── 标识符提取 ────────────────────────────────────────────────────

class TestIdentifierExtraction:
    """_extract_identifier 静态方法测试."""

    def test_prefer_doi(self):
        """应优先返回 DOI."""
        paper = {"doi": "10.1234/x", "arxiv_id": "2301.00001", "url": "https://x.org"}
        assert PaperDownloader._extract_identifier(paper) == "10.1234/x"

    def test_fallback_to_arxiv(self):
        """无 DOI 时回退到 arXiv ID."""
        paper = {"arxiv_id": "2301.00001", "url": "https://x.org"}
        assert PaperDownloader._extract_identifier(paper) == "2301.00001"

    def test_fallback_to_url(self):
        """仅有 URL 时返回 URL."""
        paper = {"url": "https://example.org/paper"}
        assert PaperDownloader._extract_identifier(paper) == "https://example.org/paper"

    def test_missing_all_returns_none(self):
        """所有字段缺失时返回 None."""
        paper = {"title": "No identifiers"}
        assert PaperDownloader._extract_identifier(paper) is None


# ── 搜索 & 下载 ───────────────────────────────────────────────────

class TestSearchAndDownload:
    """search_by_title / download_paper 接口测试."""

    def test_search_returns_list(self, downloader):
        """search_by_title 应返回列表."""
        results = downloader.search_by_title("test query")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]

    def test_download_returns_string(self, downloader):
        """download_paper 应返回路径字符串."""
        path = downloader.download_paper("10.1234/example", save_path="./papers")
        assert isinstance(path, str)

    def test_download_missing_identifier(self, downloader):
        """缺少标识符的论文应被跳过."""
        papers = [{"title": "No ID paper"}]
        result = downloader._download_many(papers)
        assert "pdf_path" not in result[0] or result[0].get("pdf_path") is None


# ── 入口 ──────────────────────────────────────────────────────────

class TestMain:
    """main() 端到端流水线测试."""

    def test_main_search_only(self, downloader):
        """download=False 时仅搜索不下载."""
        results = downloader.main("test", download=False)
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"

    def test_main_search_and_download(self, downloader, tmp_path):
        """download=True 时搜索并下载."""
        results = downloader.main("test", download=True, save_path=str(tmp_path))
        assert len(results) == 1
        assert results[0].get("pdf_path") is not None
        # pdf_path 应指向生成的 Stub 路径
        assert "test_paper.pdf" in results[0]["pdf_path"]

    def test_main_no_results(self, downloader):
        """搜索无结果时返回空列表."""
        # 临时覆盖为返回空
        original = downloader.search_by_title

        def _empty(*a, **kw):
            return []

        downloader.search_by_title = _empty  # type: ignore[method-assign]
        results = downloader.main("nonexistent")
        assert results == []
        # 恢复
        downloader.search_by_title = original  # type: ignore[method-assign]

    def test_main_max_results_truncation(self, config_file):
        """max_results 应截断结果列表."""

        class _MultiStub(PaperDownloader):
            def search_by_title(self, title, **kw):
                return [{"title": f"Paper {i}", "doi": f"10.{i}"} for i in range(10)]

            def download_paper(self, ident, save_path=None, **kw):
                return f"{save_path}/paper.pdf"

        d = _MultiStub(config_path=config_file)
        results = d.main("test", download=False, max_results=3)
        assert len(results) == 3


# ── 抽象基类约束 ──────────────────────────────────────────────────

class TestAbstractConstraints:
    """未实现抽象方法时不应允许实例化."""

    def test_cannot_instantiate_abstract(self):
        """直接实例化 PaperDownloader 应抛出 TypeError."""
        with pytest.raises(TypeError):
            PaperDownloader()  # type: ignore[abstract]


# ── 运行入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
