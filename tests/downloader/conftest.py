"""
conftest.py — pytest 全局配置与共享 fixtures

为 paper_downloader 测试套件提供:
    - 统一测试配置
    - Mock 论文数据
    - 最小合法 PDF 生成器
    - Mock 搜索工厂 / 下载管理器
    - 临时目录与文件管理
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# 抑制测试期间的日志噪音
logging.basicConfig(level=logging.WARNING)


def pytest_configure(config):
    """注册自定义 pytest 标记。"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试（需要网络连接）")
    config.addinivalue_line("markers", "network: 需要网络连接的测试")
    config.addinivalue_line("markers", "integration: 端到端集成测试")


# ═══════════════════════════════════════════════════════════════════
# 配置文件 fixture
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_config() -> Dict[str, Any]:
    """会话级通用测试配置。"""
    return {
        "search": {
            "engines": ["arxiv", "crossref"],
            "max_results": 10,
            "sort_by": "relevance",
        },
        "download": {
            "path": "./papers",
            "filename_template": "{first_author}_{year}_{title}",
        },
        "concurrency": {
            "max_downloads": 2,
            "request_delay": 0.0,
        },
        "timeout": {
            "search": 5,
            "download": 10,
            "connection": 3,
        },
        "proxy": {
            "enabled": False,
            "http": "",
            "https": "",
            "username": "",
            "password": "",
        },
        "retry": {
            "max_attempts": 2,
            "backoff_factor": 1,
        },
        "cache": {
            "enabled": False,
            "ttl": 3600,
        },
        "logging": {
            "level": "WARNING",
            "file": "/dev/null",
        },
    }


@pytest.fixture(scope="function")
def temp_config(sample_config, tmp_path) -> Dict[str, Any]:
    """函数级配置（含临时下载路径）。"""
    cfg = sample_config.copy()
    cfg["download"] = {**cfg["download"], "path": str(tmp_path / "papers")}
    return cfg


# ═══════════════════════════════════════════════════════════════════
# PDF 工具
# ═══════════════════════════════════════════════════════════════════

_MINIMAL_PDF_BYTES = (
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


@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """最小合法 PDF 字节。"""
    return _MINIMAL_PDF_BYTES


@pytest.fixture(scope="function")
def minimal_pdf(tmp_path) -> Path:
    """创建最小合法 PDF 临时文件。"""
    pdf = tmp_path / "minimal.pdf"
    pdf.write_bytes(_MINIMAL_PDF_BYTES)
    return pdf


def make_minimal_pdf(path: Path) -> Path:
    """工具函数：在给定路径创建最小合法 PDF。"""
    path.write_bytes(_MINIMAL_PDF_BYTES)
    return path


# ═══════════════════════════════════════════════════════════════════
# Mock 论文数据
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def sample_paper_dict() -> Dict[str, Any]:
    """单篇标准化论文 dict。"""
    return {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": "2017",
        "abstract": "The dominant sequence transduction models...",
        "doi": "10.5555/3295222.3295349",
        "arxiv_id": "1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
        "source": "arxiv",
        "citation_count": 100000,
        "journal": "NeurIPS 2017",
    }


@pytest.fixture(scope="session")
def sample_paper_dicts() -> List[Dict[str, Any]]:
    """多篇标准化论文 dict。"""
    return [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
            "year": "2019",
            "doi": "10.18653/v1/N19-1423",
            "arxiv_id": "1810.04805",
            "pdf_url": "https://arxiv.org/pdf/1810.04805",
            "source": "arxiv",
            "citation_count": 80000,
            "journal": "NAACL 2019",
            "url": "https://arxiv.org/abs/1810.04805",
        },
        {
            "title": "GPT-4 Technical Report",
            "authors": ["OpenAI"],
            "year": "2024",
            "doi": "10.48550/arXiv.2303.08774",
            "arxiv_id": "2303.08774",
            "pdf_url": "https://arxiv.org/pdf/2303.08774",
            "source": "arxiv",
            "citation_count": 5000,
            "url": "https://arxiv.org/abs/2303.08774",
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren"],
            "year": "2016",
            "doi": "10.1109/CVPR.2016.90",
            "arxiv_id": "1512.03385",
            "pdf_url": "https://arxiv.org/pdf/1512.03385",
            "source": "crossref",
            "citation_count": 150000,
            "url": "https://arxiv.org/abs/1512.03385",
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# Mock 搜索工厂
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def mock_search_factory(sample_paper_dict) -> MagicMock:
    """返回预设结果的 Mock SearchFactory。"""
    mock = MagicMock()
    mock.search_all.return_value = [sample_paper_dict]
    mock.available_engines = ["arxiv", "crossref"]
    mock.get_engine.return_value = None
    return mock


@pytest.fixture(scope="function")
def mock_search_factory_multi(sample_paper_dicts) -> MagicMock:
    """返回多篇结果的 Mock SearchFactory。"""
    mock = MagicMock()
    mock.search_all.return_value = sample_paper_dicts
    mock.available_engines = ["arxiv", "crossref"]
    return mock


@pytest.fixture(scope="function")
def mock_search_factory_empty() -> MagicMock:
    """返回空结果的 Mock SearchFactory。"""
    mock = MagicMock()
    mock.search_all.return_value = []
    mock.available_engines = ["arxiv"]
    return mock


# ═══════════════════════════════════════════════════════════════════
# Mock 下载管理器
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def mock_download_manager(tmp_path) -> MagicMock:
    """Mock DownloadManager — 所有任务成功。"""
    mock = MagicMock()
    mock_task = MagicMock()
    mock_task.title = "Test Paper"
    mock_task.status = MagicMock()
    mock_task.status.value = "completed"
    mock_task.pdf_path = str(tmp_path / "test.pdf")
    mock_task.file_size = 1024
    mock_task.completed_at = "2024-01-01T00:00:00"
    mock.run_all.return_value = [mock_task]
    return mock


@pytest.fixture(scope="function")
def mock_download_manager_failure(tmp_path) -> MagicMock:
    """Mock DownloadManager — 所有任务失败。"""
    mock = MagicMock()
    mock_task = MagicMock()
    mock_task.title = "Test Paper"
    mock_task.status = MagicMock()
    mock_task.status.value = "failed"
    mock_task.pdf_path = None
    mock_task.file_size = 0
    mock.run_all.return_value = [mock_task]
    return mock


# ═══════════════════════════════════════════════════════════════════
# 临时文件
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def temp_titles_file(tmp_path) -> Path:
    """包含论文标题的临时 TXT 文件。"""
    f = tmp_path / "titles.txt"
    f.write_text(
        "Attention Is All You Need\n"
        "BERT: Pre-training of Deep Bidirectional Transformers\n"
        "# This is a comment\n"
        "\n"
        "GPT-4 Technical Report\n"
    )
    return f


@pytest.fixture(scope="function")
def temp_csv_file(tmp_path) -> Path:
    """包含论文标题的临时 CSV 文件。"""
    f = tmp_path / "papers.csv"
    f.write_text(
        "title,author,year\n"
        "Attention Is All You Need,Vaswani,2017\n"
        "BERT,Devlin,2019\n"
        "GPT-4,OpenAI,2024\n"
    )
    return f


@pytest.fixture(scope="function")
def temp_json_file(tmp_path) -> Path:
    """包含论文标题的临时 JSON 文件。"""
    f = tmp_path / "papers.json"
    json.dump([
        {"title": "Paper Alpha", "year": 2024},
        {"title": "Paper Beta", "year": 2023},
    ], f.open("w"))
    return f


@pytest.fixture(scope="function")
def temp_bibtex_file(tmp_path) -> Path:
    """临时的 BibTeX 文件。"""
    f = tmp_path / "refs.bib"
    f.write_text("""
@article{vaswani2017,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam},
  year = {2017},
  doi = {10.5555/3295222},
  journal = {NeurIPS},
}
@inproceedings{devlin2019,
  title = {BERT: Pre-training of Deep Bidirectional Transformers},
  author = {Devlin, Jacob and Chang, Ming-Wei},
  year = {2019},
  doi = {10.18653/v1/N19-1423},
}
@article{he2016,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu},
  year = {2016},
  doi = {10.1109/CVPR.2016.90},
}
    """)
    return f


# ═══════════════════════════════════════════════════════════════════
# 数据库
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def temp_db_path(tmp_path) -> str:
    """临时 SQLite 数据库路径。"""
    return str(tmp_path / "test_cache.db")


# ═══════════════════════════════════════════════════════════════════
# 环境清理
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function", autouse=False)
def clean_config_singleton():
    """每个测试前重置 ConfigManager 单例。"""
    from paper_downloader.src.config.config_manager import ConfigManager
    ConfigManager._instance = None
    yield
    ConfigManager._instance = None


@pytest.fixture(scope="function", autouse=False)
def clean_loggers():
    """每个测试前清理全局日志器。"""
    from paper_downloader.src.utils.logger import _loggers as _logger_cache
    _logger_cache.clear()
    yield
    _logger_cache.clear()
