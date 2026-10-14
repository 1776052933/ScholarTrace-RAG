"""
test_search_engines.py — 搜索引擎模块单元测试.

覆盖 base_search、arxiv_search、crossref_search、
google_scholar_search 和 search_factory。
"""

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from paper_downloader.src.search_engines.base_search import SearchEngine
from paper_downloader.src.search_engines.arxiv_search import ArxivSearch
from paper_downloader.src.search_engines.crossref_search import CrossrefSearch
from paper_downloader.src.search_engines.google_scholar_search import GoogleScholarSearch
from paper_downloader.src.search_engines.search_factory import SearchFactory

# ═══════════════════════════════════════════════════════════════════
# 公共 fixtures & 工具
# ═══════════════════════════════════════════════════════════════════

SAMPLE_CONFIG: Dict[str, Any] = {
    "search": {
        "engines": ["arxiv", "crossref"],
        "max_results": 10,
        "sort_by": "relevance",
    },
    "concurrency": {"request_delay": 0.0, "max_downloads": 2},
    "timeout": {"search": 5, "download": 10, "connection": 3},
    "proxy": {"enabled": False},
    "retry": {"max_attempts": 1, "backoff_factor": 1},
    "logging": {"level": "WARNING", "file": "/dev/null"},
}


class _ConcreteEngine(SearchEngine):
    """用于测试抽象基类的具体实现."""
    ENGINE_NAME = "test"

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        return [{"title": f"Result for {query}", "authors": ["A. Tester"], "year": "2024"}]

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return {"title": "Test Paper", "doi": identifier, "authors": ["A. Tester"], "year": "2024"}


# ═══════════════════════════════════════════════════════════════════
# base_search 测试
# ═══════════════════════════════════════════════════════════════════

class TestBaseSearch:
    """SearchEngine 抽象基类测试."""

    def setup_method(self):
        self.engine = _ConcreteEngine(SAMPLE_CONFIG)

    def test_normalize_results_basic(self):
        """基本结果标准化."""
        raw = [
            {"title": "Hello World", "authors": ["Alice", "Bob"],
             "year": "2023", "doi": "10.000/xyz",
             "abstract": "An abstract.", "url": "http://x.org"},
        ]
        results = self.engine.normalize_results(raw)
        assert len(results) == 1
        assert results[0]["source"] == "test"
        assert results[0]["title"] == "Hello World"
        assert results[0]["authors"] == ["Alice", "Bob"]
        assert results[0]["year"] == "2023"
        assert results[0]["doi"] == "10.000/xyz"
        assert results[0]["abstract"] == "An abstract."
        assert results[0]["url"] == "http://x.org"

    def test_normalize_results_default_values(self):
        """缺失字段使用默认值填充."""
        raw = [{"title": "Minimal Paper"}]
        results = self.engine.normalize_results(raw)
        assert results[0]["authors"] == []
        assert results[0]["year"] is None
        assert results[0]["doi"] is None
        assert results[0]["citation_count"] is None
        assert results[0]["journal"] is None

    def test_normalize_results_empty_title_skipped(self):
        """标题为空的条目应被跳过."""
        raw = [{"title": ""}, {"title": "Valid"}]
        results = self.engine.normalize_results(raw)
        assert len(results) == 1
        assert results[0]["title"] == "Valid"

    def test_normalize_results_malformed_not_crashing(self):
        """格式异常的条目不应导致崩溃."""
        raw = [{"title": "OK"}, None, 123, "string_item"]  # type: ignore[list-item]
        results = self.engine.normalize_results(raw)  # type: ignore[arg-type]
        assert len(results) == 1

    # ── 标题提取 ───────────────────────────────────────────────

    def test_extract_title_list(self):
        """CrossRef 风格: 标题为列表."""
        assert SearchEngine._extract_title(["Actual Title", "Subtitle"]) == "Actual Title"

    def test_extract_title_html(self):
        """标题含 HTML 标签应清洗."""
        title = SearchEngine._extract_title("<i>Italic Title</i>")
        assert "<i>" not in title
        assert "Italic Title" in title

    def test_extract_title_whitespace(self):
        """多余空白应规范化."""
        title = SearchEngine._extract_title("  Too   many   spaces  ")
        assert title == "Too many spaces"

    # ── 作者标准化 ────────────────────────────────────────────

    def test_normalize_authors_string(self):
        """字符串作者按 ; 分割."""
        authors = SearchEngine._normalize_authors("Alice; Bob; Charlie")
        assert authors == ["Alice", "Bob", "Charlie"]

    def test_normalize_authors_list_of_strings(self):
        """字符串列表作者."""
        authors = SearchEngine._normalize_authors(["Alice", "Bob"])
        assert authors == ["Alice", "Bob"]

    def test_normalize_authors_dict_list(self):
        """dict 列表 (given/family)."""
        authors = SearchEngine._normalize_authors([
            {"given": "John", "family": "Smith"},
            {"given": "Jane", "family": "Doe"},
        ])
        assert authors == ["John Smith", "Jane Doe"]

    def test_normalize_authors_dict_name_only(self):
        """dict 列表 (name)."""
        authors = SearchEngine._normalize_authors([
            {"name": "John Smith"},
            {"name": "Jane Doe"},
        ])
        assert authors == ["John Smith", "Jane Doe"]

    def test_normalize_authors_object_with_name_attr(self):
        """对象有 .name 属性."""
        class AuthorObj:
            def __init__(self, name):
                self.name = name

        authors = SearchEngine._normalize_authors([
            AuthorObj("A. Einstein"),
            AuthorObj("N. Bohr"),
        ])
        assert authors == ["A. Einstein", "N. Bohr"]

    def test_normalize_authors_empty(self):
        """空输入返回空列表."""
        assert SearchEngine._normalize_authors(None) == []
        assert SearchEngine._normalize_authors("") == []
        assert SearchEngine._normalize_authors([]) == []

    # ── 年份提取 ──────────────────────────────────────────────

    def test_extract_year_int(self):
        """整数年份."""
        assert SearchEngine._extract_year(2023) == "2023"

    def test_extract_year_date_string(self):
        """日期字符串格式."""
        assert SearchEngine._extract_year("2023-05-15") == "2023"

    def test_extract_year_none(self):
        """空输入."""
        assert SearchEngine._extract_year(None) is None
        assert SearchEngine._extract_year("") is None

    def test_extract_year_text_with_year(self):
        """文本中嵌有年份."""
        assert SearchEngine._extract_year("Published: 2021. Nature") == "2021"

    # ── 空白模板 ──────────────────────────────────────────────

    def test_make_blank_result(self):
        """空白结果模板包含所有字段."""
        blank = SearchEngine.make_blank_result("test")
        assert blank["title"] == ""
        assert blank["authors"] == []
        assert blank["source"] == "test"
        assert blank["year"] is None
        assert blank["doi"] is None


# ═══════════════════════════════════════════════════════════════════
# ArxivSearch 测试
# ═══════════════════════════════════════════════════════════════════

class TestArxivSearch:
    """ArxivSearch 引擎测试（mock 外部 API）."""

    def setup_method(self):
        self.engine = ArxivSearch(SAMPLE_CONFIG)

    # ── arXiv ID 提取 ─────────────────────────────────────────

    def test_extract_arxiv_id_standard(self):
        """标准格式: 1706.03762."""
        assert ArxivSearch._extract_arxiv_id("1706.03762") == "1706.03762"

    def test_extract_arxiv_id_with_prefix(self):
        """带 arxiv: 前缀."""
        assert ArxivSearch._extract_arxiv_id("arxiv:1706.03762") == "1706.03762"

    def test_extract_arxiv_id_with_version(self):
        """带版本号."""
        assert ArxivSearch._extract_arxiv_id("1706.03762v7") == "1706.03762v7"

    def test_extract_arxiv_id_from_url(self):
        """从 URL 提取."""
        url = "https://arxiv.org/abs/1706.03762"
        assert ArxivSearch._extract_arxiv_id(url) == "1706.03762"

    def test_extract_arxiv_id_old_format(self):
        """旧格式: hep-th/9711200."""
        assert ArxivSearch._extract_arxiv_id("hep-th/9711200") == "hep-th/9711200"

    def test_extract_arxiv_id_invalid(self):
        """无效输入."""
        assert ArxivSearch._extract_arxiv_id("not an arxiv id") is None
        assert ArxivSearch._extract_arxiv_id("") is None

    def test_parse_atom_feed_without_arxiv_dependency(self):
        """第三方 arxiv 包缺失时仍可解析官方 Atom 响应."""
        atom = b'''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/1706.03762v7</id>
            <title> Attention Is All You Need </title>
            <summary>Transformer abstract</summary>
            <published>2017-06-12T17:57:34Z</published>
            <author><name>Ashish Vaswani</name></author>
            <link title="pdf" href="https://arxiv.org/pdf/1706.03762v7" type="application/pdf"/>
          </entry>
        </feed>'''

        results = ArxivSearch._parse_atom_feed(atom)

        assert results[0]["title"] == "Attention Is All You Need"
        assert results[0]["arxiv_id"] == "1706.03762v7"
        assert results[0]["year"] == "2017"

    # ── 搜索（mock arxiv 模块内部引用）───────────────────────

    @patch("paper_downloader.src.search_engines.arxiv_search.arxiv")
    def test_search_with_mock(self, mock_arxiv):
        """Mock arXiv API 返回结果."""
        self.engine._available = True

        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.authors = [MagicMock()]
        mock_result.authors[0].name = "John Smith"
        mock_result.summary = "An abstract."
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.00001"
        mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
        mock_result.published = MagicMock()
        mock_result.published.year = 2024
        mock_result.doi = "10.1234/test"
        mock_result.journal_ref = "Nature 2024"
        mock_result.comment = ""

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_arxiv.Client.return_value = mock_client

        results = self.engine.search("test", max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Test Paper"
        assert results[0]["authors"] == ["John Smith"]
        assert results[0]["year"] == "2024"
        assert results[0]["doi"] == "10.1234/test"
        assert results[0]["journal"] == "Nature 2024"
        assert results[0]["source"] == "arxiv"

    @patch("paper_downloader.src.search_engines.arxiv_search.arxiv")
    def test_search_empty_results(self, mock_arxiv):
        """arXiv 返回空结果."""
        self.engine._available = True

        mock_client = MagicMock()
        mock_client.results.return_value = []
        mock_arxiv.Client.return_value = mock_client

        results = self.engine.search("zzz_nonexistent_paper_xyz", max_results=5)
        assert results == []

    @patch("paper_downloader.src.search_engines.arxiv_search.arxiv")
    def test_search_year_filter(self, mock_arxiv):
        """年份过滤."""
        self.engine._available = True

        mock_r1 = self._make_mock_result("Paper 2023", year=2023)
        mock_r2 = self._make_mock_result("Paper 2025", year=2025)

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_r1, mock_r2]
        mock_arxiv.Client.return_value = mock_client

        results = self.engine.search("test", max_results=10, min_year=2024)
        assert len(results) == 1
        assert results[0]["year"] == "2025"

    # ── 获取单篇信息 ──────────────────────────────────────────

    @patch("paper_downloader.src.search_engines.arxiv_search.arxiv")
    def test_get_paper_info(self, mock_arxiv):
        """通过 arXiv ID 获取单篇信息."""
        self.engine._available = True

        mock_result = self._make_mock_result("Attention Paper", year=2017, arxiv_id="1706.03762")
        mock_client = MagicMock()
        mock_client.results.return_value = iter([mock_result])
        mock_arxiv.Client.return_value = mock_client

        info = self.engine.get_paper_info("1706.03762")
        assert info is not None
        assert info["title"] == "Attention Paper"
        assert info["year"] == "2017"

    def test_get_paper_info_invalid_id(self):
        """无效 arXiv ID."""
        info = self.engine.get_paper_info("not_valid")
        assert info is None

    @patch("paper_downloader.src.search_engines.arxiv_search.arxiv")
    def test_get_paper_info_not_found(self, mock_arxiv):
        """arXiv ID 不存在."""
        self.engine._available = True

        mock_client = MagicMock()
        mock_client.results.return_value = iter([])  # 空迭代器
        mock_arxiv.Client.return_value = mock_client

        info = self.engine.get_paper_info("9999.99999")
        assert info is None

    @staticmethod
    def _make_mock_result(title: str, year: int = 2024, arxiv_id: str = "2401.00001") -> MagicMock:
        """创建 mock arXiv result."""
        r = MagicMock()
        r.title = title
        r.authors = [MagicMock()]
        r.authors[0].name = "A. Author"
        r.summary = "Abstract text."
        r.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        r.entry_id = f"https://arxiv.org/abs/{arxiv_id}"
        r.published = MagicMock()
        r.published.year = year
        r.doi = f"10.1234/{arxiv_id}"
        r.journal_ref = None
        r.comment = ""
        return r


# ═══════════════════════════════════════════════════════════════════
# CrossrefSearch 测试
# ═══════════════════════════════════════════════════════════════════

class TestCrossrefSearch:
    """CrossrefSearch 引擎测试."""

    def setup_method(self):
        self.engine = CrossrefSearch(SAMPLE_CONFIG)

    # ── 响应解析 ──────────────────────────────────────────────

    def test_parse_crossref_item(self):
        """解析标准 CrossRef JSON 条目."""
        item = {
            "title": ["Machine Learning Basics"],
            "author": [
                {"given": "Alice", "family": "Wang"},
                {"given": "Bob", "family": "Li"},
            ],
            "published-print": {"date-parts": [[2023, 5, 15]]},
            "DOI": "10.1234/ml.2023",
            "abstract": "A comprehensive introduction.",
            "URL": "https://example.org/ml",
            "is-referenced-by-count": 42,
            "container-title": ["Journal of ML Research"],
            "link": [
                {"content-type": "application/pdf", "URL": "https://example.org/ml.pdf"},
            ],
        }
        result = self.engine._parse_crossref_item(item)
        assert result["title"] == "Machine Learning Basics"
        assert result["authors"] == ["Alice Wang", "Bob Li"]
        assert result["year"] == "2023"
        assert result["doi"] == "10.1234/ml.2023"
        assert result["abstract"] == "A comprehensive introduction."
        assert result["citation_count"] == 42
        assert result["journal"] == "Journal of ML Research"
        assert result["pdf_url"] == "https://example.org/ml.pdf"

    def test_parse_crossref_item_missing_fields(self):
        """缺失字段时使用默认值."""
        item = {
            "title": ["Minimal"],
            "author": [],
            "DOI": "10.0000/min",
        }
        result = self.engine._parse_crossref_item(item)
        assert result["title"] == "Minimal"
        assert result["authors"] == []
        assert result["year"] is None
        assert result["citation_count"] is None

    # ── 搜索（mock HTTP session）──────────────────────────────

    def test_search_success(self):
        """Mock CrossRef API 响应 — 直接替换 session."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Deep Learning"],
                        "author": [{"given": "Yann", "family": "LeCun"}],
                        "published-print": {"date-parts": [[2015]]},
                        "DOI": "10.1038/nature14539",
                        "abstract": "Deep learning allows...",
                        "container-title": ["Nature"],
                        "is-referenced-by-count": 5000,
                        "link": [],
                    }
                ],
                "total-results": 1,
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        self.engine._session = mock_session

        results = self.engine.search("deep learning", max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Deep Learning"
        assert results[0]["doi"] == "10.1038/nature14539"
        assert results[0]["source"] == "crossref"
        assert results[0]["citation_count"] == 5000

    def test_search_empty_response(self):
        """CrossRef 返回空结果."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"items": [], "total-results": 0}}
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        self.engine._session = mock_session

        results = self.engine.search("xyznonexistent_zzz", max_results=5)
        assert results == []

    def test_search_http_error_handled(self):
        """HTTP 错误被捕获."""
        mock_session = MagicMock()
        mock_session.get.side_effect = __import__("requests").exceptions.HTTPError("500 Server Error")
        self.engine._session = mock_session

        results = self.engine.search("test", max_results=5)
        assert results == []  # 不抛异常，返回空列表

    # ── 获取单篇信息 ──────────────────────────────────────────

    def test_get_paper_info_by_doi(self):
        """通过 DOI 获取详细信息."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "title": ["Attention Is All You Need"],
                "author": [
                    {"given": "Ashish", "family": "Vaswani"},
                ],
                "published-print": {"date-parts": [[2017]]},
                "DOI": "10.1234/attention",
                "abstract": "The dominant sequence transduction...",
                "container-title": ["NeurIPS"],
                "is-referenced-by-count": 50000,
                "link": [],
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        self.engine._session = mock_session

        info = self.engine.get_paper_info("10.1234/attention")
        assert info is not None
        assert "Attention" in info["title"]
        assert info["doi"] == "10.1234/attention"

    def test_get_paper_info_doi_url(self):
        """以 https://doi.org/ 形式的 DOI."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "title": ["Test Paper"],
                "author": [],
                "DOI": "10.1234/test",
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        self.engine._session = mock_session

        info = self.engine.get_paper_info("https://doi.org/10.1234/test")
        assert info is not None
        assert info["doi"] == "10.1234/test"

    # ── Session 管理 ──────────────────────────────────────────

    def test_close_session(self):
        """关闭 HTTP 会话."""
        self.engine._session = MagicMock()
        self.engine.close()
        self.engine._session.close.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# GoogleScholarSearch 测试
# ═══════════════════════════════════════════════════════════════════

class TestGoogleScholarSearch:
    """GoogleScholarSearch 引擎测试."""

    def setup_method(self):
        self.engine = GoogleScholarSearch(SAMPLE_CONFIG)

    # ── arXiv 提取 ────────────────────────────────────────────

    def test_extract_arxiv_from_text(self):
        """从摘要文本提取 arXiv ID."""
        text = "Preprint at arXiv:2301.00001 (2023)"
        assert GoogleScholarSearch._extract_arxiv_from_text(text) == "2301.00001"

    def test_extract_arxiv_from_url(self):
        """从 URL 提取 arXiv ID."""
        text = "https://arxiv.org/abs/1706.03762"
        assert GoogleScholarSearch._extract_arxiv_from_text(text) == "1706.03762"

    def test_extract_arxiv_none(self):
        """无 arXiv ID."""
        assert GoogleScholarSearch._extract_arxiv_from_text("No arxiv here") is None
        assert GoogleScholarSearch._extract_arxiv_from_text("") is None

    # ── 结果解析 ──────────────────────────────────────────────

    def test_parse_scholarly_pub(self):
        """解析 scholarly Publication 对象."""
        mock_pub = MagicMock()
        mock_pub.bib = {
            "title": "Deep Residual Learning",
            "author": "Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun",
            "pub_year": "2016",
            "abstract": "Deeper neural networks are more difficult to train...",
            "doi": "10.1109/CVPR.2016.90",
            "journal": "CVPR",
            "num_citations": 100000,
            "url": "https://ieeexplore.ieee.org/document/7780459",
        }
        mock_pub.eprint_url = "https://arxiv.org/abs/1512.03385"
        mock_pub.pub_url = "https://scholar.google.com/..."

        result = self.engine._parse_scholarly_pub(mock_pub)
        assert result["title"] == "Deep Residual Learning"
        assert len(result["authors"]) == 4
        assert result["authors"][0] == "Kaiming He"
        assert result["year"] == "2016"
        assert result["doi"] == "10.1109/CVPR.2016.90"
        assert result["citation_count"] == 100000
        assert result["arxiv_id"] is not None  # 从 eprint_url 提取

    def test_parse_scholarly_pub_minimal(self):
        """最简 scholarly 结果."""
        mock_pub = MagicMock()
        mock_pub.bib = {"title": "A Simple Paper"}
        mock_pub.eprint_url = ""
        mock_pub.pub_url = ""

        result = self.engine._parse_scholarly_pub(mock_pub)
        assert result["title"] == "A Simple Paper"
        assert result["authors"] == []
        assert result["year"] is None

    # ── 可用性检查 ────────────────────────────────────────────

    def test_available_flag(self):
        """检查 scholarly 是否可用."""
        # 仅检查属性存在且类型正确
        assert hasattr(self.engine, "_available")
        assert isinstance(self.engine._available, bool)


# ═══════════════════════════════════════════════════════════════════
# SearchFactory 测试
# ═══════════════════════════════════════════════════════════════════

class TestSearchFactory:
    """SearchFactory 多引擎编排与去重测试."""

    def setup_method(self):
        self.factory = SearchFactory(SAMPLE_CONFIG)

    # ── 引擎加载 ──────────────────────────────────────────────

    def test_load_engines_from_config(self):
        """根据配置加载引擎."""
        assert "arxiv" in self.factory.engines
        assert "crossref" in self.factory.engines
        assert isinstance(self.factory.engines["arxiv"], ArxivSearch)
        assert isinstance(self.factory.engines["crossref"], CrossrefSearch)

    def test_unknown_engine_skipped(self):
        """未知引擎被跳过."""
        cfg = dict(SAMPLE_CONFIG)
        cfg["search"] = {**cfg["search"], "engines": ["arxiv", "nonexistent"]}
        factory = SearchFactory(cfg)
        assert "arxiv" in factory.engines
        assert "nonexistent" not in factory.engines

    def test_engine_alias(self):
        """引擎别名解析."""
        cfg = dict(SAMPLE_CONFIG)
        cfg["search"] = {**cfg["search"], "engines": ["gs"]}
        factory = SearchFactory(cfg)
        assert "google_scholar" in factory.engines

    def test_get_engine(self):
        """获取已加载的引擎."""
        engine = self.factory.get_engine("arxiv")
        assert isinstance(engine, ArxivSearch)

    def test_get_engine_not_loaded(self):
        """获取未加载的引擎."""
        engine = self.factory.get_engine("google_scholar")
        assert engine is None

    def test_available_engines_property(self):
        """available_engines 属性."""
        assert self.factory.available_engines == ["arxiv", "crossref"]

    def test_list_registered_engines(self):
        """列出所有注册的引擎."""
        registered = SearchFactory.list_registered_engines()
        assert "arxiv" in registered
        assert "crossref" in registered
        assert "google_scholar" in registered

    # ── search_single ─────────────────────────────────────────

    def test_search_single(self):
        """单引擎直通搜索."""
        # 使用 mock 避免真实网络请求
        with patch.object(ArxivSearch, "search", return_value=[
            {"title": "Mock Result", "authors": [], "year": "2024", "source": "arxiv"}
        ]):
            results = self.factory.search_single("test", "arxiv")
            assert len(results) == 1
            assert results[0]["title"] == "Mock Result"

    def test_search_single_unavailable(self):
        """引擎不可用."""
        results = self.factory.search_single("test", "nonexistent")
        assert results == []

    # ── 并行搜索 ──────────────────────────────────────────────

    def test_parallel_search(self):
        """并行搜索多个引擎."""
        with patch.object(ArxivSearch, "search", return_value=[
            {"title": "Paper A", "authors": [], "year": "2024", "source": "arxiv", "doi": "10.a"}
        ]), patch.object(CrossrefSearch, "search", return_value=[
            {"title": "Paper B", "authors": [], "year": "2023", "source": "crossref", "doi": "10.b"}
        ]):
            results = self.factory.search_all("test", max_results=5)
            assert len(results) == 2
            titles = {r["title"] for r in results}
            assert titles == {"Paper A", "Paper B"}

    def test_parallel_search_one_engine_fails(self):
        """一个引擎失败不影响另一个."""
        with patch.object(ArxivSearch, "search", side_effect=RuntimeError("Boom")), \
             patch.object(CrossrefSearch, "search", return_value=[
                 {"title": "Paper B", "authors": [], "year": "2023", "source": "crossref", "doi": "10.b"}
             ]):
            results = self.factory.search_all("test", max_results=5)
            assert len(results) == 1
            assert results[0]["title"] == "Paper B"

    def test_search_all_no_engines(self):
        """没有可用引擎."""
        cfg = dict(SAMPLE_CONFIG)
        cfg["search"] = {**cfg["search"], "engines": []}
        factory = SearchFactory(cfg)
        results = factory.search_all("test")
        assert results == []

    def test_search_all_specific_engines(self):
        """指定引擎搜索."""
        with patch.object(ArxivSearch, "search", return_value=[
            {"title": "A", "authors": [], "year": "2024", "source": "arxiv", "doi": "10.a"}
        ]):
            results = self.factory.search_all("test", max_results=5, engines=["arxiv"])
            assert len(results) == 1
            assert results[0]["source"] == "arxiv"

    # ── 去重 ──────────────────────────────────────────────────

    def test_deduplicate_by_doi(self):
        """按 DOI 去重."""
        papers = [
            {"title": "Same Paper", "doi": "10.1234/same", "authors": ["A"], "year": "2024", "source": "arxiv"},
            {"title": "Same Paper Alt", "doi": "10.1234/same", "authors": ["B"], "year": "2024", "source": "crossref"},
        ]
        result = self.factory._deduplicate(papers)
        assert len(result) == 1
        # 合并后应包含两个来源的作者
        authors = [a.lower() for a in result[0]["authors"]]
        assert "a" in authors
        assert "b" in authors
        # 来源标记为合并
        assert "arxiv" in result[0]["source"]
        assert "crossref" in result[0]["source"]

    def test_deduplicate_by_arxiv_id(self):
        """按 arXiv ID 去重."""
        papers = [
            {"title": "Paper X", "arxiv_id": "2401.00001", "doi": None, "authors": ["A"], "year": "2024", "source": "arxiv"},
            {"title": "Paper X Variant", "arxiv_id": "2401.00001", "doi": "10.x", "authors": [], "year": "2024", "source": "crossref"},
        ]
        result = self.factory._deduplicate(papers)
        assert len(result) == 1
        # DOI 应从 secondary 合并到 primary
        assert result[0]["doi"] == "10.x"

    def test_deduplicate_by_title(self):
        """按标题去重（rapidfuzz 模糊匹配）."""
        papers = [
            {"title": "A Very Important Paper About Machine Learning", "doi": None, "authors": ["A"], "year": "2024", "source": "arxiv"},
            {"title": "A Very Important Paper about Machine Learning", "doi": None, "authors": [], "year": "2024", "source": "crossref"},
        ]
        result = self.factory._deduplicate(papers)
        # 标题相似度应 >= 90%
        assert len(result) == 1

    def test_deduplicate_distinct_papers(self):
        """不同论文保留."""
        papers = [
            {"title": "Paper Alpha", "doi": "10.a", "authors": [], "year": "2024", "source": "arxiv"},
            {"title": "Paper Beta", "doi": "10.b", "authors": [], "year": "2024", "source": "crossref"},
            {"title": "Paper Gamma", "doi": "10.c", "authors": [], "year": "2024", "source": "arxiv"},
        ]
        result = self.factory._deduplicate(papers)
        assert len(result) == 3

    def test_deduplicate_empty(self):
        """空列表."""
        assert self.factory._deduplicate([]) == []

    # ── 合并 ──────────────────────────────────────────────────

    def test_merge_preserves_primary_fields(self):
        """合并时保留 primary 的非空值."""
        primary = {"title": "Original", "doi": "10.orig", "authors": ["Alice"], "year": "2024"}
        secondary = {"title": "Overwritten?", "doi": "10.secondary", "authors": [], "abstract": "New"}
        merged = SearchFactory._merge_papers(primary, secondary)
        assert merged["title"] == "Original"  # primary 的非空值保留
        assert merged["doi"] == "10.orig"
        assert merged["abstract"] == "New"  # primary 缺失，用 secondary 填补

    def test_merge_citation_count_max(self):
        """引用计数取最大值."""
        primary = {"title": "P", "citation_count": 10}
        secondary = {"title": "P", "citation_count": 50}
        merged = SearchFactory._merge_papers(primary, secondary)
        assert merged["citation_count"] == 50

    def test_merge_null_citation(self):
        """一方引用计数为 None."""
        primary = {"title": "P", "citation_count": None}
        secondary = {"title": "P", "citation_count": 30}
        merged = SearchFactory._merge_papers(primary, secondary)
        assert merged["citation_count"] == 30

    # ── 排序 ──────────────────────────────────────────────────

    def test_sort_by_citations(self):
        """按引用数降序排列."""
        papers = [
            {"title": "A", "citation_count": 10},
            {"title": "B", "citation_count": 100},
            {"title": "C", "citation_count": None},
        ]
        result = self.factory._sort_results(papers, "citations")
        assert result[0]["title"] == "B"
        assert result[1]["title"] == "A"
        assert result[2]["title"] == "C"

    def test_sort_by_year(self):
        """按年份降序排列."""
        papers = [
            {"title": "Old", "year": "2018"},
            {"title": "New", "year": "2024"},
            {"title": "Mid", "year": "2021"},
        ]
        result = self.factory._sort_results(papers, "year")
        assert result[0]["title"] == "New"
        assert result[1]["title"] == "Mid"
        assert result[2]["title"] == "Old"

    # ── 标题标准化 ────────────────────────────────────────────

    def test_norm_title(self):
        """标题标准化去标点."""
        norm = SearchFactory._norm_title("Hello, World! A Test...")
        assert "hello" == norm.split()[0]
        assert "," not in norm
        assert "!" not in norm

    def test_norm_title_empty(self):
        """空标题."""
        assert SearchFactory._norm_title("") == ""
        assert SearchFactory._norm_title(None) == ""  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# 抽象基类约束
# ═══════════════════════════════════════════════════════════════════

class TestAbstractConstraints:
    """SearchEngine 抽象约束."""

    def test_cannot_instantiate_abstract(self):
        """直接实例化 SearchEngine 应抛出 TypeError."""
        with pytest.raises(TypeError):
            SearchEngine()  # type: ignore[abstract]

    def test_concrete_must_implement_abstract(self):
        """未实现 search/get_paper_info 的子类不能实例化."""

        class Incomplete(SearchEngine):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
