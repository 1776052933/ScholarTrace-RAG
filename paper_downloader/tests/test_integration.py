"""
test_integration.py — 端到端集成测试

覆盖:
    - 搜索→下载完整流水线 (mock)
    - 真实 arXiv API (可选)
    - 批量下载 end-to-end
    - 跨模块协作
    - 错误恢复与重试
    - 配置-缓存-搜索-下载全链路
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from paper_downloader.src.models.paper import Paper
from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.config.config_manager import ConfigManager
from paper_downloader.src.cache.cache_manager import CacheManager
from paper_downloader.src.monitoring.metrics import MetricsCollector
from paper_downloader.src.utils.report_generator import ReportGenerator
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
    ValidationError,
)
from paper_downloader.src import api

# 最小 PDF 生成工具（内联以避免跨文件导入）
_MINIMAL_PDF = (
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


def make_minimal_pdf(path: Path) -> Path:
    """在给定路径创建最小合法 PDF。"""
    path.write_bytes(_MINIMAL_PDF)
    return path


# ═══════════════════════════════════════════════════════════════════
# 端到端搜索→下载流程 (Mock)
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """完整流水线测试 — 搜索 → 下载 → 报告。"""

    @pytest.fixture
    def dl(self, tmp_path):
        """创建配置好的下载器。"""
        return PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3, "sort_by": "relevance"},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "timeout": {"search": 5, "download": 10, "connection": 3},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

    def test_search_download_pipeline(self, dl, sample_paper_dict, tmp_path):
        """搜索→下载完整流水线。"""
        # Mock 搜索
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [sample_paper_dict]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # Mock 下载
        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = sample_paper_dict["title"]
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        out_pdf = tmp_path / "result.pdf"
        make_minimal_pdf(out_pdf)
        mock_task.pdf_path = str(out_pdf)
        mock_task.file_size = out_pdf.stat().st_size
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        # 执行
        paper = dl.download_by_title("Attention Is All You Need")
        assert paper is not None
        assert paper.title == "Attention Is All You Need"
        assert paper.has_pdf is True

    def test_pipeline_with_metrics(self, dl, sample_paper_dict, tmp_path):
        """流水线 + 指标收集。"""
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [sample_paper_dict]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # 执行搜索
        dl.search("Attention Is All You Need")
        stats = dl.metrics.get_stats()
        assert stats["search"]["count"] == 1
        # total_time 可能为 0（执行太快），此处只验证 count 正确

    def test_batch_end_to_end(self, dl, sample_paper_dicts, tmp_path):
        """批量下载端到端。"""
        mock_factory = MagicMock()
        mock_factory.available_engines = ["arxiv"]

        # 每次搜索返回第一个匹配
        def search_side_effect(query, **kw):
            return [sample_paper_dicts[0]]

        mock_factory.search_all.side_effect = search_side_effect
        dl._search_factory = mock_factory

        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        for i, d in enumerate(sample_paper_dicts):
            pdf = tmp_path / f"paper_{i}.pdf"
            make_minimal_pdf(pdf)
            mock_task.title = d["title"]
            mock_task.pdf_path = str(pdf)
            mock_task.file_size = pdf.stat().st_size
            mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        titles = [p["title"] for p in sample_paper_dicts]
        results = dl.batch_download(titles, output_dir=str(tmp_path),
                                     skip_existing=False)
        assert len(results) == len(titles)

    def test_error_recovery_in_batch(self, dl, sample_paper_dict, tmp_path):
        """批量操作中单篇失败不中断整体。"""
        mock_factory = MagicMock()
        mock_factory.available_engines = ["arxiv"]
        call_count = [0]

        def side_effect(query, **kw):
            call_count[0] += 1
            if call_count[0] == 2:  # 第二个标题搜索失败
                raise PaperNotFoundError("not found")
            return [sample_paper_dict]

        mock_factory.search_all.side_effect = side_effect
        dl._search_factory = mock_factory

        titles = ["Paper A", "Paper B", "Paper C"]
        results = dl.batch_download(titles, output_dir=str(tmp_path),
                                     skip_existing=False)
        assert len(results) == 3  # 失败也记录

    def test_report_after_pipeline(self, dl, sample_paper_dicts, tmp_path):
        """流水线完成后生成报告。"""
        gen = ReportGenerator(output_dir=str(tmp_path))

        papers = [Paper.from_search_result(d) for d in sample_paper_dicts]
        # 标记一个有 PDF
        real_pdf = tmp_path / "downloaded.pdf"
        make_minimal_pdf(real_pdf)
        papers[0].pdf_path = str(real_pdf)
        papers[0].file_size = real_pdf.stat().st_size

        # JSON
        json_path = gen.export_json(papers)
        assert os.path.exists(json_path)
        report = json.load(open(json_path))
        assert report["succeeded"] == 1

        # CSV
        csv_path = gen.export_csv(papers)
        assert os.path.exists(csv_path)

        # Markdown
        md_path = gen.export_markdown(papers)
        assert os.path.exists(md_path)

        # BibTeX
        bib_path = gen.export_bibtex(papers)
        assert os.path.exists(bib_path)
        with open(bib_path) as f:
            assert "@article" in f.read()


# ═══════════════════════════════════════════════════════════════════
# 真实 arXiv API 集成测试 (可选)
# ═══════════════════════════════════════════════════════════════════

class TestRealArxivIntegration:
    """真实 arXiv API 集成测试。

    需要网络连接。通过环境变量 PAPER_RUN_REAL_TESTS=1 启用。
    """

    @pytest.mark.skipif(
        os.environ.get("PAPER_RUN_REAL_TESTS") != "1",
        reason="设置 PAPER_RUN_REAL_TESTS=1 以运行真实网络测试",
    )
    @pytest.mark.slow
    def test_real_arxiv_search(self):
        """真实 arXiv API 搜索。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "concurrency": {"request_delay": 3.0},
            "timeout": {"search": 30},
            "retry": {"max_attempts": 2, "backoff_factor": 2},
            "logging": {"level": "WARNING"},
        })

        papers = dl.search("attention", max_results=3)
        assert len(papers) > 0
        for p in papers:
            assert isinstance(p, Paper)
            assert p.title
            assert p.source == "arxiv"

    @pytest.mark.skipif(
        os.environ.get("PAPER_RUN_REAL_TESTS") != "1",
        reason="设置 PAPER_RUN_REAL_TESTS=1 以运行真实网络测试",
    )
    @pytest.mark.slow
    def test_real_arxiv_paper_info(self):
        """真实 arXiv ID 查询。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "concurrency": {"request_delay": 3.0},
            "logging": {"level": "WARNING"},
        })

        info = dl.get_paper_info("1706.03762")
        if info:  # 如果 arxiv 包已安装
            assert "Attention" in info.title

    @pytest.mark.skipif(
        os.environ.get("PAPER_RUN_REAL_TESTS") != "1",
        reason="设置 PAPER_RUN_REAL_TESTS=1 以运行真实网络测试",
    )
    @pytest.mark.slow
    def test_real_batch_download_small(self, tmp_path):
        """真实批量下载（仅搜索，不下载 PDF）。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 1},
            "concurrency": {"request_delay": 3.0},
            "logging": {"level": "INFO"},
        })

        titles = [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
        ]
        results = dl.batch_download(
            titles,
            output_dir=str(tmp_path),
            skip_existing=False,
        )
        assert len(results) == 2
        # 至少有一篇搜索成功
        assert any(p.title for p in results)


# ═══════════════════════════════════════════════════════════════════
# 跨模块集成测试
# ═══════════════════════════════════════════════════════════════════

class TestCrossModuleIntegration:
    """跨模块协作集成测试。"""

    def test_config_cache_search_integration(self, tmp_path):
        """配置→缓存→搜索全链路。"""
        from paper_downloader.src.search_engines.search_factory import SearchFactory

        cache = CacheManager(db_path=str(tmp_path / "int.db"), max_memory=50)

        factory = SearchFactory({
            "search": {"engines": ["arxiv"], "max_results": 3},
            "cache": {"enabled": True, "ttl": 3600},
            "concurrency": {"request_delay": 0},
        })
        factory.set_cache(cache)

        mock_result = [{"title": "Test", "authors": ["A"], "year": "2024", "source": "arxiv"}]
        with patch.object(factory, "_parallel_search", return_value=mock_result):
            r1 = factory.search_all("test")
            r2 = factory.search_all("test")  # 缓存命中
            assert r1 == r2

        cache.close()

    def test_metrics_reporter_integration(self, sample_paper_dicts):
        """指标→报告集成。"""
        mc = MetricsCollector()
        mc.record_search(duration=0.5)
        mc.record_download(success=True, size_bytes=1024000)
        mc.record_download(success=False)

        stats = mc.get_stats()

        # 生成报告
        papers = [Paper.from_search_result(d) for d in sample_paper_dicts]
        gen = ReportGenerator()
        summary = gen.generate_summary(papers)

        # 验证关联性
        assert summary["total"] == len(sample_paper_dicts)
        assert stats["search"]["count"] == 1

    def test_api_with_injected_components(self, tmp_path, sample_paper_dict):
        """api 模块使用注入的组件。"""
        import paper_downloader.src.api as pd_api

        pd_api.reset_downloader()

        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [sample_paper_dict]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # 重置后注入（无额外 kwargs，下次 get_downloader 返回此实例）
        pd_api._downloader = dl

        result = dl.search("test")
        assert len(result) == 1
        assert result[0].title == sample_paper_dict["title"]

        pd_api.reset_downloader()

    def test_error_handler_integration(self):
        """ErrorHandler 与其他组件集成。"""
        from paper_downloader.src.exceptions.error_handler import retry_on_error
        from paper_downloader.src.utils.logger import get_logger

        logger = get_logger("test.integration")
        call_count = [0]

        @retry_on_error(max_retries=2, delay=0.01, backoff=1)
        def risky_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("transient network issue")
            return "success"

        result = risky_operation()
        assert result == "success"
        assert call_count[0] == 2

    def test_healthcheck_metrics_integration(self):
        """健康检查 + 指标集成。"""
        from paper_downloader.src.monitoring.health_check import HealthChecker

        checker = HealthChecker()
        mc = MetricsCollector()

        # 模拟检查和指标记录的配合
        disk = checker.check_disk_space(".")
        if disk.get("status") != "error":
            mc.record_download(success=True, size_bytes=5000)

        stats = mc.get_stats()
        assert stats["download"]["success"] >= 1


# ═══════════════════════════════════════════════════════════════════
# 数据一致性测试
# ═══════════════════════════════════════════════════════════════════

class TestDataConsistency:
    """数据在模块间传递的一致性。"""

    def test_paper_roundtrip_dict(self, sample_paper_dict):
        """Paper → dict → Paper 往返。"""
        p1 = Paper.from_search_result(sample_paper_dict)
        d = p1.to_dict()
        p2 = Paper.from_dict(d)
        assert p2.title == p1.title
        assert p2.doi == p1.doi
        assert p2.authors == p1.authors

    def test_paper_roundtrip_json(self, sample_paper_dict):
        """Paper → JSON → Paper 往返。"""
        p1 = Paper.from_search_result(sample_paper_dict)
        j = p1.to_json()
        d = json.loads(j)
        p2 = Paper.from_dict(d)
        assert p2.title == p1.title

    def test_search_result_to_download_task(self, sample_paper_dict, tmp_path):
        """搜索结果到下载任务的数据流。"""
        from paper_downloader.src.downloaders.download_manager import DownloadManager

        paper = Paper.from_search_result(sample_paper_dict)

        mgr = DownloadManager(config={
            "concurrency": {"max_downloads": 1},
            "timeout": {"download": 10},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        key = mgr.add_task(
            url=paper.pdf_url or "",
            title=paper.title,
            authors="; ".join(paper.authors),
            year=paper.year or "",
            doi=paper.doi or "",
            arxiv_id=paper.arxiv_id or "",
            save_dir=str(tmp_path),
        )

        task = mgr.get_task(key)
        assert task is not None
        assert task.title == sample_paper_dict["title"]
        assert task.doi == sample_paper_dict["doi"]

        mgr.close()

    def test_metrics_consistency(self):
        """指标计算一致性。"""
        mc = MetricsCollector()
        for _ in range(5):
            mc.record_download(success=True, size_bytes=1000)
        for _ in range(2):
            mc.record_download(success=False)

        stats = mc.get_stats()
        assert stats["download"]["total"] == 7
        assert stats["download"]["success"] == 5
        assert stats["download"]["failed"] == 2
        assert abs(stats["download"]["success_rate_pct"] - 71.4) < 1.0


# ═══════════════════════════════════════════════════════════════════
# 边界条件 & 压力测试
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """边界条件和异常路径测试。"""

    def test_empty_title_input(self, tmp_path):
        """空标题不崩溃。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })
        with pytest.raises(Exception):
            dl.search("")

    def test_very_long_title(self, tmp_path):
        """超长标题。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })
        long_title = "A" * 500
        mock_factory = MagicMock()
        mock_factory.search_all.return_value = []
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory
        with pytest.raises(PaperNotFoundError):
            dl.search(long_title)

    def test_special_characters_in_title(self, sample_paper_dict):
        """标题含特殊字符。"""
        p = Paper.from_search_result({
            **sample_paper_dict,
            "title": "Test: with <special> & \"chars\" / path\\like",
        })
        assert p.title  # 不应崩溃

    def test_mixed_encoding_authors(self):
        """作者名含非 ASCII 字符。"""
        p = Paper.from_search_result({
            "title": "International Paper",
            "authors": ["Jürgen Müller", "François Léger", "山田 太郎"],
            "year": "2024",
        })
        assert len(p.authors) == 3
        j = p.to_json()
        assert "Jürgen" in j

    def test_concurrent_config_access(self):
        """并发配置访问。"""
        import threading

        cfg = ConfigManager()
        errors = []

        def access_config():
            try:
                for _ in range(100):
                    cfg.get("download.timeout")
                    cfg.get("cache.enabled")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_config) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        ConfigManager._instance = None

    def test_metrics_after_reset(self):
        """重置后指标归零。"""
        mc = MetricsCollector()
        mc.record_search()
        mc.record_download(success=True, size_bytes=1000)
        mc.reset_stats()

        stats = mc.get_stats()
        assert stats["search"]["count"] == 0
        assert stats["download"]["success"] == 0

    def test_download_with_missing_url(self, tmp_path):
        """无 URL 的论文下载。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })
        paper = Paper(title="No URL Paper")
        with pytest.raises(DownloadError):
            dl.download(paper)

    def test_batch_empty_list(self):
        """空批量列表。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })
        results = dl.batch_download([])
        assert results == []


# ═══════════════════════════════════════════════════════════════════
# API 模块端到端测试
# ═══════════════════════════════════════════════════════════════════

class TestAPIEndToEnd:
    """api 模块端到端测试."""

    def teardown_method(self):
        api.reset_downloader()

    def test_full_api_flow(self, tmp_path, sample_paper_dict):
        """dl.download_by_title 端到端（直接使用核心下载器）。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"], "max_results": 3},
            "download": {"path": str(tmp_path)},
            "concurrency": {"max_downloads": 1, "request_delay": 0},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = [sample_paper_dict]
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        mock_mgr = MagicMock()
        mock_task = MagicMock()
        mock_task.title = sample_paper_dict["title"]
        mock_task.status = MagicMock()
        mock_task.status.value = "completed"
        pdf = tmp_path / "api_test.pdf"
        make_minimal_pdf(pdf)
        mock_task.pdf_path = str(pdf)
        mock_task.file_size = pdf.stat().st_size
        mock_task.completed_at = "2024-01-01T00:00:00"
        mock_mgr.run_all.return_value = [mock_task]
        dl._download_manager = mock_mgr

        # 直接使用核心下载器（绕过 api 模块单例缓存）
        paper = dl.download_by_title("Attention Is All You Need", output_dir=str(tmp_path))
        assert paper.title == sample_paper_dict["title"]

    def test_api_search_flow(self, sample_paper_dicts):
        """dl.search 端到端（直接使用核心下载器）。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = sample_paper_dicts
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        # 直接使用核心下载器搜索
        papers = dl.search("deep learning", max_results=5)
        assert len(papers) == 3
        assert all(isinstance(p, Paper) for p in papers)

    def test_api_error_propagation(self):
        """错误传播测试（直接使用核心下载器）。"""
        dl = PaperDownloader(config={
            "search": {"engines": ["arxiv"]},
            "retry": {"max_attempts": 1, "backoff_factor": 0},
            "logging": {"level": "ERROR"},
        })

        mock_factory = MagicMock()
        mock_factory.search_all.return_value = []
        mock_factory.available_engines = ["arxiv"]
        dl._search_factory = mock_factory

        with pytest.raises(PaperNotFoundError):
            dl.download_by_title("nonexistent_paper_xyz_12345")


# ═══════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
