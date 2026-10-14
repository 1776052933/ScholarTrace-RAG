"""
core/downloader.py — 核心论文下载器

整合搜索引擎、下载管理器和 PDF 处理器，
提供一站式的论文搜索与下载编排能力。
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from paper_downloader.src.models.paper import Paper
from paper_downloader.src.search_engines.search_factory import SearchFactory
from paper_downloader.src.downloaders.download_manager import DownloadManager
from paper_downloader.src.downloaders.pdf_processor import PDFProcessor
from paper_downloader.src.downloaders.arxiv_downloader import ArxivPDFDownloader
from paper_downloader.src.downloaders.source_resolver import DownloadSourceResolver
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
    ValidationError,
    ConfigError,
    SearchError,
)
from paper_downloader.src.exceptions.error_handler import retry_on_error
from paper_downloader.src.monitoring.metrics import MetricsCollector
from paper_downloader.src.utils.pdf_identity import title_similarity


class PaperDownloader:
    """核心论文下载器。

    整合多引擎搜索 → PDF 下载 → 后处理的全流程，
    是 paper_downloader 模块对外的主要入口。

    功能:
        - 按标题/关键词搜索论文
        - 单篇 / 批量下载 PDF
        - 自动重命名和元数据提取
        - 进度回调（用于 GUI / Web 集成）

    Usage::

        # 最简用法
        from paper_downloader.src.core.downloader import PaperDownloader

        dl = PaperDownloader()
        paper = dl.download_by_title("Attention Is All You Need")

        # 批量下载
        papers = dl.batch_download(
            ["GPT-4 Technical Report", "BERT: Pre-training of Deep Bidirectional Transformers"],
            output_dir="./my_papers",
        )
    """

    # 默认配置文件路径
    _DEFAULT_CONFIG_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "config",
    )
    _DEFAULT_CONFIG_FILE = os.path.join(_DEFAULT_CONFIG_DIR, "config.yaml")

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """初始化下载器。

        Args:
            config_path: YAML 配置文件路径。为 None 时使用默认配置。
            config:      直接传入配置字典（优先级高于 config_path）。
            **kwargs:    覆盖配置中的特定字段，如 engines=["arxiv"]。
        """
        # 加载配置
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = self._load_yaml(config_path)
        else:
            self._config = self._load_default_config()

        # kwargs 覆盖
        self._apply_kwargs(kwargs)

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 组件（延迟初始化）
        self._search_factory: Optional[SearchFactory] = None
        self._download_manager: Optional[DownloadManager] = None
        self._source_resolver: Optional[DownloadSourceResolver] = None
        self._progress_callback: Optional[Callable[[Paper], None]] = None

        # 监控指标
        self._metrics = MetricsCollector()

        self.logger.info("PaperDownloader 初始化完成 (engines=%s)",
                         self._config.get("search", {}).get("engines", []))

    # ── 配置加载 ──────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件。"""
        if not os.path.exists(path):
            raise ConfigError(f"配置文件不存在: {path}", config_path=path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置文件。"""
        default_path = os.path.normpath(self._DEFAULT_CONFIG_FILE)
        if os.path.exists(default_path):
            return self._load_yaml(default_path)
        # 配置文件不存在时返回最小有效配置
        return {
            "search": {"engines": ["arxiv", "crossref"], "max_results": 10, "sort_by": "relevance"},
            "download": {"path": "./papers", "filename_template": "{first_author}_{year}_{title}"},
            "concurrency": {"max_downloads": 3, "request_delay": 1.0},
            "timeout": {"search": 30, "download": 120, "connection": 10},
            "proxy": {"enabled": False},
            "retry": {"max_attempts": 3, "backoff_factor": 2},
            "logging": {"level": "INFO", "file": "./logs/paper_downloader.log"},
        }

    def _apply_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """将 kwargs 合并到配置中。"""
        for key, value in kwargs.items():
            if key in ("engines", "max_results", "sort_by", "min_year"):
                self._config.setdefault("search", {})[key] = value
            elif key in ("max_downloads", "request_delay"):
                self._config.setdefault("concurrency", {})[key] = value
            elif key in ("output_dir", "download_path"):
                self._config.setdefault("download", {})["path"] = value
            elif key == "filename_template":
                self._config.setdefault("download", {})[key] = value
            elif key == "proxy":
                self._config.setdefault("proxy", {}).update(value)

    def _init_search_factory(self) -> SearchFactory:
        """延迟初始化搜索引擎工厂。"""
        if self._search_factory is None:
            self._search_factory = SearchFactory(self._config)
        return self._search_factory

    def _init_download_manager(self) -> DownloadManager:
        """延迟初始化下载管理器。"""
        if self._download_manager is None:
            self._download_manager = DownloadManager(
                config=self._config,
                max_workers=int(self._config.get("concurrency", {}).get("max_downloads", 3)),
            )
        return self._download_manager

    def _init_source_resolver(self) -> DownloadSourceResolver:
        """延迟初始化开放获取 PDF 地址解析器。"""
        if self._source_resolver is None:
            self._source_resolver = DownloadSourceResolver(self._config)
        return self._source_resolver

    # ── 搜索 ──────────────────────────────────────────────────────

    def search(
        self,
        title: str,
        max_results: int = 5,
        engines: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Paper]:
        """搜索论文，返回标准化的 Paper 对象列表。

        Args:
            title:       论文标题或关键词。
            max_results: 每个引擎的最大返回数量。
            engines:     指定搜索引擎列表，None 使用配置中的默认引擎。
            **kwargs:    传递给 SearchFactory.search_all() 的额外参数。

        Returns:
            Paper 对象列表。

        Raises:
            SearchError: 所有引擎均搜索失败。
        """
        if not title or not title.strip():
            raise ValidationError("搜索标题或关键词不能为空")

        factory = self._init_search_factory()
        self.logger.info("搜索: '%s' (engines=%s, max_results=%d)",
                         title, engines or "default", max_results)

        start = time.time()
        results = factory.search_all(
            query=title,
            max_results=max_results,
            engines=engines,
            **kwargs,
        )

        if not results:
            raise PaperNotFoundError(
                f"未找到论文: '{title}'",
                query=title,
                engines=engines or factory.available_engines,
            )

        papers = [Paper.from_search_result(r) for r in results]
        elapsed = time.time() - start
        self._metrics.record_search(duration=elapsed)
        self.logger.info("搜索完成: %d 篇论文", len(papers))
        return papers

    # ── 下载 ──────────────────────────────────────────────────────

    def download(
        self,
        papers: Union[Paper, List[Paper], Dict[str, Any], List[Dict[str, Any]]],
        output_dir: Optional[str] = None,
        rename: bool = True,
        **kwargs: Any,
    ) -> List[Paper]:
        """下载单篇或多篇论文的 PDF。

        Args:
            papers:     Paper 对象 / dict / 列表。支持多种输入格式。
            output_dir: 输出目录，默认使用配置中的 download.path。
            rename:     是否按规范重命名下载的 PDF。默认 True。
            **kwargs:   传递给 DownloadManager 的额外参数。

        Returns:
            更新后的 Paper 列表（含 pdf_path 字段）。

        Raises:
            DownloadError: 全部下载均失败时抛出。
        """
        # 统一输入格式
        papers_list = self._normalize_paper_input(papers)
        if not papers_list:
            raise ValidationError("没有可下载的论文（输入为空）")

        output_dir = output_dir or self._config.get("download", {}).get("path", "./papers")

        self.logger.info("开始下载 %d 篇论文 → %s", len(papers_list), output_dir)

        manager = self._init_download_manager()
        resolver = self._init_source_resolver()

        # 添加任务
        for paper in papers_list:
            candidates = resolver.resolve(paper)
            if not candidates:
                self.logger.warning("所有下载引擎均未解析到 PDF 地址，跳过: %s", paper.title)
                continue

            filename = None
            if rename:
                filename = self._config.get("download", {}).get(
                    "filename_template", "{first_author}_{year}_{title}"
                )
                filename = (
                    filename.replace("{first_author}", paper.first_author_surname)
                    .replace("{year}", paper.year or "nodate")
                    .replace("{title}", paper.title[:80])
                )
                # 清理非法字符（Windows 兼容：? * : " < > | 等）
                filename = self._sanitize_filename(filename)

            manager.add_task(
                url=candidates[0].url,
                candidate_urls=[candidate.url for candidate in candidates[1:]],
                title=paper.title,
                authors="; ".join(paper.authors),
                year=paper.year or "",
                doi=paper.doi or "",
                arxiv_id=paper.arxiv_id or "",
                save_dir=output_dir,
                filename=filename,
            )

        # 设置进度回调
        if self._progress_callback:

            def _adapter(task: Any) -> None:
                for p in papers_list:
                    if p.title == task.title:
                        p.pdf_path = task.pdf_path
                        p.file_size = task.file_size
                        p.downloaded_at = task.completed_at
                        if self._progress_callback:
                            self._progress_callback(p)
                        return

            manager.set_progress_callback(_adapter)

        # 执行下载
        tasks = manager.run_all()

        # 回填结果到 Paper 对象
        succeeded = 0
        for paper in papers_list:
            # 查找匹配的任务
            for task in tasks:
                if (task.title == paper.title and
                        task.status.value in ("completed",) and
                        task.pdf_path):
                    paper.pdf_path = task.pdf_path
                    paper.file_size = task.file_size
                    paper.downloaded_at = task.completed_at
                    succeeded += 1
                    break

            # 尝试 arXiv 专用下载器回退
            if not paper.pdf_path and paper.arxiv_id:
                try:
                    arxiv_dl = ArxivPDFDownloader(self._config)
                    pdf_path = arxiv_dl.download(paper.arxiv_id, save_path=output_dir)
                    if pdf_path and pdf_path.exists():
                        paper.pdf_path = str(pdf_path)
                        paper.file_size = pdf_path.stat().st_size
                        paper.downloaded_at = datetime.now().isoformat()
                        succeeded += 1
                except Exception:
                    pass

        self.logger.info("下载完成: %d/%d 成功", succeeded, len(papers_list))

        # 记录指标
        for paper in papers_list:
            self._metrics.record_download(
                success=paper.has_pdf,
                size_bytes=paper.file_size,
            )

        if succeeded == 0 and len(papers_list) > 0:
            raise DownloadError("所有论文下载均失败", attempt_count=0)

        return papers_list

    def _normalize_paper_input(
        self,
        papers: Union[Paper, List[Paper], Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Paper]:
        """统一输入为 Paper 列表。"""
        if isinstance(papers, Paper):
            return [papers]
        if isinstance(papers, list):
            if not papers:
                return []
            if isinstance(papers[0], Paper):
                return papers  # type: ignore[return-value]
            if isinstance(papers[0], dict):
                return [Paper.from_search_result(p) for p in papers]  # type: ignore[arg-type]
        if isinstance(papers, dict):
            return [Paper.from_search_result(papers)]
        raise ValidationError(f"不支持的 papers 类型: {type(papers)}")

    # ── 便捷接口 ──────────────────────────────────────────────────

    @staticmethod
    def _find_best_match(
        query: str,
        papers: List[Paper],
        min_similarity: float = 0.55,
        strict: bool = False,
    ) -> tuple[Optional[Paper], float]:
        """用标题相似度找到最佳匹配。

        使用 rapidfuzz 计算每个候选与查询的相似度，
        低于阈值的视为不相关。

        Args:
            query:          用户输入的标题/查询。
            papers:         搜索结果列表。
            min_similarity: 最低相似度阈值（0~1）。
            strict:         严格模式 — 排除 partial_ratio（避免短关键词
                            匹配长标题中的子串）。下载论文时建议开启。

        Returns:
            (best_paper, similarity_score): best_paper 为 None 表示无合格匹配。
        """
        if not papers:
            return None, 0.0

        q = query.lower().strip()
        import re
        q_clean = re.sub(r'[^\w\s]', '', q).strip()

        best: Optional[Paper] = None
        best_score = 0.0

        try:
            from rapidfuzz import fuzz as rfuzz

            for paper in papers:
                t = paper.title.lower().strip()
                t_clean = re.sub(r'[^\w\s]', '', t).strip()

                # 严格模式: 只用 ratio + token_sort，排除 partial_ratio
                if strict:
                    scores = [
                        rfuzz.ratio(q, t),
                        rfuzz.token_sort_ratio(q, t),
                    ]
                else:
                    scores = [
                        rfuzz.ratio(q, t),
                        rfuzz.partial_ratio(q, t),
                        rfuzz.token_sort_ratio(q, t),
                    ]
                if q_clean and t_clean:
                    scores.append(rfuzz.ratio(q_clean, t_clean))

                score = max(scores) / 100.0

                if score > best_score:
                    best_score = score
                    best = paper
        except ImportError:
            for paper in papers:
                score = title_similarity(query, paper.title)
                if score > best_score:
                    best_score = score
                    best = paper

        if best_score < min_similarity:
            return None, best_score

        return best, best_score

    def download_by_title(
        self,
        title: str,
        output_dir: Optional[str] = None,
        max_results: int = 3,
        engines: Optional[List[str]] = None,
        rename: bool = True,
        match_threshold: float = 0.82,
        **kwargs: Any,
    ) -> Paper:
        """通过论文标题直接搜索并下载 PDF（最常用接口）。

        使用标题相似度匹配确保下载正确的论文。

        Args:
            title:           论文标题。
            output_dir:      PDF 输出目录。
            max_results:     搜索候选数（取最佳匹配）。
            engines:         搜索引擎列表。
            rename:          是否重命名 PDF。
            match_threshold: 标题匹配阈值（0~1），低于此值视为不匹配。默认 0.82。
            **kwargs:        传递给 search() 的额外参数。

        Returns:
            下载成功的 Paper 对象。

        Raises:
            PaperNotFoundError: 搜索无结果或无匹配标题。
            DownloadError:      下载失败。
        """
        self.logger.info("=" * 50)
        self.logger.info("下载论文: %s", title)

        # 搜索
        papers = self.search(title, max_results=max(max_results, 5), engines=engines, **kwargs)

        # 过滤有 PDF URL 的
        downloadable = [p for p in papers if p.pdf_url or p.arxiv_id or p.doi]
        if not downloadable:
            downloadable = papers

        # 取最佳匹配（按标题相似度）
        best, score = self._find_best_match(title, downloadable, min_similarity=match_threshold, strict=True)
        if best is None:
            closest_title = downloadable[0].title[:80] if downloadable else "N/A"
            raise PaperNotFoundError(
                f"未找到标题匹配的论文 (相似度 {score:.0%} < {match_threshold:.0%})\n"
                f"  查询: '{title}'\n"
                f"  最近似: '{closest_title}'",
                query=title,
            )

        self.logger.info("最佳匹配 (%.0%%): %s (%s)", score * 100, best.title, best.identifier)

        # 下载
        results = self.download(best, output_dir=output_dir, rename=rename)
        return results[0]

    def batch_download(
        self,
        titles: List[str],
        output_dir: Optional[str] = None,
        max_results: int = 3,
        engines: Optional[List[str]] = None,
        rename: bool = True,
        concurrent_searches: int = 3,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> List[Paper]:
        """批量下载多篇论文。

        支持并发搜索、断点续传（跳过已下载）、失败记录。

        Args:
            titles:             论文标题列表。
            output_dir:         PDF 输出目录。
            max_results:        每个标题的搜索候选数。
            engines:            搜索引擎列表。
            rename:             是否重命名。
            concurrent_searches: 并发搜索线程数（默认 3）。
            skip_existing:      跳过输出目录中已有的 PDF。默认 True。
            **kwargs:           传递给 search() 的额外参数。

        Returns:
            Paper 对象列表（包含成功和失败的）。
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from paper_downloader.src.core.progress_tracker import ProgressTracker
        from paper_downloader.src.core.callback_manager import CallbackEvent

        total = len(titles)
        output_dir = output_dir or self._config.get("download", {}).get("path", "./papers")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("批量下载 %d 篇论文 (并发搜索=%d, skip_existing=%s)",
                         total, concurrent_searches, skip_existing)

        tracker = ProgressTracker(total=total, description="Batch Download")
        tracker.start()

        # 预扫描已下载的
        existing_pdfs: set[str] = set()
        if skip_existing:
            out = Path(output_dir)
            if out.exists():
                existing_pdfs = {p.stem.lower() for p in out.glob("*.pdf")}
            if existing_pdfs:
                self.logger.info("检测到 %d 个已有 PDF，将跳过匹配的标题", len(existing_pdfs))

        # 第一阶段：并发搜索
        self.logger.info("阶段 1/2: 并发搜索 %d 个标题 (workers=%d)", total, concurrent_searches)
        search_results: Dict[int, List[Paper]] = {}
        search_errors: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=concurrent_searches) as executor:
            futures = {}
            for i, title in enumerate(titles):
                # 跳过已有 PDF
                if skip_existing and existing_pdfs:
                    import re
                    title_stem = re.sub(r'[<>:"/\\|?*]', '_', title[:50]).strip().lower()
                    if any(title_stem in pdf for pdf in existing_pdfs):
                        tracker.skip(reason=f"已有PDF: {title[:50]}")
                        search_results[i] = [Paper(title=title)]
                        continue

                future = executor.submit(
                    self._safe_search, title, max_results, engines, **kwargs,
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    papers = future.result()
                    if papers:
                        search_results[idx] = papers
                        self.logger.info("[%d/%d] 搜索完成: %s", idx + 1, total, papers[0].title[:60])
                    else:
                        search_results[idx] = [Paper(title=titles[idx])]
                        search_errors[idx] = "未找到"
                except Exception as exc:
                    search_results[idx] = [Paper(title=titles[idx])]
                    search_errors[idx] = str(exc)
                    self.logger.warning("[%d/%d] 搜索失败: %s — %s", idx + 1, total, titles[idx], exc)

        # 第二阶段：收集最佳匹配并下载
        self.logger.info("阶段 2/2: 下载 %d 篇论文", len(search_results))
        to_download: List[Paper] = []
        result_papers: List[Paper] = []

        for i in range(total):
            if i in search_results and search_results[i]:
                best = search_results[i][0]
                if best.pdf_url or best.arxiv_id or best.doi:
                    to_download.append(best)
                    result_papers.append(best)
                else:
                    # 无可用URL，保留但标记
                    result_papers.append(best)
                    tracker.update(success=False, message=f"无可用URL: {best.title[:50]}")
            else:
                p = Paper(title=titles[i])
                result_papers.append(p)
                tracker.update(success=False, message=f"搜索失败: {titles[i][:50]}")

        # 统一下载
        if to_download:
            try:
                downloaded = self.download(
                    to_download,
                    output_dir=output_dir,
                    rename=rename,
                )
                # 回填结果
                dl_by_title = {p.title: p for p in downloaded}
                for i, paper in enumerate(result_papers):
                    if paper.title in dl_by_title and dl_by_title[paper.title].pdf_path:
                        updated = dl_by_title[paper.title]
                        result_papers[i] = updated
                        tracker.update(message=updated.title)
                    elif paper.pdf_url or paper.arxiv_id:
                        tracker.update(success=False, message=f"下载失败: {paper.title[:50]}")
                    else:
                        tracker.update(success=False, message=f"跳过: {paper.title[:50]}")
            except DownloadError as exc:
                self.logger.error("批量下载阶段失败: %s", exc)
                for paper in to_download:
                    tracker.update(success=False, message=f"下载错误: {paper.title[:50]}")

        # 汇总
        summary = tracker.finish()
        succeeded = summary["completed"] - summary["failed"]
        self.logger.info("批量下载完成: %d/%d 成功, %d 失败, %d 跳过",
                         succeeded, total, summary["failed"], summary["skipped"])

        if self._progress_callback:
            self._progress_callback(result_papers[-1] if result_papers else Paper())

        return result_papers

    def _safe_search(
        self,
        title: str,
        max_results: int,
        engines: Optional[List[str]],
        **kwargs: Any,
    ) -> List[Paper]:
        """安全的搜索包装，异常不中断。"""
        try:
            return self.search(title, max_results=max_results, engines=engines, **kwargs)
        except PaperNotFoundError:
            return []
        except Exception:
            return []

    # ── 指标 ──────────────────────────────────────────────────────

    @property
    def metrics(self) -> MetricsCollector:
        """返回监控指标收集器。"""
        return self._metrics

    # ── 进度回调 ──────────────────────────────────────────────────

    def set_progress_callback(self, callback: Optional[Callable[[Paper], None]]) -> None:
        """设置进度回调函数。

        每当一篇论文下载完成时调用，传入更新后的 Paper 对象。
        适用于 GUI 进度条、WebSocket 推送等场景。

        Args:
            callback: 回调函数，签名为 (Paper) -> None。
                      传 None 取消回调。

        Example::

            def on_progress(paper: Paper):
                print(f"完成: {paper.title} → {paper.pdf_path}")

            dl.set_progress_callback(on_progress)
        """
        self._progress_callback = callback

    # ── 工具 ──────────────────────────────────────────────────────

    @property
    def config(self) -> Dict[str, Any]:
        """返回当前配置的只读副本。"""
        import copy
        return copy.deepcopy(self._config)

    def get_paper_info(self, identifier: str) -> Optional[Paper]:
        """根据 DOI 或 arXiv ID 获取论文信息。

        Args:
            identifier: DOI（如 "10.1038/nature14539"）或
                        arXiv ID（如 "2401.00001"）。

        Returns:
            Paper 对象，未找到返回 None。
        """
        factory = self._init_search_factory()
        # 尝试多个引擎
        for engine_name in factory.available_engines:
            engine = factory.get_engine(engine_name)
            if engine is None:
                continue
            info = engine.get_paper_info(identifier)
            if info:
                return Paper.from_search_result(info)
        return None

    @staticmethod
    def _sanitize_filename(name: str, max_len: int = 200) -> str:
        """清理文件名中的 Windows/macOS/Linux 非法字符。"""
        import re
        illegal = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(illegal, "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = re.sub(r"_(\.\w+)$", r"\1", sanitized)
        sanitized = sanitized.strip("_ .")
        if not sanitized:
            sanitized = "paper"
        if len(sanitized) > max_len:
            dot_idx = sanitized.rfind(".")
            if dot_idx > 0:
                ext = sanitized[dot_idx:]
                sanitized = sanitized[:max_len - len(ext)] + ext
            else:
                sanitized = sanitized[:max_len]
        return sanitized

    def __enter__(self) -> "PaperDownloader":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._download_manager:
            self._download_manager.close()
        if self._source_resolver:
            self._source_resolver.close()
