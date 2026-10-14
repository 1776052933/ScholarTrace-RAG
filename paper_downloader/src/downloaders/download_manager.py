"""
download_manager.py — 下载管理器

管理并发 PDF 下载队列，提供:
    - ThreadPoolExecutor 并发控制
    - 下载队列管理（FIFO）
    - 指数退避重试
    - JSON 格式下载历史记录
    - 线程安全的状态追踪
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from paper_downloader.src.downloaders.http_downloader import HTTPDownloader
from paper_downloader.src.downloaders.pdf_processor import PDFProcessor


# ── 下载状态 ──────────────────────────────────────────────────────

class DownloadStatus(str, Enum):
    """下载任务状态枚举。"""
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


# ── 下载结果 ──────────────────────────────────────────────────────

@dataclass
class DownloadTask:
    """单个下载任务的数据结构。"""
    url: str
    title: str = ""
    authors: str = ""
    year: str = ""
    doi: str = ""
    arxiv_id: str = ""
    save_dir: str = "./papers"
    filename: Optional[str] = None
    candidate_urls: List[str] = field(default_factory=list)
    download_source: str = ""
    # 运行时状态
    status: DownloadStatus = DownloadStatus.PENDING
    pdf_path: Optional[str] = None
    file_size: int = 0
    error_message: str = ""
    attempt_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


# ── 下载管理器 ────────────────────────────────────────────────────

class DownloadManager:
    """并发下载管理器。

    功能:
        - 线程池并发下载（可配置并发数）
        - 下载队列管理
        - 指数退避重试
        - JSON 格式下载历史持久化
        - 实时进度回调

    Usage::

        manager = DownloadManager(config, max_workers=3)
        manager.add_task("https://arxiv.org/pdf/2401.00001", title="My Paper")
        manager.add_task("https://example.org/paper.pdf", title="Another")
        results = manager.run_all()
        # results: List[DownloadTask]

        # 查看历史
        history = manager.get_history()
    """

    # 历史记录文件默认路径
    _DEFAULT_HISTORY_FILE = "download_history.json"

    # 默认下载目录
    _DEFAULT_SAVE_DIR = "./papers"
    _HISTORY_LOCK = threading.RLock()

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
        history_file: Optional[str] = None,
    ):
        """初始化下载管理器。

        Args:
            config:       配置字典（代理、超时、重试参数）。
            max_workers:  最大并发下载线程数。
            history_file: 下载历史 JSON 文件路径，None 使用默认。
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # 并发控制
        max_workers_cfg = int(self.config.get("concurrency", {}).get("max_downloads", max_workers))
        self._max_workers = max_workers_cfg

        # 重试参数
        retry_cfg = self.config.get("retry", {})
        self._max_retries = int(retry_cfg.get("max_attempts", 3))
        self._backoff_factor = float(retry_cfg.get("backoff_factor", 2))

        # 历史记录
        self._history_file = history_file or self._DEFAULT_HISTORY_FILE

        # 线程安全
        self._lock = threading.RLock()
        self._tasks: OrderedDict[str, DownloadTask] = OrderedDict()
        self._completed_count = 0
        self._failed_count = 0

        # 下载器实例（延迟创建，避免初始化时配置未就绪）
        self._downloader: Optional[HTTPDownloader] = None

        # 进度回调
        self._progress_callback: Optional[Callable[[DownloadTask], None]] = None

        self.logger.info("DownloadManager 初始化完成 (workers=%d, retries=%d)",
                         self._max_workers, self._max_retries)

    # ── 属性 ──────────────────────────────────────────────────────

    @property
    def total_tasks(self) -> int:
        """总任务数。"""
        with self._lock:
            return len(self._tasks)

    @property
    def completed_count(self) -> int:
        """已完成数。"""
        with self._lock:
            return self._completed_count

    @property
    def failed_count(self) -> int:
        """失败数。"""
        with self._lock:
            return self._failed_count

    def set_progress_callback(self, callback: Callable[[DownloadTask], None]) -> None:
        """设置进度回调函数（每完成一个任务时调用）。"""
        self._progress_callback = callback

    # ── 任务管理 ──────────────────────────────────────────────────

    def add_task(
        self,
        url: str,
        title: str = "",
        authors: str = "",
        year: str = "",
        doi: str = "",
        arxiv_id: str = "",
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        candidate_urls: Optional[List[str]] = None,
    ) -> str:
        """添加下载任务到队列。

        Args:
            url:       PDF 下载 URL。
            title:     论文标题。
            authors:   作者。
            year:      年份。
            doi:       DOI。
            arxiv_id:  arXiv ID。
            save_dir:  保存目录，默认使用配置中的路径。
            filename:  自定义文件名（不含 .pdf）。
            candidate_urls: 按优先级排列的备用 PDF 地址。

        Returns:
            任务的唯一 key（用于追踪）。
        """
        # 用 URL + 文件名作为去重 key
        key = f"{url}::{filename or ''}"
        with self._lock:
            if key in self._tasks and self._tasks[key].status in (
                DownloadStatus.COMPLETED, DownloadStatus.RUNNING,
            ):
                self.logger.warning("任务已存在，跳过: %s", key[:80])
                return key

            task = DownloadTask(
                url=url,
                title=title,
                authors=authors,
                year=str(year) if year else "",
                doi=doi,
                arxiv_id=arxiv_id,
                save_dir=save_dir or self.config.get("download", {}).get("path", self._DEFAULT_SAVE_DIR),
                filename=filename,
                candidate_urls=list(candidate_urls or []),
            )
            self._tasks[key] = task

        self.logger.info("添加任务 [%s]: %s", key[:60], title[:60] if title else url[:60])
        return key

    def add_tasks_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """批量添加下载任务。

        Args:
            items: 任务 dict 列表，每个包含 url / title / authors / year / doi / arxiv_id 等字段。

        Returns:
            所有任务的 key 列表。
        """
        keys = []
        for item in items:
            key = self.add_task(
                url=item.get("url", item.get("pdf_url", "")),
                title=item.get("title", ""),
                authors=item.get("authors", ""),
                year=item.get("year", ""),
                doi=item.get("doi", ""),
                arxiv_id=item.get("arxiv_id", ""),
                filename=item.get("filename"),
            )
            keys.append(key)
        return keys

    def get_task(self, key: str) -> Optional[DownloadTask]:
        """按 key 获取任务信息。"""
        with self._lock:
            return self._tasks.get(key)

    def get_all_tasks(self) -> List[DownloadTask]:
        """获取所有任务的副本。"""
        with self._lock:
            return list(self._tasks.values())

    # ── 执行下载 ──────────────────────────────────────────────────

    def run_all(self) -> List[DownloadTask]:
        """执行所有待下载任务。

        使用 ThreadPoolExecutor 并发下载，带指数退避重试。

        Returns:
            所有任务的最终状态列表。
        """
        # 筛选待执行任务
        with self._lock:
            pending = [
                (k, t) for k, t in self._tasks.items()
                if t.status == DownloadStatus.PENDING
            ]

        if not pending:
            self.logger.info("没有待下载的任务")
            return self.get_all_tasks()

        self.logger.info("开始并发下载 %d 个任务 (并发数=%d)", len(pending), self._max_workers)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: Dict[Future, str] = {}
            for key, task in pending:
                future = executor.submit(self._execute_task, key, task)
                futures[future] = key

            for future in futures:
                key = futures[future]
                try:
                    future.result(timeout=self._get_task_timeout())
                except Exception as exc:
                    self.logger.error("任务线程异常 [%s]: %s", key[:60], exc)
                    with self._lock:
                        if key in self._tasks:
                            self._tasks[key].status = DownloadStatus.FAILED
                            self._tasks[key].error_message = str(exc)
                            self._failed_count += 1

        # 保存历史
        self._save_history()

        results = self.get_all_tasks()
        completed = sum(1 for t in results if t.status == DownloadStatus.COMPLETED)
        failed = sum(1 for t in results if t.status == DownloadStatus.FAILED)
        self.logger.info("下载完成: %d 成功, %d 失败", completed, failed)

        return results

    def _execute_task(self, key: str, task: DownloadTask) -> None:
        """执行单个下载任务（含重试逻辑）。

        在 ThreadPoolExecutor 的工作线程中调用。
        """
        with self._lock:
            task.status = DownloadStatus.RUNNING

        last_error = ""
        downloader = self._downloader
        owns_downloader = downloader is None
        if downloader is None:
            downloader = HTTPDownloader(self.config)
        urls = [task.url]
        urls.extend(url for url in task.candidate_urls if url and url not in urls)
        try:
            for attempt in range(1, self._max_retries + 1):
                with self._lock:
                    task.attempt_count = attempt

                self.logger.info("[%s] 第 %d/%d 次尝试下载",
                                 task.title[:40] or key[:40], attempt, self._max_retries)

                for candidate_index, candidate_url in enumerate(urls, 1):
                    try:
                        self.logger.info(
                            "[%s] 尝试候选源 %d/%d: %s",
                            task.title[:40], candidate_index, len(urls), candidate_url[:100],
                        )
                        pdf_path = downloader.download(
                            url=candidate_url,
                            save_path=task.save_dir,
                            filename=task.filename,
                            resume=False,
                        )

                        if pdf_path and pdf_path.exists():
                            # 校验 PDF
                            if PDFProcessor.check_corrupted(pdf_path):
                                last_error = "PDF 文件已损坏"
                                self.logger.warning("[%s] PDF 损坏，尝试下一来源...", task.title[:40])
                                try:
                                    pdf_path.unlink()
                                except OSError:
                                    pass
                                continue

                            with self._lock:
                                task.status = DownloadStatus.COMPLETED
                                task.url = candidate_url
                                task.download_source = self._source_from_url(candidate_url)
                                task.pdf_path = str(pdf_path)
                                task.file_size = pdf_path.stat().st_size
                                task.completed_at = datetime.now().isoformat()
                                self._completed_count += 1
                            self.logger.info("[%s] 下载成功: %s", task.title[:40], pdf_path)
                            if self._progress_callback:
                                self._progress_callback(task)
                            return

                        last_error = f"候选源未返回有效 PDF: {candidate_url}"
                    except Exception as exc:
                        last_error = str(exc)
                        self.logger.error(
                            "[%s] 候选源异常 (attempt %d): %s",
                            task.title[:40], attempt, exc,
                        )

                # 指数退避：backoff^attempt 秒
                if attempt < self._max_retries:
                    sleep_sec = self._backoff_factor ** attempt
                    self.logger.debug("等待 %.1f 秒后重试...", sleep_sec)
                    time.sleep(sleep_sec)
        finally:
            if owns_downloader:
                downloader.close()

        # 所有重试均失败
        with self._lock:
            task.status = DownloadStatus.FAILED
            task.error_message = last_error
            self._failed_count += 1
        self.logger.error("[%s] 下载失败: %s", task.title[:40], last_error)
        if self._progress_callback:
            self._progress_callback(task)

    @staticmethod
    def _source_from_url(url: str) -> str:
        """从最终 URL 标记实际下载源。"""
        lowered = url.lower()
        if "arxiv.org" in lowered:
            return "arxiv"
        if "europepmc.org" in lowered:
            return "europe_pmc"
        return "http"

    # ── 下载历史 ──────────────────────────────────────────────────

    def _save_history(self) -> None:
        """将当前任务状态保存为 JSON 历史记录。"""
        with self._lock:
            records = []
            for task in self._tasks.values():
                records.append({
                    "url": task.url,
                    "title": task.title,
                    "authors": task.authors,
                    "year": task.year,
                    "doi": task.doi,
                    "arxiv_id": task.arxiv_id,
                    "status": task.status.value,
                    "pdf_path": task.pdf_path,
                    "file_size": task.file_size,
                    "error_message": task.error_message,
                    "attempt_count": task.attempt_count,
                    "download_source": task.download_source,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                })

        try:
            with self._HISTORY_LOCK:
                history_path = Path(self._history_file)
                history_path.parent.mkdir(parents=True, exist_ok=True)
                existing: List[Dict[str, Any]] = []
                if history_path.exists():
                    try:
                        data = json.loads(history_path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            existing = [item for item in data if isinstance(item, dict)]
                    except (json.JSONDecodeError, OSError):
                        existing = []
                merged = OrderedDict()
                for record in [*existing, *records]:
                    identity = (
                        record.get("doi")
                        or record.get("arxiv_id")
                        or record.get("url")
                        or record.get("title")
                    )
                    merged[str(identity)] = record
                tmp_path = history_path.with_suffix(".tmp")
                tmp_path.write_text(
                    json.dumps(list(merged.values()), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_path.replace(history_path)
                self.logger.debug("历史记录已保存: %s (%d 条)", history_path, len(merged))
        except OSError as exc:
            self.logger.error("保存历史记录失败: %s", exc)

    def get_history(self) -> List[Dict[str, Any]]:
        """读取下载历史记录。

        Returns:
            历史任务记录列表。
        """
        history_path = Path(self._history_file)
        if not history_path.exists():
            return []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            self.logger.warning("读取历史记录失败: %s", exc)
            return []

    def clear_history(self) -> None:
        """清空下载历史。"""
        try:
            Path(self._history_file).unlink(missing_ok=True)
            self.logger.info("历史记录已清空")
        except OSError as exc:
            self.logger.error("清空历史记录失败: %s", exc)

    # ── 统计 ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取下载统计信息。

        Returns:
            包含 total / completed / failed / pending / total_bytes 的 dict。
        """
        with self._lock:
            total = len(self._tasks)
            completed = sum(1 for t in self._tasks.values() if t.status == DownloadStatus.COMPLETED)
            failed = sum(1 for t in self._tasks.values() if t.status == DownloadStatus.FAILED)
            pending = sum(1 for t in self._tasks.values() if t.status == DownloadStatus.PENDING)
            total_bytes = sum(t.file_size for t in self._tasks.values() if t.file_size > 0)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2),
        }

    # ── 工具 ──────────────────────────────────────────────────────

    def _get_task_timeout(self) -> int:
        """从配置获取单个任务的超时（下载超时 + 重试缓冲）。"""
        base = int(self.config.get("timeout", {}).get("download", 120))
        # 加上重试等待时间
        retry_buffer = sum(self._backoff_factor ** i for i in range(1, self._max_retries))
        return base + int(retry_buffer) + 30  # 额外 30s 缓冲

    def close(self) -> None:
        """关闭管理器，释放资源。"""
        self._save_history()
        if self._downloader:
            self._downloader.close()
        self.logger.info("DownloadManager 已关闭")

    def __enter__(self) -> "DownloadManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
