"""
async_downloader.py — 异步下载器

基于 asyncio + aiohttp 的异步论文下载实现，
继承 PaperDownloader，与同步版本共享相同的配置和接口。
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.core.callback_manager import CallbackManager, CallbackEvent
from paper_downloader.src.core.progress_tracker import ProgressTracker
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import (
    PaperNotFoundError,
    DownloadError,
    ValidationError,
)

# aiohttp 可选导入
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False


class AsyncPaperDownloader(PaperDownloader):
    """异步论文下载器。

    使用 asyncio + aiohttp 实现并发搜索与下载，
    API 兼容 PaperDownloader，同步和异步版本可无缝切换。

    Usage::

        # 异步方式
        import asyncio
        from paper_downloader.src.core.async_downloader import AsyncPaperDownloader

        async def main():
            async with AsyncPaperDownloader() as dl:
                papers = await dl.batch_download_async([
                    "Attention Is All You Need",
                    "BERT: Pre-training of Deep Bidirectional Transformers",
                ])

        asyncio.run(main())
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        max_concurrent_searches: int = 3,
        max_concurrent_downloads: int = 5,
        **kwargs: Any,
    ):
        """初始化异步下载器。

        Args:
            config_path:            配置文件路径。
            config:                 配置字典。
            max_concurrent_searches: 最大并发搜索数。
            max_concurrent_downloads: 最大并发下载数。
            **kwargs:               PaperDownloader 的其他参数。
        """
        super().__init__(config_path=config_path, config=config, **kwargs)

        # 并发控制
        self._search_semaphore = asyncio.Semaphore(max_concurrent_searches)
        self._download_semaphore = asyncio.Semaphore(max_concurrent_downloads)

        # 回调管理器
        self._callbacks = CallbackManager()

        # aiohttp session（延迟创建）
        self._aio_session: Optional[Any] = None
        self._session_owner = False

        if not AIOHTTP_AVAILABLE:
            self.logger.warning("aiohttp 未安装，异步功能不可用。请运行: pip install aiohttp")

    # ── Session 管理 ──────────────────────────────────────────────

    async def _get_session(self) -> Any:
        """获取或创建 aiohttp ClientSession。"""
        if self._aio_session is None or self._aio_session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self._config.get("timeout", {}).get("download", 120),
                connect=self._config.get("timeout", {}).get("connection", 10),
            )
            self._aio_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "AutoPaper/0.1"},
            )
            self._session_owner = True
        return self._aio_session

    async def _close_session(self) -> None:
        """关闭 aiohttp session。"""
        if self._session_owner and self._aio_session and not self._aio_session.closed:
            await self._aio_session.close()

    # ── 异步搜索 ──────────────────────────────────────────────────

    async def search_async(
        self,
        title: str,
        max_results: int = 5,
        engines: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Paper]:
        """异步搜索论文。

        Args:
            title:       论文标题。
            max_results: 最大返回数。
            engines:     搜索引擎列表。
            **kwargs:    额外搜索参数。

        Returns:
            Paper 对象列表。
        """
        self.logger.info("异步搜索: '%s'", title)

        self._callbacks.trigger(CallbackEvent.ON_SEARCH_START, title)

        # 在线程池中运行同步搜索（SearchFactory 不是异步的）
        async with self._search_semaphore:
            loop = asyncio.get_running_loop()
            try:
                papers = await loop.run_in_executor(
                    None,
                    lambda: self.search(title, max_results=max_results, engines=engines, **kwargs),
                )
                self._callbacks.trigger(CallbackEvent.ON_SEARCH_COMPLETE, papers)
                return papers
            except Exception as exc:
                self._callbacks.trigger(CallbackEvent.ON_SEARCH_ERROR, title, exc)
                raise

    # ── 异步下载 ──────────────────────────────────────────────────

    async def download_async(
        self,
        papers: List[Paper],
        output_dir: Optional[str] = None,
        rename: bool = True,
    ) -> List[Paper]:
        """使用 aiohttp 异步下载多篇论文。

        Args:
            papers:     Paper 对象列表。
            output_dir: 输出目录。
            rename:     是否重命名。

        Returns:
            更新后的 Paper 列表。
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp 未安装，无法使用异步下载。pip install aiohttp")

        if not papers:
            raise ValidationError("没有可下载的论文")

        output_dir = output_dir or self._config.get("download", {}).get("path", "./papers")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("异步下载 %d 篇论文 → %s", len(papers), output_dir)

        session = await self._get_session()
        sem = self._download_semaphore

        async def _download_one(paper: Paper) -> Paper:
            async with sem:
                self._callbacks.trigger(CallbackEvent.ON_DOWNLOAD_START, paper)
                try:
                    url = paper.pdf_url or paper.url or paper.doi
                    if not url:
                        return paper
                    if not url.startswith("http") and paper.doi:
                        url = f"https://doi.org/{paper.doi}"

                    filename = self._make_filename(paper) if rename else None
                    filepath = Path(output_dir) / (filename or self._url_to_filename(url))

                    # 获取文件大小检查断点续传
                    headers = {}
                    if filepath.exists():
                        existing_size = filepath.stat().st_size
                        if existing_size > 0:
                            headers["Range"] = f"bytes={existing_size}-"

                    retry_count = self._config.get("retry", {}).get("max_attempts", 3)
                    backoff = self._config.get("retry", {}).get("backoff_factor", 2)

                    for attempt in range(1, retry_count + 1):
                        try:
                            async with session.get(url, headers=headers) as resp:
                                resp.raise_for_status()
                                mode = "ab" if headers else "wb"
                                with open(filepath, mode) as f:
                                    async for chunk in resp.content.iter_chunked(8192):
                                        f.write(chunk)

                            if filepath.exists() and filepath.stat().st_size > 0:
                                paper.pdf_path = str(filepath)
                                paper.file_size = filepath.stat().st_size
                                paper.downloaded_at = datetime.now().isoformat()
                                self._callbacks.trigger(
                                    CallbackEvent.ON_DOWNLOAD_COMPLETE, paper,
                                )
                                return paper

                        except aiohttp.ClientError as exc:
                            if attempt == retry_count:
                                raise
                            await asyncio.sleep(backoff ** attempt)
                        except Exception as exc:
                            if attempt == retry_count:
                                raise
                            await asyncio.sleep(backoff ** attempt)

                    return paper

                except Exception as exc:
                    self.logger.error("异步下载失败 [%s]: %s", paper.title, exc)
                    self._callbacks.trigger(CallbackEvent.ON_DOWNLOAD_ERROR, paper, exc)
                    return paper

        # 并发下载所有论文
        tasks = [_download_one(p) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        final: List[Paper] = []
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final.append(papers[i])
                failed += 1
            else:
                final.append(result)

        if failed == len(papers) and len(papers) > 0:
            raise DownloadError("所有异步下载均失败")

        return final

    def _make_filename(self, paper: Paper) -> Optional[str]:
        """生成下载文件名（含非法字符清理）。"""
        template = self._config.get("download", {}).get(
            "filename_template", "{first_author}_{year}_{title}"
        )
        raw = (
            template.replace("{first_author}", paper.first_author_surname)
            .replace("{year}", paper.year or "nodate")
            .replace("{title}", paper.title[:80])
        ) + ".pdf"
        # 清理 Windows 非法字符: ? * : " < > | / \ 等
        return self._sanitize_filename(raw)

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """从 URL 提取文件名。"""
        from urllib.parse import unquote, urlparse
        path = urlparse(url).path
        name = unquote(path.rsplit("/", 1)[-1]) if "/" in path else "paper.pdf"
        return name if name.endswith(".pdf") else name + ".pdf"

    # ── 异步批量下载 ──────────────────────────────────────────────

    async def batch_download_async(
        self,
        titles: List[str],
        output_dir: Optional[str] = None,
        max_results: int = 3,
        engines: Optional[List[str]] = None,
        rename: bool = True,
        **kwargs: Any,
    ) -> List[Paper]:
        """异步批量下载多篇论文。

        并发执行: 搜索全部标题 → 统一下载所有结果。

        Args:
            titles:      论文标题列表。
            output_dir:  输出目录。
            max_results: 搜索候选数。
            engines:     搜索引擎。
            rename:      是否重命名。
            **kwargs:    额外参数。

        Returns:
            Paper 对象列表。
        """
        total = len(titles)
        self.logger.info("异步批量下载 %d 篇论文", total)

        tracker = ProgressTracker(total=total, description="Async Batch Download")
        tracker.start()

        self._callbacks.trigger(CallbackEvent.ON_BATCH_START, total)

        # 第一阶段：并发搜索所有标题
        self.logger.info("阶段 1/2: 并发搜索 %d 个标题", total)
        search_tasks = [
            self._search_with_fallback(title, max_results, engines, **kwargs)
            for title in titles
        ]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 收集可下载的论文
        all_papers: List[Paper] = []
        failed_titles: Set[int] = set()
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                all_papers.append(Paper(title=titles[i]))
                failed_titles.add(i)
                self.logger.warning("[%d/%d] 搜索失败: %s — %s", i + 1, total, titles[i], result)
            elif isinstance(result, list) and result:
                best = result[0]
                all_papers.append(best)
                self.logger.info("[%d/%d] 搜索完成: %s", i + 1, total, best.title[:60])
            else:
                all_papers.append(Paper(title=titles[i]))
                failed_titles.add(i)

        # 第二阶段：并发下载
        downloadable = [p for i, p in enumerate(all_papers) if i not in failed_titles and (p.pdf_url or p.arxiv_id or p.doi)]
        self.logger.info("阶段 2/2: 并发下载 %d 篇", len(downloadable))

        if downloadable:
            try:
                downloaded = await self.download_async(downloadable, output_dir=output_dir, rename=rename)
                # 回填结果
                dl_map = {p.title: p for p in downloaded}
                for i, paper in enumerate(all_papers):
                    if paper.title in dl_map and dl_map[paper.title].pdf_path:
                        updated = dl_map[paper.title]
                        all_papers[i] = updated
                        tracker.update(message=updated.title, paper=updated)
                        self._callbacks.trigger(
                            CallbackEvent.ON_BATCH_PROGRESS,
                            i + 1, total, updated,
                        )
                    elif i not in failed_titles:
                        tracker.update(success=False, message=f"下载失败: {paper.title}")
            except DownloadError as exc:
                self._callbacks.trigger(CallbackEvent.ON_BATCH_ERROR, exc)

        # 标记搜索失败的
        for i in failed_titles:
            tracker.update(success=False, message=f"搜索失败: {titles[i]}")

        summary = tracker.finish()
        self._callbacks.trigger(CallbackEvent.ON_BATCH_COMPLETE, all_papers)
        self._callbacks.trigger(CallbackEvent.ON_ALL_COMPLETE, summary)

        return all_papers

    async def _search_with_fallback(
        self,
        title: str,
        max_results: int,
        engines: Optional[List[str]],
        **kwargs: Any,
    ) -> List[Paper]:
        """异步搜索，失败时返回 PaperNotFoundError 对应的空列表。"""
        try:
            return await self.search_async(title, max_results=max_results, engines=engines, **kwargs)
        except PaperNotFoundError:
            return []
        except Exception:
            return []

    # ── 回调 ──────────────────────────────────────────────────────

    @property
    def callbacks(self) -> CallbackManager:
        """返回回调管理器（用于注册事件监听）。"""
        return self._callbacks

    # ── 资源管理 ──────────────────────────────────────────────────

    async def __aenter__(self) -> "AsyncPaperDownloader":
        await self._get_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._close_session()
        await asyncio.sleep(0)  # 让 pending 任务有机会清理

    def __exit__(self, *args: Any) -> None:
        """同步退出也需清理异步资源。"""
        super().__exit__(*args)
        if self._aio_session and not self._aio_session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_session())
                else:
                    loop.run_until_complete(self._close_session())
            except Exception:
                pass
