"""将 paper_downloader 的纯字典接口接入论文资源数据库。"""

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .download_tracker import DownloadStatus, DownloadTracker
from .models import DownloadTaskModel, PaperFileModel, PaperModel
from .paper_manager import PaperManager
from .repository import FileRepository, PaperRepository
from .schema import PaperAvailability
from paper_downloader.src.utils.pdf_identity import pdf_matches_title


DownloadFunction = Callable[..., Mapping[str, Any]]


class PaperDownloadCoordinator:
    """执行 PDF 下载并同步文件、任务、可用性及人工上传状态。

    该类只依赖 ``paper_downloader.download_pdf`` 的稳定字典协议，不依赖下载
    模块内部的搜索器或 Downloader 实现，因此后续升级下载模块时边界保持稳定。
    """

    def __init__(
        self,
        session: Session,
        *,
        download_function: Optional[DownloadFunction] = None,
        output_dir: str = "./papers",
        engine: str = "academic_oa",
        timeout: int = 30,
        max_workers: int = 3,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        output = Path(output_dir).expanduser()
        if not str(output).strip():
            raise ValueError("output_dir must not be empty")
        self.session = session
        self._metadata_input = download_function is None
        self.download_function = download_function or _default_download_function()
        self.output_dir = output
        self.engine = engine.strip() or "academic_oa"
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self.paper_repository = PaperRepository(session)
        self.file_repository = FileRepository(session)
        self.download_tracker = DownloadTracker(session)

    def download(
        self,
        paper_id: str,
        *,
        retry_failed: bool = False,
    ) -> Dict[str, Any]:
        """下载单篇论文，并将结果写入当前数据库事务。"""
        paper_id = _required_text(paper_id, "paper_id")
        results = self.download_many([paper_id], retry_failed=retry_failed)
        if not results:
            raise RuntimeError(f"download produced no result: {paper_id}")
        return results[0]

    def download_many(
        self,
        paper_ids: Iterable[str],
        *,
        retry_failed: bool = False,
    ) -> List[Dict[str, Any]]:
        """并发执行网络下载，按输入顺序串行回写数据库状态。"""
        ordered_ids = [
            _required_text(paper_id, "paper_id")
            for paper_id in dict.fromkeys(paper_ids)
        ]
        if not ordered_ids:
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        immediate: Dict[str, Dict[str, Any]] = {}
        pending: Dict[str, tuple[PaperModel, DownloadTaskModel, Dict[str, Any]]] = {}

        for paper_id in ordered_ids:
            paper = self.paper_repository.get_paper_by_id(paper_id)
            if paper is None:
                raise LookupError(f"paper not found: {paper_id}")
            existing_path = self._available_pdf(paper_id)
            if existing_path is not None:
                immediate[paper_id] = self._result(
                    paper,
                    success=True,
                    file_path=existing_path,
                    error=None,
                    engine_used=None,
                    skipped=True,
                )
                continue

            task = self._latest_task(paper_id)
            if task is None or task.status == DownloadStatus.SUCCESS.value:
                task = self.download_tracker.create_task(paper_id)
            elif task.status == DownloadStatus.FAILED.value:
                if not retry_failed:
                    immediate[paper_id] = self._result(
                        paper,
                        success=False,
                        file_path=None,
                        error=task.error or "previous download failed",
                        engine_used=None,
                        skipped=True,
                    )
                    continue
                task = self.download_tracker.retry_task(task.id)

            self.download_tracker.update_status(task.id, DownloadStatus.DOWNLOADING)
            paper.availability = PaperAvailability.DOWNLOADING.value
            pending[paper_id] = (paper, task, _download_payload(paper))
        self.session.flush()

        network_results: Dict[str, Dict[str, Any]] = {}
        if pending:
            workers = min(self.max_workers, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._invoke_download, payload): paper_id
                    for paper_id, (_, _, payload) in pending.items()
                }
                for future in as_completed(futures):
                    paper_id = futures[future]
                    try:
                        network_results[paper_id] = dict(future.result())
                    except Exception as exc:
                        network_results[paper_id] = {
                            "success": False,
                            "file_path": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "engine_used": None,
                        }

        results: List[Dict[str, Any]] = []
        for paper_id in ordered_ids:
            if paper_id in immediate:
                results.append(immediate[paper_id])
                continue
            paper, task, _ = pending[paper_id]
            result = network_results[paper_id]
            path = _successful_pdf_path(result)
            if path is None:
                error = str(result.get("error") or "paper downloader returned no PDF")
                results.append(self._record_failure(
                    paper,
                    task,
                    error,
                    result.get("engine_used"),
                ))
                continue

            if pdf_matches_title(path, paper.title) is False:
                self._invalidate_derived_files(paper.paper_id)
                results.append(self._record_failure(
                    paper,
                    task,
                    "downloaded PDF title does not match the requested paper",
                    result.get("engine_used"),
                ))
                continue

            conflicting_paper = self._conflicting_path_owner(paper_id, path)
            if conflicting_paper is not None:
                self._invalidate_derived_files(paper.paper_id)
                results.append(self._record_failure(
                    paper,
                    task,
                    f"downloaded PDF is already bound to a different paper: {conflicting_paper}",
                    result.get("engine_used"),
                ))
                continue

            self._record_pdf(paper_id, path)
            self.download_tracker.update_status(task.id, DownloadStatus.SUCCESS)
            paper.availability = PaperAvailability.METADATA_ONLY.value
            self.session.flush()
            results.append(self._result(
                paper,
                success=True,
                file_path=path,
                error=None,
                engine_used=result.get("engine_used"),
                skipped=False,
            ))
        return results

    def _invoke_download(self, payload: Dict[str, Any]) -> Mapping[str, Any]:
        value: Any = payload if self._metadata_input else payload["title"]
        return self.download_function(
            value,
            output_dir=str(self.output_dir),
            engine=self.engine,
            timeout=self.timeout,
            rename=True,
        )

    def _record_failure(
        self,
        paper: PaperModel,
        task: DownloadTaskModel,
        error: str,
        engine_used: Any,
    ) -> Dict[str, Any]:
        self.download_tracker.fail_task(task.id, error)
        paper.availability = PaperAvailability.DOWNLOAD_FAILED.value
        # 复用 PaperManager 的重要性判断：CORE 失败会进入人工上传队列。
        state = PaperManager(self.session).prepare_for_rag(paper.paper_id)
        self.session.flush()
        result = self._result(
            paper,
            success=False,
            file_path=None,
            error=error,
            engine_used=engine_used,
            skipped=False,
        )
        result["need_user_upload"] = bool(state["need_user_upload"])
        return result

    def _record_pdf(self, paper_id: str, path: str) -> None:
        normalized = str(Path(path).expanduser().resolve())
        for record in self.file_repository.list_by_paper(paper_id):
            if (
                record.file_type.casefold() == "pdf"
                and str(Path(record.file_path).expanduser().resolve()) == normalized
            ):
                record.status = "AVAILABLE"
                self.session.flush()
                return
        self.file_repository.add_file(
            paper_id,
            normalized,
            status="AVAILABLE",
            file_type="pdf",
        )

    def _conflicting_path_owner(self, paper_id: str, path: str) -> Optional[str]:
        normalized = str(Path(path).expanduser().resolve())
        records = list(self.session.scalars(
            select(PaperFileModel).where(
                PaperFileModel.file_path == normalized,
                PaperFileModel.file_type == "pdf",
            )
        ))
        for record in records:
            if record.paper_id == paper_id or record.status.casefold() == "mismatch":
                continue
            return record.paper_id
        return None

    def _available_pdf(self, paper_id: str) -> Optional[str]:
        paper = self.paper_repository.get_paper_by_id(paper_id)
        for record in self.file_repository.list_by_paper(paper_id):
            if (
                record.file_type.casefold() == "pdf"
                and record.status.casefold() in {"available", "completed", "downloaded", "success"}
                and Path(record.file_path).is_file()
            ):
                path = str(Path(record.file_path).resolve())
                if paper is not None and pdf_matches_title(path, paper.title) is False:
                    record.status = "MISMATCH"
                    paper.embedding_status = "PENDING"
                    self._invalidate_derived_files(paper_id)
                    self.session.flush()
                    continue
                conflicting_paper = self._conflicting_path_owner(paper_id, path)
                if conflicting_paper is not None:
                    record.status = "MISMATCH"
                    if paper is not None:
                        paper.embedding_status = "PENDING"
                    self._invalidate_derived_files(paper_id)
                    self.session.flush()
                    continue
                return path
        return None

    def _invalidate_derived_files(self, paper_id: str) -> None:
        paper = self.paper_repository.get_paper_by_id(paper_id)
        if paper is not None:
            paper.embedding_status = "PENDING"
        for record in self.file_repository.list_by_paper(paper_id):
            if record.file_type.casefold() in {"text", "txt", "parsed_text"}:
                record.status = "MISMATCH"
        self.session.flush()

    def _latest_task(self, paper_id: str) -> Optional[DownloadTaskModel]:
        statement = (
            select(DownloadTaskModel)
            .where(DownloadTaskModel.paper_id == paper_id)
            .order_by(DownloadTaskModel.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def _result(
        self,
        paper: PaperModel,
        *,
        success: bool,
        file_path: Optional[str],
        error: Optional[str],
        engine_used: Any,
        skipped: bool,
    ) -> Dict[str, Any]:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "success": success,
            "file_path": file_path,
            "error": error,
            "engine_used": engine_used,
            "skipped": skipped,
            "need_user_upload": False,
        }


def _default_download_function() -> DownloadFunction:
    try:
        from paper_downloader import download_pdf
    except ImportError as exc:
        raise ImportError(
            "PaperDownloadCoordinator requires the paper_downloader module"
        ) from exc
    return download_pdf


def _successful_pdf_path(result: Mapping[str, Any]) -> Optional[str]:
    if not bool(result.get("success")):
        return None
    raw_path = result.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        return None
    return str(path.resolve())


def _download_payload(paper: PaperModel) -> Dict[str, Any]:
    """保留排序阶段已有的精确标识，避免下载器重新盲搜标题。"""
    paper_id = str(paper.paper_id or "").strip()
    url = paper_id if paper_id.startswith(("http://", "https://")) else ""
    arxiv_id = paper.arxiv_id or _extract_arxiv_id(url)
    doi = paper.doi or _extract_doi(paper_id)
    pdf_url = url if _looks_like_pdf_url(url) else ""
    return {
        "paper_id": paper_id,
        "title": paper.title,
        "authors": list(paper.authors or []),
        "abstract": paper.abstract,
        "year": paper.year,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": url,
        "pdf_url": pdf_url,
        "source": _infer_source(url, arxiv_id, doi),
    }


def _extract_arxiv_id(value: str) -> Optional[str]:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None


def _extract_doi(value: str) -> Optional[str]:
    match = re.search(r"(?:doi\.org/|^)(10\.\d{4,9}/[^?#\s]+)", value, re.I)
    return match.group(1).rstrip(".,;)") if match else None


def _looks_like_pdf_url(value: str) -> bool:
    lowered = value.casefold()
    return bool(value) and (
        lowered.endswith(".pdf")
        or "/pdf/" in lowered
        or "/article/download/" in lowered
        or "pdf=render" in lowered
        or "type=pdf" in lowered
    )


def _infer_source(url: str, arxiv_id: Optional[str], doi: Optional[str]) -> str:
    if arxiv_id or "arxiv.org" in url.casefold():
        return "arxiv"
    if _looks_like_pdf_url(url):
        return "direct"
    if doi:
        return "doi"
    return ""


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result
