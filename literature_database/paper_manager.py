"""论文注册、全文准备和 RAG 可用性编排。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .availability import AvailabilityChecker
from .download_tracker import DownloadStatus, DownloadTracker
from .importance import PaperImportanceClassifier
from .models import DownloadTaskModel, PaperFileModel, PaperModel
from .repository import PaperRepository
from .schema import PaperAvailability, PaperImportance, PaperMetadata
from .upload_queue import UploadQueueManager


class PaperManager:
    """协调论文仓储、资源检查、下载任务和人工上传队列。"""

    def __init__(
        self,
        session: Session,
        *,
        availability_checker: Optional[AvailabilityChecker] = None,
        download_tracker: Optional[DownloadTracker] = None,
        importance_classifier: Optional[PaperImportanceClassifier] = None,
        upload_queue: Optional[UploadQueueManager] = None,
    ) -> None:
        self.session = session
        self.paper_repository = PaperRepository(session)
        self.availability_checker = availability_checker or AvailabilityChecker(session)
        self.download_tracker = download_tracker or DownloadTracker(session)
        self.importance_classifier = importance_classifier or PaperImportanceClassifier()
        self.upload_queue = upload_queue or UploadQueueManager(
            session,
            importance_classifier=self.importance_classifier,
        )

    def register_paper(
        self,
        paper: Any,
        *,
        rank_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """注册 Ranking Agent 论文并确保全文获取任务已经准备。"""
        metadata = _metadata(paper, rank_score=rank_score)
        model = self._find_existing(metadata)
        if model is None:
            model = self.paper_repository.add_paper(metadata)
        else:
            _merge_metadata(model, metadata)
            self.session.flush()

        availability = self.availability_checker.check(model.paper_id)
        if availability is PaperAvailability.FULL_AVAILABLE:
            model.availability = PaperAvailability.FULL_AVAILABLE.value
            self.session.flush()
        elif self._downloaded_pdf_available(model.paper_id):
            # PDF 下载已完成，等待 RAG Reader 解析和向量化，不创建重复下载任务。
            model.availability = PaperAvailability.METADATA_ONLY.value
            self.session.flush()
        else:
            task = self._latest_download_task(model.paper_id)
            if task is None or task.status == DownloadStatus.SUCCESS.value:
                self.download_tracker.create_task(model.paper_id)
                model.availability = PaperAvailability.DOWNLOADING.value
                self.session.flush()
            elif task.status in {DownloadStatus.PENDING.value, DownloadStatus.DOWNLOADING.value}:
                model.availability = PaperAvailability.DOWNLOADING.value
                self.session.flush()

        return self.prepare_for_rag(model.paper_id)

    def prepare_for_rag(self, paper_id: str) -> Dict[str, Any]:
        """返回论文进入 RAG 前的资源准备状态。"""
        paper = self.paper_repository.get_paper_by_id(paper_id)
        if paper is None:
            raise LookupError(f"paper not found: {paper_id}")

        status = self.availability_checker.check(paper_id)
        need_download = False
        need_user_upload = False

        if status is PaperAvailability.FULL_AVAILABLE:
            paper.availability = status.value
        elif self._downloaded_pdf_available(paper_id):
            # PDF 已可供 RAG Reader 摄取，文本和 embedding 尚未完成。
            status = PaperAvailability.METADATA_ONLY
            need_download = False
            paper.availability = status.value
        else:
            task = self._latest_download_task(paper_id)
            if task is None:
                need_download = True
                status = PaperAvailability.METADATA_ONLY
            elif task.status in {DownloadStatus.PENDING.value, DownloadStatus.DOWNLOADING.value}:
                need_download = True
                status = PaperAvailability.DOWNLOADING
                paper.availability = status.value
            elif task.status == DownloadStatus.FAILED.value:
                paper.availability = PaperAvailability.DOWNLOAD_FAILED.value
                importance = self.importance_classifier.classify_paper(_model_metadata(paper))
                if importance is PaperImportance.CORE:
                    request = self.upload_queue.create_request(
                        paper_id,
                        importance,
                        reason=task.error,
                    )
                    need_user_upload = request is not None
                    need_download = False
                    status = PaperAvailability.USER_UPLOAD_REQUIRED
                else:
                    need_download = True
                    status = PaperAvailability.DOWNLOAD_FAILED
            elif task.status == DownloadStatus.SUCCESS.value:
                # 下载已完成但解析文本或 embedding 仍未就绪，不重复下载。
                status = PaperAvailability.METADATA_ONLY

        if status is PaperAvailability.USER_UPLOAD_REQUIRED:
            need_user_upload = True
            need_download = False
        self.session.flush()
        return {
            "paper_id": paper_id,
            "status": status.value,
            "need_download": need_download,
            "need_user_upload": need_user_upload,
        }

    def _find_existing(self, paper: PaperMetadata) -> Optional[PaperModel]:
        existing = self.paper_repository.get_paper_by_id(paper.paper_id)
        if existing is None and paper.doi:
            existing = self.paper_repository.find_by_doi(paper.doi)
        if existing is None:
            existing = self.paper_repository.find_by_title(paper.title)
        return existing

    def _latest_download_task(self, paper_id: str) -> Optional[DownloadTaskModel]:
        statement = (
            select(DownloadTaskModel)
            .where(DownloadTaskModel.paper_id == paper_id)
            .order_by(DownloadTaskModel.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def _downloaded_pdf_available(self, paper_id: str) -> bool:
        statement = select(PaperFileModel).where(
            PaperFileModel.paper_id == paper_id,
            PaperFileModel.file_type == "pdf",
        )
        return any(
            item.status.casefold() in {"available", "completed", "downloaded", "success"}
            and Path(item.file_path).is_file()
            for item in self.session.scalars(statement)
        )


def _metadata(value: Any, *, rank_score: Optional[float]) -> PaperMetadata:
    if isinstance(value, PaperMetadata):
        if rank_score is None:
            return value
        return value.model_copy(update={"rank_score": rank_score})
    data = _mapping(value)
    title = str(data.get("title") or "").strip()
    score = rank_score
    if score is None:
        score_data = data.get("score")
        if isinstance(score_data, Mapping):
            score = score_data.get("final_score")
        score = data.get("rank_score", score)
    paper_id = data.get("paper_id") or data.get("doi") or data.get("url") or _title_id(title)
    return PaperMetadata(
        paper_id=str(paper_id),
        title=title,
        authors=[str(author) for author in data.get("authors") or []],
        abstract=str(data.get("abstract") or ""),
        doi=data.get("doi"),
        arxiv_id=data.get("arxiv_id") or _extract_arxiv_id(str(data.get("url") or paper_id)),
        year=data.get("year"),
        venue=str(data.get("venue") or ""),
        citation_count=int(data.get("citation_count") or 0),
        rank_score=float(score or 0.0),
    )


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    raise TypeError("paper must be a mapping or Pydantic model")


def _model_metadata(model: PaperModel) -> PaperMetadata:
    return PaperMetadata(
        paper_id=model.paper_id,
        title=model.title,
        authors=model.authors,
        abstract=model.abstract,
        doi=model.doi,
        arxiv_id=model.arxiv_id,
        year=model.year,
        venue=model.venue,
        citation_count=model.citation_count,
        rank_score=model.rank_score,
    )


def _merge_metadata(model: PaperModel, metadata: PaperMetadata) -> None:
    """用后续来源补齐可下载标识，同时保留已持久化主键。"""
    for name in ("doi", "arxiv_id", "year"):
        if getattr(model, name) in (None, "") and getattr(metadata, name) not in (None, ""):
            setattr(model, name, getattr(metadata, name))
    for name in ("abstract", "venue"):
        if not getattr(model, name) and getattr(metadata, name):
            setattr(model, name, getattr(metadata, name))
    if metadata.authors and not model.authors:
        model.authors = list(metadata.authors)
    model.citation_count = max(model.citation_count or 0, metadata.citation_count or 0)
    model.rank_score = max(model.rank_score or 0.0, metadata.rank_score or 0.0)


def _title_id(title: str) -> str:
    normalized = "".join(character.casefold() for character in title if character.isalnum())
    return f"title:{normalized}"


def _extract_arxiv_id(value: str) -> Optional[str]:
    import re

    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None
