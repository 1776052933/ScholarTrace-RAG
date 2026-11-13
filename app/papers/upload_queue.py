"""核心论文下载失败后的人工上传任务管理。"""

from typing import List, Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from .download_tracker import DownloadStatus
from .importance import PaperImportanceClassifier
from .models import DownloadTaskModel, PaperModel, UploadQueueModel
from .repository import PaperRepository, UploadRepository
from .schema import PaperAvailability, PaperImportance, PaperMetadata, UploadRequest


class UploadQueueManager:
    """仅为下载失败的 CORE 论文创建人工上传提醒。"""

    def __init__(
        self,
        session: Session,
        *,
        importance_classifier: Optional[PaperImportanceClassifier] = None,
    ) -> None:
        self.session = session
        self.paper_repository = PaperRepository(session)
        self.upload_repository = UploadRepository(session)
        self.importance_classifier = importance_classifier or PaperImportanceClassifier()

    def create_request(
        self,
        paper_id: str,
        importance: Optional[Union[PaperImportance, str]] = None,
        *,
        reason: Optional[str] = None,
    ) -> Optional[UploadRequest]:
        """满足“下载失败且为 CORE”时创建请求，否则返回 ``None``。"""
        paper = self.paper_repository.get_paper_by_id(paper_id)
        if paper is None:
            raise LookupError(f"paper not found: {paper_id}")

        level = (
            PaperImportance(importance)
            if importance is not None
            else self.importance_classifier.classify_paper(_metadata(paper))
        )
        download_task = self._latest_download_task(paper_id)
        if (
            level is not PaperImportance.CORE
            or download_task is None
            or download_task.status != DownloadStatus.FAILED.value
        ):
            return None

        existing = self._pending_for_paper(paper_id)
        if existing is not None:
            return _request(existing, paper)

        failure_reason = (reason or download_task.error or "核心论文自动下载失败，需要用户上传全文").strip()
        if not failure_reason:
            raise ValueError("reason must not be empty")
        model = self.upload_repository.add_upload(
            paper_id,
            level,
            status="PENDING",
            reason=failure_reason,
        )
        paper.availability = PaperAvailability.USER_UPLOAD_REQUIRED.value
        self.session.flush()
        return _request(model, paper)

    def list_pending(self) -> List[UploadRequest]:
        """返回当前所有需要用户上传的论文列表。"""
        result = []
        for model in self.upload_repository.list_by_status("PENDING"):
            paper = self.paper_repository.get_paper_by_id(model.paper_id)
            if paper is not None:
                result.append(_request(model, paper))
        return result

    def complete_upload(self, request: Union[int, str]) -> bool:
        """按上传任务 ID 或 paper_id 将待处理请求标记为完成。"""
        model = self._find_request(request)
        if model is None or model.status != "PENDING":
            return False
        model.status = "COMPLETED"
        self.session.flush()
        return True

    def _latest_download_task(self, paper_id: str) -> Optional[DownloadTaskModel]:
        statement = (
            select(DownloadTaskModel)
            .where(DownloadTaskModel.paper_id == paper_id)
            .order_by(DownloadTaskModel.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def _pending_for_paper(self, paper_id: str) -> Optional[UploadQueueModel]:
        statement = (
            select(UploadQueueModel)
            .where(
                UploadQueueModel.paper_id == paper_id,
                UploadQueueModel.status == "PENDING",
            )
            .order_by(UploadQueueModel.id.desc())
            .limit(1)
        )
        return self.session.scalar(statement)

    def _find_request(self, request: Union[int, str]) -> Optional[UploadQueueModel]:
        if isinstance(request, bool):
            raise TypeError("request must be an integer ID or string paper_id")
        if isinstance(request, int):
            return self.upload_repository.get_upload_by_id(request)
        if not isinstance(request, str):
            raise TypeError("request must be an integer ID or string paper_id")
        return self._pending_for_paper(request.strip())


def _request(model: UploadQueueModel, paper: PaperModel) -> UploadRequest:
    return UploadRequest(
        paper_id=paper.paper_id,
        title=paper.title,
        reason=model.reason,
        priority=PaperImportance(model.importance),
    )


def _metadata(model: PaperModel) -> PaperMetadata:
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
