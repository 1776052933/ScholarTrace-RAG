"""检查论文是否具备进入 RAG 的完整本地资源。"""

from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from .models import PaperFileModel
from .repository import FileRepository, PaperRepository
from .schema import PaperAvailability


class AvailabilityChecker:
    """检查论文元数据、PDF、解析文本和 embedding 是否完整。"""

    def __init__(
        self,
        session: Session,
        *,
        paper_repository: Optional[PaperRepository] = None,
        file_repository: Optional[FileRepository] = None,
    ) -> None:
        self.paper_repository = paper_repository or PaperRepository(session)
        self.file_repository = file_repository or FileRepository(session)

    def check(self, paper_id: str) -> PaperAvailability:
        """返回论文当前可用状态。

        只有数据库记录、真实 PDF 文件、真实解析文本文件和完成的 embedding
        同时存在时，论文才可进入 RAG，并返回 ``FULL_AVAILABLE``。
        """
        paper = self.paper_repository.get_paper_by_id(paper_id)
        if paper is None:
            return PaperAvailability.METADATA_ONLY

        files = self.file_repository.list_by_paper(paper_id)
        pdf_available = _has_local_file(files, _PDF_TYPES, _AVAILABLE_FILE_STATUSES)
        text_available = _has_local_file(files, _TEXT_TYPES, _PARSED_TEXT_STATUSES)
        embedding_available = paper.embedding_status.casefold() in _EMBEDDING_READY_STATUSES

        if pdf_available and text_available and embedding_available:
            return PaperAvailability.FULL_AVAILABLE

        # 资源尚未完整时保留能够表达工作流状态的持久化值。
        try:
            stored = PaperAvailability(paper.availability)
        except ValueError:
            stored = PaperAvailability.METADATA_ONLY
        if stored in {
            PaperAvailability.DOWNLOADING,
            PaperAvailability.DOWNLOAD_FAILED,
            PaperAvailability.USER_UPLOAD_REQUIRED,
        }:
            return stored
        return PaperAvailability.METADATA_ONLY

    def is_rag_ready(self, paper_id: str) -> bool:
        """便捷判断论文是否可直接进入 RAG。"""
        return self.check(paper_id) is PaperAvailability.FULL_AVAILABLE


_PDF_TYPES = {"pdf", "fulltext", "full_text"}
_TEXT_TYPES = {"text", "txt", "parsed_text"}
_AVAILABLE_FILE_STATUSES = {"available", "completed", "downloaded", "success"}
_PARSED_TEXT_STATUSES = {"available", "completed", "parsed", "success"}
_EMBEDDING_READY_STATUSES = {"available", "completed", "generated", "ready", "success"}


def _has_local_file(
    records: Iterable[PaperFileModel],
    accepted_types: set[str],
    accepted_statuses: set[str],
) -> bool:
    for record in records:
        if (
            record.file_type.casefold() in accepted_types
            and record.status.casefold() in accepted_statuses
            and Path(record.file_path).is_file()
        ):
            return True
    return False
