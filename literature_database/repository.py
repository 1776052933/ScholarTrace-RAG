"""论文、文件和人工上传记录的 SQLAlchemy CRUD 仓储。"""

from collections.abc import Mapping
from typing import Any, List, Optional, Union

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import PaperFileModel, PaperModel, UploadQueueModel
from .schema import PaperAvailability, PaperImportance, PaperMetadata, UploadRequest


class PaperRepository:
    """只负责论文表 CRUD，不包含下载或重要性业务判断。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_paper(
        self,
        paper: Union[PaperMetadata, PaperModel, Mapping[str, Any]],
    ) -> PaperModel:
        model = paper if isinstance(paper, PaperModel) else _paper_model(paper)
        self.session.add(model)
        self.session.flush()
        return model

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperModel]:
        return self.session.get(PaperModel, paper_id)

    def find_by_doi(self, doi: str) -> Optional[PaperModel]:
        value = doi.strip()
        if not value:
            return None
        statement = select(PaperModel).where(func.lower(PaperModel.doi) == value.casefold())
        return self.session.scalar(statement)

    def find_by_title(self, title: str) -> Optional[PaperModel]:
        value = title.strip()
        if not value:
            return None
        statement = select(PaperModel).where(func.lower(PaperModel.title) == value.casefold())
        return self.session.scalar(statement)

    def update_status(
        self,
        paper_id: str,
        status: Union[PaperAvailability, str],
    ) -> Optional[PaperModel]:
        model = self.get_paper_by_id(paper_id)
        if model is None:
            return None
        model.availability = PaperAvailability(status).value
        self.session.flush()
        return model

    def delete_paper(self, paper_id: str) -> bool:
        model = self.get_paper_by_id(paper_id)
        if model is None:
            return False
        self.session.delete(model)
        self.session.flush()
        return True


class FileRepository:
    """论文本地文件记录的基础 CRUD。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_file(
        self,
        paper_id: str,
        file_path: str,
        *,
        status: str = "AVAILABLE",
        file_type: str = "pdf",
    ) -> PaperFileModel:
        model = PaperFileModel(
            paper_id=paper_id,
            file_path=file_path,
            status=status,
            file_type=file_type,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def get_file_by_id(self, file_id: int) -> Optional[PaperFileModel]:
        return self.session.get(PaperFileModel, file_id)

    def list_by_paper(self, paper_id: str) -> List[PaperFileModel]:
        statement = (
            select(PaperFileModel)
            .where(PaperFileModel.paper_id == paper_id)
            .order_by(PaperFileModel.id)
        )
        return list(self.session.scalars(statement))

    def update_status(self, file_id: int, status: str) -> Optional[PaperFileModel]:
        model = self.get_file_by_id(file_id)
        if model is None:
            return None
        model.status = status
        self.session.flush()
        return model

    def delete_file(self, file_id: int) -> bool:
        model = self.get_file_by_id(file_id)
        if model is None:
            return False
        self.session.delete(model)
        self.session.flush()
        return True


class UploadRepository:
    """人工上传队列表的基础 CRUD。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add_upload(
        self,
        paper_id: str,
        importance: Union[PaperImportance, str] = PaperImportance.NORMAL,
        *,
        status: str = "PENDING",
        reason: str = "",
    ) -> UploadQueueModel:
        model = UploadQueueModel(
            paper_id=paper_id,
            importance=PaperImportance(importance).value,
            status=status,
            reason=reason,
        )
        self.session.add(model)
        self.session.flush()
        return model

    def add_request(
        self,
        request: UploadRequest,
        *,
        status: str = "PENDING",
    ) -> UploadQueueModel:
        return self.add_upload(
            request.paper_id,
            request.priority,
            status=status,
            reason=request.reason,
        )

    def get_upload_by_id(self, upload_id: int) -> Optional[UploadQueueModel]:
        return self.session.get(UploadQueueModel, upload_id)

    def list_by_paper(self, paper_id: str) -> List[UploadQueueModel]:
        statement = (
            select(UploadQueueModel)
            .where(UploadQueueModel.paper_id == paper_id)
            .order_by(UploadQueueModel.id)
        )
        return list(self.session.scalars(statement))

    def list_by_status(self, status: str) -> List[UploadQueueModel]:
        statement = (
            select(UploadQueueModel)
            .where(UploadQueueModel.status == status)
            .order_by(UploadQueueModel.id)
        )
        return list(self.session.scalars(statement))

    def update_status(self, upload_id: int, status: str) -> Optional[UploadQueueModel]:
        model = self.get_upload_by_id(upload_id)
        if model is None:
            return None
        model.status = status
        self.session.flush()
        return model

    def delete_upload(self, upload_id: int) -> bool:
        model = self.get_upload_by_id(upload_id)
        if model is None:
            return False
        self.session.delete(model)
        self.session.flush()
        return True


def _paper_model(value: Union[PaperMetadata, Mapping[str, Any]]) -> PaperModel:
    data = value.model_dump(mode="python") if isinstance(value, PaperMetadata) else dict(value)
    metadata = PaperMetadata.model_validate(data)
    return PaperModel(**metadata.model_dump(mode="python"))
