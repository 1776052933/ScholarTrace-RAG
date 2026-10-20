"""Local PDF and paper collection operations shared by the API."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from app.api.schemas import IngestionRequest
from app.services.ingestions import IngestionJobService
from app.storage.models import (
    ConversationModel, MultiPaperAnswerModel, PaperCollectionItemModel,
    PaperCollectionModel, QuestionAnswerModel, ResearchJobModel, utc_now,
)
from literature_database.database import DatabaseManager
from literature_database.models import PaperFileModel, PaperModel


class LibraryConflictError(ValueError):
    pass


class PaperLibraryService:
    def __init__(self, database: DatabaseManager, data_root: Path, ingestions: IngestionJobService):
        self.database = database
        self.upload_dir = data_root / "uploads"
        self.ingestions = ingestions

    def collections(self) -> list[dict]:
        with self.database.get_session() as session:
            rows = session.scalars(select(PaperCollectionModel).order_by(PaperCollectionModel.created_at)).all()
            return [self._collection(session, row) for row in rows]

    @staticmethod
    def _collection(session, row: PaperCollectionModel) -> dict:
        members = session.scalars(select(PaperCollectionItemModel).where(
            PaperCollectionItemModel.collection_id == row.collection_id,
        ).order_by(PaperCollectionItemModel.added_at, PaperCollectionItemModel.paper_id)).all()
        return {"collection_id": row.collection_id, "name": row.name,
                "paper_ids": [item.paper_id for item in members]}

    def get_collection(self, collection_id: str) -> dict:
        with self.database.get_session() as session:
            row = session.get(PaperCollectionModel, collection_id)
            if row is None:
                raise KeyError(collection_id)
            return self._collection(session, row)

    def create_collection(self, name: str) -> dict:
        with self.database.get_session() as session:
            row = PaperCollectionModel(collection_id=uuid4().hex, name=name.strip())
            session.add(row)
            session.commit()
            return self._collection(session, row)

    def rename_collection(self, collection_id: str, name: str) -> dict:
        with self.database.get_session() as session:
            row = session.get(PaperCollectionModel, collection_id)
            if row is None:
                raise KeyError(collection_id)
            row.name, row.updated_at = name.strip(), utc_now()
            session.commit()
            return self._collection(session, row)

    def delete_collection(self, collection_id: str) -> None:
        with self.database.get_session() as session:
            row = session.get(PaperCollectionModel, collection_id)
            if row is None:
                raise KeyError(collection_id)
            session.query(PaperCollectionItemModel).filter_by(collection_id=collection_id).delete()
            session.delete(row)
            session.commit()

    def add_paper(self, collection_id: str, paper_id: str) -> dict:
        with self.database.get_session() as session:
            row = session.get(PaperCollectionModel, collection_id)
            if row is None or session.get(PaperModel, paper_id) is None:
                raise KeyError("collection or paper not found")
            if session.get(PaperCollectionItemModel, (collection_id, paper_id)) is None:
                session.add(PaperCollectionItemModel(collection_id=collection_id, paper_id=paper_id))
                row.updated_at = utc_now()
                session.commit()
            return self._collection(session, row)

    def remove_paper(self, collection_id: str, paper_id: str) -> dict:
        with self.database.get_session() as session:
            row = session.get(PaperCollectionModel, collection_id)
            if row is None:
                raise KeyError(collection_id)
            item = session.get(PaperCollectionItemModel, (collection_id, paper_id))
            if item is not None:
                session.delete(item)
                row.updated_at = utc_now()
                session.commit()
            return self._collection(session, row)

    def register_upload(self, temporary: Path, digest: str, title: str,
                        collection_id: str | None = None,
                        target_paper_id: str | None = None) -> tuple[str, str | None]:
        """Move a validated PDF into managed storage and queue its first indexing."""
        paper_id = target_paper_id or f"upload:{digest}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{sha256(paper_id.encode('utf-8')).hexdigest()[:16]}_" if target_paper_id else ""
        destination = self.upload_dir / f"{prefix}{digest}.pdf"
        created_file = False
        try:
            with self.database.get_session() as session:
                if collection_id and session.get(PaperCollectionModel, collection_id) is None:
                    raise KeyError(collection_id)
                paper = session.get(PaperModel, paper_id)
                if paper is None:
                    if target_paper_id:
                        raise KeyError(target_paper_id)
                    paper = PaperModel(paper_id=paper_id, title=title, authors=[], abstract="",
                                       venue="", citation_count=0, rank_score=0.0,
                                       availability="METADATA_ONLY", embedding_status="PENDING")
                    session.add(paper)
                    session.flush()
                elif target_paper_id:
                    if self._pdf_path(session, paper_id):
                        raise LibraryConflictError("该论文已有可用 PDF，无需再次关联上传。")
                    paper.embedding_status = "PENDING"
                    paper.index_signature = None
                    paper.availability = "METADATA_ONLY"
                if not destination.exists():
                    temporary.replace(destination)
                    created_file = True
                file_row = session.scalar(select(PaperFileModel).where(
                    PaperFileModel.paper_id == paper_id, PaperFileModel.file_type == "pdf",
                    PaperFileModel.file_path == str(destination)))
                if file_row is None:
                    session.add(PaperFileModel(paper_id=paper_id, file_path=str(destination),
                                               status="UPLOADED", file_type="pdf"))
                else:
                    file_row.file_path, file_row.status = str(destination), "UPLOADED"
                if collection_id and session.get(PaperCollectionItemModel, (collection_id, paper_id)) is None:
                    session.add(PaperCollectionItemModel(collection_id=collection_id, paper_id=paper_id))
                session.commit()
                ready = paper.embedding_status.casefold() == "completed"
        except Exception:
            if created_file:
                destination.unlink(missing_ok=True)
            raise
        if ready:
            return paper_id, None
        with self.database.get_session() as session:
            active = session.scalars(select(ResearchJobModel).where(
                ResearchJobModel.job_type == "ingestion",
                ResearchJobModel.status.in_(("queued", "running", "cancelling")),
            )).all()
            for job in active:
                if paper_id in (job.options or {}).get("paper_ids", []):
                    return paper_id, job.job_id
        return paper_id, self.reindex(paper_id)

    def reindex(self, paper_id: str) -> str:
        with self.database.get_session() as session:
            paper = session.get(PaperModel, paper_id)
            if paper is None:
                raise KeyError(paper_id)
            if not self._pdf_path(session, paper_id):
                raise LibraryConflictError("该论文没有可用的本地 PDF。")
            active = session.scalars(select(ResearchJobModel).where(
                ResearchJobModel.job_type == "ingestion",
                ResearchJobModel.status.in_(("queued", "running", "cancelling")),
            )).all()
            if any(paper_id in (job.options or {}).get("paper_ids", []) for job in active):
                raise LibraryConflictError("该论文已有正在进行的入库任务。")
            source_id = uuid4().hex
            session.add(ResearchJobModel(
                job_id=source_id, job_type="research", query=paper.title,
                options={"source": "local_library"}, status="completed", stage="completed",
                progress=100, logs=[], retryable=False,
                result={"stages": {"ranking_agent": {"result": {"papers": [{
                    "paper_id": paper_id, "title": paper.title,
                }]}}}}, finished_at=utc_now(),
            ))
            session.commit()
        return self.ingestions.create(IngestionRequest(
            discovery_job_id=source_id, paper_ids=[paper_id],
        )).job_id

    @staticmethod
    def _pdf_path(session, paper_id: str) -> Path | None:
        rows = session.scalars(select(PaperFileModel).where(
            PaperFileModel.paper_id == paper_id, PaperFileModel.file_type == "pdf",
        )).all()
        for row in rows:
            path = Path(row.file_path)
            if row.status.casefold() in {"available", "uploaded", "downloaded", "completed", "success"} and path.is_file():
                return path
        return None

    def pdf_path(self, paper_id: str) -> Path:
        with self.database.get_session() as session:
            if session.get(PaperModel, paper_id) is None:
                raise KeyError(paper_id)
            path = self._pdf_path(session, paper_id)
            if path is None:
                raise FileNotFoundError(paper_id)
            return path

    def delete_uploaded(self, paper_id: str) -> None:
        if not paper_id.startswith("upload:"):
            raise LibraryConflictError("仅能删除从本地上传的论文；检索论文可从集合中移除。")
        with self.database.get_session() as session:
            paper = session.get(PaperModel, paper_id)
            if paper is None:
                raise KeyError(paper_id)
            active = session.scalars(select(ResearchJobModel).where(
                ResearchJobModel.job_type == "ingestion",
                ResearchJobModel.status.in_(("queued", "running", "cancelling")),
            )).all()
            if any(paper_id in (job.options or {}).get("paper_ids", []) for job in active):
                raise LibraryConflictError("入库任务尚未结束，不能删除。")
            if session.scalar(select(QuestionAnswerModel.answer_id).where(QuestionAnswerModel.paper_id == paper_id).limit(1)):
                raise LibraryConflictError("这篇论文已有问答记录，请先保留其证据。")
            conversations = session.scalars(select(ConversationModel)).all()
            multi_answers = session.scalars(select(MultiPaperAnswerModel)).all()
            if any(paper_id in (row.paper_ids or []) for row in (*conversations, *multi_answers)):
                raise LibraryConflictError("这篇论文已有多论文问答或会话记录，请先保留其证据。")
            files = session.scalars(select(PaperFileModel).where(PaperFileModel.paper_id == paper_id)).all()
            safe_files = []
            for row in files:
                path = Path(row.file_path).resolve()
                if path.is_relative_to(self.upload_dir.resolve()) or (
                    row.file_type.casefold() in {"text", "txt", "parsed_text"}
                    and path.is_relative_to(self.upload_dir.parent.resolve() / "papers" / "text")
                ):
                    safe_files.append(path)
            # Delete vectors before metadata; a vector-store failure leaves the document intact.
            from rag_reader.config import RAGSettings
            from rag_reader.store_factory import create_vector_store
            settings = RAGSettings.from_env()
            create_vector_store(settings, chroma_directory=str(self.upload_dir.parent / "chroma")).delete(
                paper_id=paper_id,
            )
            session.query(PaperCollectionItemModel).filter_by(paper_id=paper_id).delete()
            session.delete(paper)
            session.commit()
        for path in safe_files:
            path.unlink(missing_ok=True)
