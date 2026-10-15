"""Paper question service using the configured RAG index."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from typing import Any

from sqlalchemy import select

from app.storage.models import QuestionAnswerModel
from app.config import ModelSettings
from literature_database.database import DatabaseManager
from literature_database.models import PaperModel
from rag_reader.config import RAGSettings
from rag_reader.llm_client import DeepSeekClient
from rag_reader.pipeline import RAGReader
from rag_reader.store_factory import create_vector_store, milvus_index_scope


class PaperIndexUnavailableError(RuntimeError):
    """The selected vector index does not contain this paper version."""


class QuestionService:
    def __init__(self, database: DatabaseManager, data_root: Path) -> None:
        self.database = database
        self.data_root = Path(data_root)

    def ask(self, paper_id: str, question: str) -> tuple[str, dict[str, Any]]:
        settings = RAGSettings.from_env()
        with self.database.get_session() as session:
            paper = session.get(PaperModel, paper_id)
            if paper is None:
                raise KeyError(paper_id)
            if settings.vector_backend == "milvus" and (
                paper.embedding_status.casefold() != "completed"
                or not (paper.index_signature or "").startswith(milvus_index_scope(settings) + ":")
            ):
                raise PaperIndexUnavailableError(
                    f"paper {paper_id} has not been rebuilt in the current Milvus index"
                )
        provider = ModelSettings.from_env()
        settings = settings.model_copy(update={
            "llm_model": provider.model,
            "llm_base_url": provider.base_url,
            "llm_timeout_seconds": provider.timeout_seconds,
        })
        store = create_vector_store(settings, chroma_directory=str(self.data_root / "chroma"))
        if settings.vector_backend == "milvus" and not store.has_paper(paper_id):
            raise PaperIndexUnavailableError(
                f"paper {paper_id} is absent from the current Milvus collection"
            )
        llm = DeepSeekClient(**provider.client_kwargs(), timeout=provider.timeout_seconds)
        reader = RAGReader(vector_store=store, llm_client=llm, settings=settings)
        answer = reader.ask(paper_id, question).model_dump(mode="json")
        answer_id = uuid4().hex
        with self.database.get_session() as session:
            session.add(QuestionAnswerModel(answer_id=answer_id, paper_id=paper_id, question=question, answer=answer))
            session.commit()
        return answer_id, answer

    def evidence(self, answer_id: str) -> list[dict[str, Any]]:
        with self.database.get_session() as session:
            row = session.get(QuestionAnswerModel, answer_id)
            if row is None:
                raise KeyError(answer_id)
            return [dict(item) for item in row.answer.get("citations") or [] if isinstance(item, dict)]
