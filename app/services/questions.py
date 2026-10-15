"""M2 paper question service backed by the existing Chroma index."""
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
from rag_reader.vector_store import VectorStore


class QuestionService:
    def __init__(self, database: DatabaseManager, data_root: Path) -> None:
        self.database = database
        self.data_root = Path(data_root)

    def ask(self, paper_id: str, question: str) -> tuple[str, dict[str, Any]]:
        with self.database.get_session() as session:
            if session.get(PaperModel, paper_id) is None:
                raise KeyError(paper_id)
        provider = ModelSettings.from_env()
        settings = RAGSettings.from_env().model_copy(update={
            "llm_model": provider.model,
            "llm_base_url": provider.base_url,
            "llm_timeout_seconds": provider.timeout_seconds,
        })
        llm = DeepSeekClient(**provider.client_kwargs(), timeout=provider.timeout_seconds)
        reader = RAGReader(
            vector_store=VectorStore(persist_directory=str(self.data_root / "chroma")),
            llm_client=llm,
            settings=settings,
        )
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
