"""Paper question service using the configured RAG index."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from typing import Any

from sqlalchemy import select

from app.storage.models import MultiPaperAnswerModel, QuestionAnswerModel
from app.config import ModelSettings
from literature_database.database import DatabaseManager
from literature_database.models import PaperModel
from rag_reader.config import RAGSettings
from rag_reader.llm_client import DeepSeekClient
from rag_reader.pipeline import RAGReader
from rag_reader.store_factory import create_vector_store, milvus_index_scope
from observability.tracing import trace_span, flush_traces


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
        with trace_span("rag.question", input={"paper_ids": [paper_id]}) as span:
            answer = reader.ask(paper_id, question).model_dump(mode="json")
            span.update(output={"source_count": len(answer["sources"]),
                                "confidence": answer["confidence"]})
        answer_id = uuid4().hex
        with self.database.get_session() as session:
            session.add(QuestionAnswerModel(answer_id=answer_id, paper_id=paper_id, question=question, answer=answer))
            session.commit()
        flush_traces()
        return answer_id, answer

    def ask_many(self, paper_ids: list[str], question: str) -> tuple[str, dict[str, Any]]:
        if len(paper_ids) < 2 or len(paper_ids) > 5 or len(set(paper_ids)) != len(paper_ids):
            raise ValueError("select two to five distinct papers")
        settings = RAGSettings.from_env()
        with self.database.get_session() as session:
            for paper_id in paper_ids:
                paper = session.get(PaperModel, paper_id)
                if paper is None:
                    raise KeyError(paper_id)
                if settings.vector_backend == "milvus" and (
                    paper.embedding_status.casefold() != "completed"
                    or not (paper.index_signature or "").startswith(milvus_index_scope(settings) + ":")
                ):
                    raise PaperIndexUnavailableError(f"paper {paper_id} is not ready in the current Milvus index")
        provider = ModelSettings.from_env()
        settings = settings.model_copy(update={
            "llm_model": provider.model,
            "llm_base_url": provider.base_url,
            "llm_timeout_seconds": provider.timeout_seconds,
        })
        store = create_vector_store(settings, chroma_directory=str(self.data_root / "chroma"))
        if settings.vector_backend == "milvus":
            missing = [paper_id for paper_id in paper_ids if not store.has_paper(paper_id)]
            if missing:
                raise PaperIndexUnavailableError(f"papers absent from the current Milvus collection: {missing}")
        llm = DeepSeekClient(**provider.client_kwargs(), timeout=provider.timeout_seconds)
        reader = RAGReader(vector_store=store, llm_client=llm, settings=settings)
        with trace_span("rag.question", input={"paper_ids": paper_ids}) as span:
            answer = reader.qa.ask_many(paper_ids, question).model_dump(mode="json")
            span.update(output={"source_count": len(answer["sources"]),
                                "confidence": answer["confidence"]})
        answer_id = uuid4().hex
        with self.database.get_session() as session:
            session.add(MultiPaperAnswerModel(
                answer_id=answer_id, paper_ids=paper_ids, question=question, answer=answer,
            ))
            session.commit()
        flush_traces()
        return answer_id, answer

    def evidence(self, answer_id: str) -> list[dict[str, Any]]:
        with self.database.get_session() as session:
            row = session.get(QuestionAnswerModel, answer_id)
            if row is None:
                row = session.get(MultiPaperAnswerModel, answer_id)
            if row is None:
                raise KeyError(answer_id)
            return [dict(item) for item in row.answer.get("citations") or [] if isinstance(item, dict)]

    def export(self, answer_id: str) -> str:
        with self.database.get_session() as session:
            row = session.get(MultiPaperAnswerModel, answer_id)
            if row is None:
                row = session.get(QuestionAnswerModel, answer_id)
            if row is None:
                raise KeyError(answer_id)
            ids = row.paper_ids if isinstance(row, MultiPaperAnswerModel) else [row.paper_id]
            answer = row.answer
            lines = ["# ScholarTrace 论文问答", "", f"问题：{row.question}",
                     f"论文：{', '.join(ids)}", "", str(answer.get("answer") or ""), "", "## 引用", ""]
            for citation in answer.get("citations") or []:
                if isinstance(citation, dict):
                    lines.append(f"- {citation.get('title') or citation.get('paper_id') or '论文'}，"
                                 f"第 {citation.get('page') or '?'} 页，"
                                 f"chunk `{citation.get('chunk_id') or ''}`："
                                 f"{str(citation.get('excerpt') or '').replace(chr(10), ' ')[:500]}")
            return "\n".join(lines).strip() + "\n"
