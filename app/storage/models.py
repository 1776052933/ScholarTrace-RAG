"""M2 durable research-job and question records."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from literature_database.models import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ResearchJobModel(Base):
    __tablename__ = "research_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    job_type: Mapped[str] = mapped_column(String(24), nullable=False, default="research", index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    stage: Mapped[str | None] = mapped_column(String(80), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    logs: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retryable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (Index("ix_research_jobs_created_at", "created_at"),)


class QuestionAnswerModel(Base):
    __tablename__ = "question_answers"

    answer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    paper_id: Mapped[str] = mapped_column(String(255), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class MultiPaperAnswerModel(Base):
    __tablename__ = "multi_paper_answers"

    answer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    paper_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class ConversationModel(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    paper_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    scope_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_turn_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    active_since: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now, index=True)


class ConversationTurnModel(Base):
    """One exchange; only completed exchanges may enter a later model context."""

    __tablename__ = "conversation_turns"
    turn_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False,
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (UniqueConstraint("conversation_id", "turn_number", name="uq_conversation_turn_number"),)


class PaperCollectionModel(Base):
    __tablename__ = "paper_collections"

    collection_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now, onupdate=utc_now)


class PaperCollectionItemModel(Base):
    __tablename__ = "paper_collection_items"

    collection_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("paper_collections.collection_id", ondelete="CASCADE"), primary_key=True,
    )
    paper_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("papers.paper_id", ondelete="CASCADE"), primary_key=True,
    )
    added_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)

    __table_args__ = (Index("ix_paper_collection_items_paper_id", "paper_id"),)
