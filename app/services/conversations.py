"""Durable, scoped conversations with bounded history and atomic streamed turns."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
from typing import Any, Iterator
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import or_, select, update

from app.storage.models import (
    ConversationModel, ConversationTurnModel, MultiPaperAnswerModel, QuestionAnswerModel, utc_now,
)
from observability.tracing import flush_traces
from rag_reader.errors import RAGGenerationError


class ConversationBusyError(RuntimeError):
    pass


class ConversationSettings(BaseModel):
    history_max_turns: int = Field(default=6, ge=0, le=30)
    history_max_chars: int = Field(default=6000, ge=0, le=30000)
    prompt_max_chars: int = Field(default=24000, ge=4000, le=200000)
    max_output_tokens: int = Field(default=2048, ge=128, le=8192)
    lease_seconds: int = Field(default=900, ge=120, le=3600)

    @classmethod
    def from_env(cls):
        return cls(**{name: os.environ["CHAT_" + name.upper()]
                      for name in cls.model_fields if "CHAT_" + name.upper() in os.environ})


@dataclass
class PreparedTurn:
    conversation_id: str
    turn_id: str
    question: str
    paper_ids: list[str]
    papers: list[dict[str, str]]
    history: list[dict[str, str]]
    context: dict[str, Any]
    settings: ConversationSettings


def scope_key(paper_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(paper_ids), ensure_ascii=False).encode()).hexdigest()


def bounded_history(turns, settings: ConversationSettings) -> list[dict[str, str]]:
    """Take a suffix of complete exchanges; never include a half-finished answer."""
    pairs = []
    for turn in reversed(turns[-settings.history_max_turns:] if settings.history_max_turns else []):
        pair = [{"role": "user", "content": turn.question},
                {"role": "assistant", "content": str(turn.answer["answer"])}]
        candidate = pair + [message for previous in reversed(pairs) for message in previous]
        if len(json.dumps(candidate, ensure_ascii=False)) > settings.history_max_chars:
            break
        pairs.append(pair)
    return [message for pair in reversed(pairs) for message in pair]


class ConversationService:
    def __init__(self, database, questions, *, settings: ConversationSettings | None = None):
        self.database = database
        self.questions = questions
        self.settings = settings or ConversationSettings.from_env()

    def create(self, paper_ids: list[str], title: str | None = None) -> dict[str, Any]:
        papers = self.questions.validate_papers(paper_ids)
        row = ConversationModel(
            conversation_id=uuid4().hex, paper_ids=paper_ids, scope_key=scope_key(paper_ids),
            title=(title or " / ".join(p["title"] for p in papers))[:200],
        )
        with self.database.get_session() as session:
            session.add(row)
            session.commit()
            return self._summary(row)

    @staticmethod
    def _summary(row) -> dict[str, Any]:
        return {"conversation_id": row.conversation_id, "title": row.title,
                "paper_ids": row.paper_ids, "turn_count": row.turn_count,
                "active_turn_id": row.active_turn_id,
                "created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat()}

    def recent(self, paper_ids: list[str] | None = None, limit: int = 20):
        statement = select(ConversationModel)
        if paper_ids:
            statement = statement.where(ConversationModel.scope_key == scope_key(paper_ids))
        with self.database.get_session() as session:
            return [self._summary(row) for row in session.scalars(
                statement.order_by(ConversationModel.updated_at.desc()).limit(limit),
            )]

    def get(self, conversation_id: str, *, limit: int = 50, before: int | None = None):
        with self.database.get_session() as session:
            row = session.get(ConversationModel, conversation_id)
            if row is None:
                raise KeyError(conversation_id)
            statement = select(ConversationTurnModel).where(
                ConversationTurnModel.conversation_id == conversation_id,
            )
            if before is not None:
                statement = statement.where(ConversationTurnModel.turn_number < before)
            turns = list(reversed(session.scalars(statement.order_by(
                ConversationTurnModel.turn_number.desc()).limit(limit + 1)).all()))
            has_older = len(turns) > limit
            turns = turns[-limit:]
            return {**self._summary(row), "has_older": has_older, "turns": [
                {"turn_id": t.turn_id, "turn_number": t.turn_number, "status": t.status,
                 "question": t.question, "retrieval_question": t.retrieval_question,
                 "answer": t.answer, "context": t.context, "error": t.error,
                 "created_at": t.created_at.isoformat()} for t in turns
            ]}

    def begin(self, conversation_id: str, question: str) -> PreparedTurn:
        now = utc_now()
        cutoff = now - timedelta(seconds=self.settings.lease_seconds)
        with self.database.get_session() as session:
            row = session.get(ConversationModel, conversation_id)
            if row is None:
                raise KeyError(conversation_id)
            papers = self.questions.validate_papers(row.paper_ids)
            turn_id = uuid4().hex
            old_active = row.active_turn_id
            result = session.execute(update(ConversationModel).where(
                ConversationModel.conversation_id == conversation_id,
                ConversationModel.turn_count == row.turn_count,
                or_(ConversationModel.active_turn_id.is_(None), ConversationModel.active_since < cutoff),
            ).values(active_turn_id=turn_id, active_since=now,
                     turn_count=row.turn_count + 1, updated_at=now),
                execution_options={"synchronize_session": False})
            if result.rowcount != 1:
                raise ConversationBusyError("当前会话正在生成回答，请等待完成后继续。")
            if old_active:
                session.execute(update(ConversationTurnModel).where(
                    ConversationTurnModel.turn_id == old_active,
                    ConversationTurnModel.status == "generating",
                ).values(status="interrupted", error="上次生成已失联，请重新提问。", finished_at=now))
            completed = session.scalars(select(ConversationTurnModel).where(
                ConversationTurnModel.conversation_id == conversation_id,
                ConversationTurnModel.status == "completed",
            ).order_by(ConversationTurnModel.turn_number.desc()).limit(self.settings.history_max_turns)).all()
            history = bounded_history(list(reversed(completed)), self.settings)
            context = {"history_turns": len(history) // 2,
                       "history_chars": len(json.dumps(history, ensure_ascii=False)),
                       "history_max_turns": self.settings.history_max_turns,
                       "history_max_chars": self.settings.history_max_chars,
                       "prompt_max_chars": self.settings.prompt_max_chars}
            session.add(ConversationTurnModel(
                turn_id=turn_id, conversation_id=conversation_id, turn_number=row.turn_count + 1,
                question=question, status="generating", context=context,
            ))
            session.commit()
            return PreparedTurn(conversation_id, turn_id, question, list(row.paper_ids),
                                papers, history, context, self.settings)

    def stream(self, turn: PreparedTurn) -> Iterator[dict[str, Any]]:
        completed = False
        try:
            yield {"event": "meta", "data": {
                "conversation_id": turn.conversation_id, "turn_id": turn.turn_id,
                "paper_ids": turn.paper_ids, "context": turn.context,
            }}
            reader = self.questions.reader_for(turn.paper_ids)
            retrieval_question = turn.question
            for event in reader.qa.stream(
                turn.paper_ids, turn.question,
                history=turn.history, paper_titles=turn.papers,
                prompt_max_chars=turn.settings.prompt_max_chars,
                max_tokens=turn.settings.max_output_tokens,
            ):
                if event["event"] == "status" and "retrieval_question" in event["data"]:
                    retrieval_question = event["data"]["retrieval_question"]
                if event["event"] == "status" and "prompt_chars" in event["data"]:
                    turn.context["prompt_chars"] = event["data"]["prompt_chars"]
                if event["event"] == "answer":
                    answer = event["data"]
                    self._complete(turn, retrieval_question, answer)
                    completed = True
                    yield {"event": "done", "data": {
                        "conversation_id": turn.conversation_id, "turn_id": turn.turn_id,
                        "answer_id": turn.turn_id, "answer": answer,
                        "retrieval_question": retrieval_question, "context": turn.context,
                    }}
                    return
                yield event
            raise RAGGenerationError("回答流未完整结束。")
        except GeneratorExit:
            if not completed:
                self.fail(turn, "cancelled", "连接已断开，本轮未完成，未加入后续上下文。")
            raise
        except Exception:
            self.fail(turn, "failed", "本轮生成失败，未加入后续上下文，请重试。")
            raise
        finally:
            if not completed:
                self.fail(turn, "cancelled", "本轮未完成，请重试。")
            flush_traces()

    def _complete(self, turn, retrieval_question, answer):
        with self.database.get_session() as session:
            released = session.execute(update(ConversationModel).where(
                ConversationModel.conversation_id == turn.conversation_id,
                ConversationModel.active_turn_id == turn.turn_id,
            ).values(active_turn_id=None, active_since=None, updated_at=utc_now()))
            if released.rowcount != 1:
                raise ConversationBusyError("本轮生成已过期，请刷新会话。")
            row = session.get(ConversationTurnModel, turn.turn_id)
            row.status, row.answer = "completed", answer
            row.retrieval_question, row.context, row.finished_at = retrieval_question, turn.context, utc_now()
            if len(turn.paper_ids) == 1:
                session.add(QuestionAnswerModel(answer_id=turn.turn_id, paper_id=turn.paper_ids[0],
                                               question=turn.question, answer=answer))
            else:
                session.add(MultiPaperAnswerModel(answer_id=turn.turn_id, paper_ids=turn.paper_ids,
                                                 question=turn.question, answer=answer))
            session.commit()

    def fail(self, turn, status, message):
        with self.database.get_session() as session:
            session.execute(update(ConversationTurnModel).where(
                ConversationTurnModel.turn_id == turn.turn_id,
                ConversationTurnModel.status == "generating",
            ).values(status=status, error=message, finished_at=utc_now()))
            session.execute(update(ConversationModel).where(
                ConversationModel.conversation_id == turn.conversation_id,
                ConversationModel.active_turn_id == turn.turn_id,
            ).values(active_turn_id=None, active_since=None))
            session.commit()
