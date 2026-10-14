"""Persistent cache for validated DeepSeek paper evaluations."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any, Dict, Optional

from .prompts import PAPER_BATCH_EVALUATION_PROMPT, PAPER_EVALUATION_PROMPT
from .schema import Paper, QueryContext


logger = logging.getLogger(__name__)

CACHE_VERSION = "ranking-evaluation-v2"


class EvaluationCache:
    """Store only successfully validated LLM evaluations in SQLite."""

    def __init__(self, path: Optional[str] = None) -> None:
        configured = path or os.getenv("RANKING_CACHE_PATH")
        self.path = Path(configured or "./data/ranking_evaluations.sqlite3").expanduser()
        self._write_lock = Lock()
        self._enabled = self._initialize()

    def get(
        self,
        context: QueryContext,
        paper: Paper,
        *,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        cache_key = _cache_key(context, paper, model=model)
        try:
            with sqlite3.connect(self.path, timeout=10.0) as connection:
                row = connection.execute(
                    "SELECT response_json FROM evaluations WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("读取论文评分缓存失败：%s", exc)
            return None
        if row is None:
            return None
        try:
            value = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
        if not _valid_evaluation(value):
            return None
        logger.info("复用论文评分缓存：%s", paper.paper_id)
        return dict(value)

    def put(
        self,
        context: QueryContext,
        paper: Paper,
        evaluation: Dict[str, Any],
        *,
        model: str,
    ) -> None:
        if not self._enabled or not _valid_evaluation(evaluation):
            return
        cache_key = _cache_key(context, paper, model=model)
        payload = json.dumps(evaluation, ensure_ascii=False, sort_keys=True)
        try:
            with self._write_lock, sqlite3.connect(self.path, timeout=10.0) as connection:
                connection.execute(
                    """
                    INSERT INTO evaluations(cache_key, response_json, created_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response_json = excluded.response_json,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (cache_key, payload),
                )
                connection.commit()
        except sqlite3.Error as exc:
            logger.warning("写入论文评分缓存失败：%s", exc)

    def _initialize(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path, timeout=10.0) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evaluations (
                        cache_key TEXT PRIMARY KEY,
                        response_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            return True
        except (OSError, sqlite3.Error) as exc:
            logger.warning("初始化论文评分缓存失败，将继续无缓存运行：%s", exc)
            return False


def _cache_key(context: QueryContext, paper: Paper, *, model: str) -> str:
    payload = {
        "version": CACHE_VERSION,
        "model": model.strip() or "deepseek-chat",
        "prompt_sha256": sha256(
            (PAPER_EVALUATION_PROMPT + PAPER_BATCH_EVALUATION_PROMPT).encode("utf-8")
        ).hexdigest(),
        "query_context": context.model_dump(mode="json"),
        "paper": paper.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _valid_evaluation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for name in ("relevance_score", "method_match", "task_match"):
        score = value.get(name)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if not 0.0 <= float(score) <= 1.0:
            return False
    return isinstance(value.get("reason"), str)
