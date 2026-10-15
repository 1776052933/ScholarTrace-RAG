"""Durable M2 job execution around the existing domain workflow."""
from __future__ import annotations

from datetime import datetime
import logging
import re
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select, update

from app.api.schemas import JobSnapshot, ResearchOptions
from app.storage.models import ResearchJobModel, utc_now
from literature_database.database import DatabaseManager


_STAGE_PROGRESS = {
    "query_understanding": 8,
    "query_decomposition": 18,
    "search_agent": 32,
    "citation_tracking": 47,
    "ranking_agent": 55,
    "literature_database": 75,
    "rag_reader": 92,
}
_ACTIVE = {"queued", "running", "cancelling"}
_SECRET_PATTERN = re.compile(r"(?i)(\b(?:api[_-]?key|token|secret)\s*[=:]\s*)[^\s,;]+|\bsk-[A-Za-z0-9_-]{12,}\b")


def _safe_message(value: str) -> str:
    return _SECRET_PATTERN.sub(lambda match: (match.group(1) or "") + "[REDACTED]", str(value))


class JobBusyError(RuntimeError):
    """The single-process M2 executor is already running a research job."""


class JobStore:
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def create(self, options: ResearchOptions) -> JobSnapshot:
        job_id = uuid4().hex
        with self.database.get_session() as session:
            row = ResearchJobModel(
                job_id=job_id,
                query=options.query,
                options=options.model_dump(mode="json"),
                status="queued",
                progress=0,
                logs=[],
                retryable=False,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._snapshot(row)

    def get(self, job_id: str) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None:
                raise KeyError(job_id)
            return self._snapshot(row)

    def list_recent(self, limit: int = 20) -> list[JobSnapshot]:
        with self.database.get_session() as session:
            rows = session.scalars(
                select(ResearchJobModel).order_by(ResearchJobModel.created_at.desc()).limit(limit)
            ).all()
            return [self._snapshot(row) for row in rows]

    def update(self, job_id: str, **changes: Any) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None:
                raise KeyError(job_id)
            for name, value in changes.items():
                setattr(row, name, value)
            row.updated_at = utc_now()
            session.commit()
            session.refresh(row)
            return self._snapshot(row)

    def request_cancel(self, job_id: str) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None:
                raise KeyError(job_id)
            if row.status in {"queued", "running"}:
                row.status = "cancelling"
                row.updated_at = utc_now()
                session.commit()
                session.refresh(row)
            return self._snapshot(row)

    def append_log(self, job_id: str, line: str) -> None:
        line = _safe_message(line)
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None or row.status not in _ACTIVE:
                return
            row.logs = [*(row.logs or []), line][-100:]
            for stage, percent in _STAGE_PROGRESS.items():
                if f"开始阶段：{stage}" in line:
                    row.stage = stage
                    row.progress = max(row.progress, percent)
                elif f"完成阶段：{stage}" in line:
                    row.stage = stage
                    row.progress = max(row.progress, min(percent + 8, 97))
            row.updated_at = utc_now()
            session.commit()

    def mark_interrupted(self) -> int:
        with self.database.get_session() as session:
            result = session.execute(
                update(ResearchJobModel)
                .where(ResearchJobModel.status.in_(tuple(_ACTIVE)))
                .values(status="interrupted", error="API 服务重启，运行中的任务已中断。", retryable=True, finished_at=utc_now(), updated_at=utc_now())
            )
            session.commit()
            return int(result.rowcount or 0)

    @staticmethod
    def _snapshot(row: ResearchJobModel) -> JobSnapshot:
        end = row.finished_at or utc_now()
        start = row.started_at or row.created_at
        return JobSnapshot(
            job_id=row.job_id,
            query=row.query,
            status=row.status,
            stage=row.stage,
            progress=row.progress,
            result=row.result,
            error=row.error,
            retryable=row.retryable,
            logs=row.logs or [],
            created_at=row.created_at,
            updated_at=row.updated_at,
            elapsed_seconds=max(0.0, (end - start).total_seconds()),
        )


class _StageLogHandler(logging.Handler):
    def __init__(self, store: JobStore, job_id: str, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.store = store
        self.job_id = job_id
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id:
            return
        try:
            self.store.append_log(self.job_id, record.getMessage())
        except Exception:
            self.handleError(record)


class ResearchJobRunner:
    """Single-process executor; the database is the source of truth for job state."""

    def __init__(
        self,
        database: DatabaseManager,
        data_root: Path,
        *,
        workflow_factory: Callable[[ResearchOptions], Any] | None = None,
    ) -> None:
        self.database = database
        self.store = JobStore(database)
        self.data_root = Path(data_root)
        self.workflow_factory = workflow_factory or self._default_workflow
        self._events: dict[str, Event] = {}
        self._threads: dict[str, Thread] = {}
        self._lock = Lock()

    def _default_workflow(self, options: ResearchOptions) -> Any:
        from app.services.workflow import AutoPaperWorkflow
        from app.config import ModelSettings
        return AutoPaperWorkflow(
            use_query_llm=options.use_query_llm,
            max_search_rounds=options.max_search_rounds,
            database_url=self.database.database_url,
            chroma_directory=str(self.data_root / "chroma"),
            enable_rag=options.enable_rag,
            enable_citation_tracking=options.enable_citation_tracking,
            model_settings=ModelSettings.from_env(),
        )

    def start(self, options: ResearchOptions) -> JobSnapshot:
        with self._lock:
            if any(thread.is_alive() for thread in self._threads.values()):
                raise JobBusyError("已有研究任务正在运行，请等待完成或取消后重试。")
            snapshot = self.store.create(options)
            event = Event()
            thread = Thread(target=self._execute, args=(snapshot.job_id, options, event), daemon=True, name=f"research-{snapshot.job_id[:8]}")
            self._events[snapshot.job_id] = event
            self._threads[snapshot.job_id] = thread
            try:
                thread.start()
            except RuntimeError as exc:
                self._events.pop(snapshot.job_id, None)
                self._threads.pop(snapshot.job_id, None)
                self.store.update(snapshot.job_id, status="failed", error=_safe_message(str(exc)), retryable=True, finished_at=utc_now())
                raise
        return self.store.get(snapshot.job_id)

    def cancel(self, job_id: str) -> JobSnapshot:
        snapshot = self.store.get(job_id)
        if snapshot.status not in _ACTIVE:
            return snapshot
        with self._lock:
            event = self._events.get(job_id)
        if event is None:
            return self.store.update(job_id, status="interrupted", error="任务执行器已失联。", retryable=True, finished_at=utc_now())
        event.set()
        return self.store.request_cancel(job_id)

    def join(self, job_id: str, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)

    def _execute(self, job_id: str, options: ResearchOptions, event: Event) -> None:
        from app.services.workflow import WorkflowCancelled

        logger = logging.getLogger("auto_paper")
        logger.disabled = False
        logger.setLevel(logging.INFO)
        handler = _StageLogHandler(self.store, job_id, get_ident())
        logger.addHandler(handler)
        try:
            self.store.update(job_id, status="running", started_at=utc_now())
            if event.is_set():
                raise WorkflowCancelled("用户已取消本次检索。")
            self.data_root.mkdir(parents=True, exist_ok=True)
            (self.data_root / "papers" / "text").mkdir(parents=True, exist_ok=True)
            workflow = self.workflow_factory(options)
            result = workflow.run(
                options.query,
                max_papers=options.max_papers,
                download_papers=options.download_papers,
                ingest_rag=options.enable_rag,
                papers_directory=str(self.data_root / "papers"),
                text_directory=str(self.data_root / "papers" / "text"),
                download_engine=options.download_engine,
                download_timeout=options.download_timeout,
                retry_failed_downloads=options.retry_failed_downloads,
                question=options.question,
                summarize=options.summarize,
                extract_knowledge=options.extract_knowledge,
                publication_date_to=options.publication_date_to,
                should_cancel=event.is_set,
            )
            if event.is_set():
                raise WorkflowCancelled("用户已取消本次检索。")
            self.store.update(job_id, status="completed", stage="completed", progress=100, result=result, finished_at=utc_now())
        except WorkflowCancelled:
            self.store.update(job_id, status="cancelled", stage="cancelled", error=None, retryable=True, finished_at=utc_now())
        except Exception as exc:
            logger.exception("研究任务失败：%s", type(exc).__name__)
            self.store.update(job_id, status="failed", stage="failed", error=_safe_message(f"{type(exc).__name__}: {exc}"), retryable=True, finished_at=utc_now())
        finally:
            logger.removeHandler(handler)
            with self._lock:
                self._events.pop(job_id, None)
                self._threads.pop(job_id, None)
