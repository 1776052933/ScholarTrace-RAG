"""Durable M2 job execution around the existing domain workflow."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import logging
import re
from pathlib import Path
from threading import Event, Lock, Thread, get_ident
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select, update

from app.api.schemas import JobListItem, JobSnapshot, ResearchOptions
from app.storage.models import ResearchJobModel, utc_now
from observability.tracing import flush_traces, trace_span
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


def _paper_processing_outcome(result: dict[str, Any], options: ResearchOptions) -> tuple[str, str | None]:
    """Derive the job outcome from per-paper work, preserving the full result."""
    if not options.download_papers:
        return "completed", None
    stages = result.get("stages") or {}
    literature = stages.get("literature_database") if isinstance(stages, Mapping) else None
    if not isinstance(literature, Mapping):
        return "failed", "工作流没有返回文献处理结果。"
    downloads = [
        item for item in literature.get("download_results") or []
        if isinstance(item, Mapping)
    ]
    ingestions = [
        item for item in literature.get("ingestion_results") or []
        if isinstance(item, Mapping)
    ]
    failed_downloads = sum(item.get("success") is False for item in downloads)
    failed_ingestions = sum(item.get("success") is False for item in ingestions)
    usable = (
        sum(bool(item.get("success")) for item in ingestions if item.get("paper_id"))
        if options.enable_rag
        else sum(bool(item.get("success")) for item in downloads)
    )
    if usable == 0:
        target = "RAG 问答" if options.enable_rag else "阅读"
        detail = (
            f"下载失败 {failed_downloads} 篇，入库失败 {failed_ingestions} 篇"
            if downloads or ingestions else "没有可处理的检索结果"
        )
        return "failed", f"未获得可用于{target}的论文（{detail}）。"
    if failed_downloads or failed_ingestions:
        return (
            "partial_success",
            f"已有 {usable} 篇论文可用；下载失败 {failed_downloads} 篇，"
            f"入库失败 {failed_ingestions} 篇。请查看论文处理明细。",
        )
    return "completed", None


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
                job_type="research",
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

    def list_recent(self, limit: int = 20) -> list[JobListItem]:
        # Keep large result/log JSON out of MySQL's ORDER BY and the list payload.
        with self.database.get_session() as session:
            rows = session.execute(
                select(
                    ResearchJobModel.job_id,
                    ResearchJobModel.job_type,
                    ResearchJobModel.query,
                    ResearchJobModel.status,
                    ResearchJobModel.stage,
                    ResearchJobModel.progress,
                    ResearchJobModel.error,
                    ResearchJobModel.retryable,
                    ResearchJobModel.created_at,
                    ResearchJobModel.updated_at,
                    ResearchJobModel.started_at,
                    ResearchJobModel.finished_at,
                )
                .order_by(ResearchJobModel.created_at.desc(), ResearchJobModel.job_id.desc())
                .limit(limit)
            ).all()
            return [JobListItem(
                job_id=row.job_id,
                job_type=row.job_type,
                query=row.query,
                status=row.status,
                stage=row.stage,
                progress=row.progress,
                error=row.error,
                retryable=row.retryable,
                created_at=row.created_at,
                updated_at=row.updated_at,
                elapsed_seconds=max(0.0, ((row.finished_at or utc_now()) - (row.started_at or row.created_at)).total_seconds()),
            ) for row in rows]

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
                .where(ResearchJobModel.status.in_(tuple(_ACTIVE)), ResearchJobModel.job_type == "research")
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
            job_type=row.job_type,
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
            with trace_span("research.job", input={"job_id": job_id}) as span:
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
                span.update(output={"job_id": job_id,
                                    "stage_count": len(result.get("stages") or {})})
            if event.is_set():
                raise WorkflowCancelled("用户已取消本次检索。")
            status, message = _paper_processing_outcome(result, options)
            result = {**result, "status": status}
            if message:
                logger.warning("论文处理结果：%s", message)
            self.store.update(
                job_id, status=status, stage=status, progress=100, result=result,
                error=message, retryable=status != "completed", finished_at=utc_now(),
            )
        except WorkflowCancelled:
            self.store.update(job_id, status="cancelled", stage="cancelled", error=None, retryable=True, finished_at=utc_now())
        except Exception as exc:
            logger.exception("研究任务失败：%s", type(exc).__name__)
            self.store.update(job_id, status="failed", stage="failed", error=_safe_message(f"{type(exc).__name__}: {exc}"), retryable=True, finished_at=utc_now())
        finally:
            flush_traces()
            logger.removeHandler(handler)
            with self._lock:
                self._events.pop(job_id, None)
                self._threads.pop(job_id, None)
