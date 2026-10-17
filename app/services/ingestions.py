"""Durable selected-paper ingestion executed by a Celery worker.

MySQL owns the job and per-paper state. Redis only transports a job ID, so a
redelivered message cannot create duplicate work or lose a completed paper.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
import json
import logging
from pathlib import Path
from threading import Event, Thread
from typing import Any, Callable

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.api.schemas import IngestionRequest, JobSnapshot
from app.services.jobs import JobStore, _safe_message
from app.storage.models import ResearchJobModel, utc_now
from literature_database.database import DatabaseManager


logger = logging.getLogger(__name__)
MAX_PAPER_ATTEMPTS = 3


class IngestionConflictError(ValueError):
    """The selected papers do not belong to a completed discovery job."""


class LostIngestionLease(RuntimeError):
    """Another worker now owns this redelivered job."""


class IngestionJobService:
    def __init__(
        self,
        database: DatabaseManager,
        data_root: Path,
        *,
        dispatch: Callable[[str], None] | None = None,
        literature_factory: Callable[[], Any] | None = None,
        reader_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.database = database
        self.data_root = Path(data_root)
        self.store = JobStore(database)
        self.dispatch = dispatch or self._dispatch_celery
        self.literature_factory = literature_factory or self._literature_service
        self.reader_factory = reader_factory or self._rag_reader
        self._lease_started_at = None

    @staticmethod
    def _dispatch_celery(job_id: str) -> None:
        from app.tasks import run_ingestion
        run_ingestion.apply_async(args=[job_id], task_id=f"ingest-{job_id}")

    def _literature_service(self) -> Any:
        from literature_database.service import LiteratureService
        return LiteratureService(self.database, create_tables=False)

    def _rag_reader(self) -> Any:
        from rag_reader.config import RAGSettings
        from rag_reader.pipeline import RAGReader
        from rag_reader.store_factory import create_vector_store
        settings = RAGSettings.from_env()
        store = create_vector_store(settings, chroma_directory=str(self.data_root / "chroma"))
        return RAGReader(vector_store=store, settings=settings)

    @staticmethod
    def _candidates(source: ResearchJobModel) -> dict[str, dict[str, Any]]:
        if source.job_type != "research" or source.status != "completed" or not isinstance(source.result, dict):
            raise IngestionConflictError("请先完成候选论文检索任务。")
        ranking = (source.result.get("stages") or {}).get("ranking_agent") or {}
        papers = (ranking.get("result") or {}).get("papers") or []
        return {str(item["paper_id"]): item for item in papers
                if isinstance(item, dict) and item.get("paper_id")}

    def create(self, request: IngestionRequest) -> JobSnapshot:
        with self.database.get_session() as session:
            source = session.get(ResearchJobModel, request.discovery_job_id)
            if source is None:
                raise KeyError(request.discovery_job_id)
            candidates = self._candidates(source)
            if any(paper_id not in candidates for paper_id in request.paper_ids):
                raise IngestionConflictError("选择的论文不属于该检索任务的候选集。")
            rank = {paper_id: index for index, paper_id in enumerate(candidates)}
            selected = sorted(request.paper_ids, key=rank.__getitem__)
            identity = json.dumps({
                "discovery_job_id": source.job_id,
                "paper_ids": sorted(selected),
                "download_engine": request.download_engine,
                "download_timeout": request.download_timeout,
            }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            job_id = sha256(identity.encode("utf-8")).hexdigest()[:32]
            existing = session.get(ResearchJobModel, job_id)
            if existing is None:
                row = ResearchJobModel(
                    job_id=job_id, job_type="ingestion", query=source.query,
                    options={"discovery_job_id": source.job_id, "paper_ids": selected,
                             "download_engine": request.download_engine,
                             "download_timeout": request.download_timeout},
                    status="queued", stage="queued", progress=0, logs=[], retryable=True,
                    result={"papers": [
                        {"paper_id": paper_id, "title": str(candidates[paper_id].get("title") or paper_id),
                         "status": "queued", "attempts": 0, "error": None, "chunk_count": 0}
                        for paper_id in selected]},
                )
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
            # The deterministic key makes repeated selection requests idempotent.
        snapshot = self.store.get(job_id)
        if snapshot.status == "queued":
            self._try_dispatch(job_id)
        return self.store.get(job_id)

    def _try_dispatch(self, job_id: str) -> None:
        try:
            self.dispatch(job_id)
        except Exception as exc:
            logger.warning("Ingestion dispatch unavailable for %s: %s", job_id, type(exc).__name__)
            self.store.update(job_id, error="队列暂不可用；任务已保存，恢复后会自动补发。", retryable=True)

    def retry(self, job_id: str) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None or row.job_type != "ingestion":
                raise KeyError(job_id)
            if row.status == "queued":
                dispatch = True
            elif row.status in {"failed", "partial_success", "interrupted", "cancelled"} and row.retryable:
                result = deepcopy(row.result or {})
                changed = False
                for item in result.get("papers") or []:
                    if item["status"] != "completed" and item.get("attempts", 0) < MAX_PAPER_ATTEMPTS:
                        item.update(status="queued", error=None)
                        changed = True
                if not changed:
                    raise IngestionConflictError("没有可重试的论文。")
                row.result = result
                row.status = "queued"
                row.stage = "queued"
                row.error = None
                row.finished_at = None
                row.updated_at = utc_now()
                session.commit()
                dispatch = True
            else:
                raise IngestionConflictError("任务尚未失败，或已达到重试次数。")
        if dispatch:
            self._try_dispatch(job_id)
        return self.store.get(job_id)

    def cancel(self, job_id: str) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None or row.job_type != "ingestion":
                raise KeyError(job_id)
            if row.status == "queued":
                row.status, row.stage, row.finished_at = "cancelled", "cancelled", utc_now()
            elif row.status == "running":
                row.status = "cancelling"
            row.updated_at = utc_now()
            session.commit()
        return self.store.get(job_id)

    def _update(self, job_id: str, mutate: Callable[[ResearchJobModel, dict[str, Any]], None]) -> JobSnapshot:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None:
                raise KeyError(job_id)
            if self._lease_started_at is not None and (
                row.started_at != self._lease_started_at or row.status not in {"running", "cancelling"}
            ):
                raise LostIngestionLease(job_id)
            result = deepcopy(row.result or {})
            mutate(row, result)
            row.result = result
            row.updated_at = utc_now()
            session.commit()
        return self.store.get(job_id)

    def _heartbeat(self, job_id: str, stop: Event) -> None:
        while not stop.wait(10):
            try:
                with self.database.get_session() as session:
                    session.execute(update(ResearchJobModel).where(
                        ResearchJobModel.job_id == job_id,
                        ResearchJobModel.started_at == self._lease_started_at,
                        ResearchJobModel.status.in_(("running", "cancelling")),
                    ).values(updated_at=utc_now()))
                    session.commit()
            except Exception:
                logger.exception("Ingestion heartbeat failed for %s", job_id)

    def execute(self, job_id: str) -> JobSnapshot:
        # MySQL DATETIME columns do not retain fractional seconds by default.
        # Use the stored precision for the worker lease and heartbeat predicate.
        claimed_at = utc_now().replace(microsecond=0)
        with self.database.get_session() as session:
            claimed = session.execute(update(ResearchJobModel).where(
                ResearchJobModel.job_id == job_id,
                ResearchJobModel.job_type == "ingestion",
                ResearchJobModel.status == "queued",
            ).values(status="running", stage="starting", started_at=claimed_at, updated_at=claimed_at))
            session.commit()
            if claimed.rowcount != 1:
                return self.store.get(job_id)
        self._lease_started_at = claimed_at
        stop = Event()
        heartbeat = Thread(target=self._heartbeat, args=(job_id, stop), daemon=True)
        heartbeat.start()
        try:
            options = self._options(job_id)
            source = self._source_candidates(options["discovery_job_id"])
            literature = self.literature_factory()
            reader = self.reader_factory()
            papers_dir = self.data_root / "papers"
            text_dir = papers_dir / "text"
            papers_dir.mkdir(parents=True, exist_ok=True)
            text_dir.mkdir(parents=True, exist_ok=True)
            for paper_id in options["paper_ids"]:
                if self.store.get(job_id).status == "cancelling":
                    break
                current = next(item for item in (self.store.get(job_id).result or {})["papers"]
                               if item["paper_id"] == paper_id)
                if current["status"] == "completed" or current.get("attempts", 0) >= MAX_PAPER_ATTEMPTS:
                    continue
                self._update(job_id, lambda row, result, pid=paper_id: self._set_paper(
                    row, result, pid, status="running", stage="download",
                    attempts_delta=1, error=None,
                ))
                try:
                    resources = literature.prepare_for_reader(
                        {"papers": [source[paper_id]]}, target_count=1,
                        output_dir=str(papers_dir),
                        engine=options["download_engine"],
                        timeout=options["download_timeout"], retry_failed=True,
                    )
                    if not resources:
                        failed = [str(item.get("error") or "PDF 不可用")
                                  for item in literature.last_download_results if not item.get("success")]
                        raise RuntimeError("下载失败：" + "; ".join(failed or ["未获得可用 PDF"]))
                    self._update(job_id, lambda row, result, pid=paper_id: self._set_paper(
                        row, result, pid, status="running", stage="ingest",
                    ))
                    outcome = literature.ingest_into_reader(resources, reader, text_output_dir=str(text_dir))
                    first = next((item for item in outcome if item.get("paper_id") == paper_id), None)
                    if not first or not first.get("success"):
                        raise RuntimeError(str((first or {}).get("error") or "解析或向量入库失败"))
                    self._update(job_id, lambda row, result, pid=paper_id, item=first: self._set_paper(
                        row, result, pid, status="completed", stage="ingest",
                        error=None, chunk_count=int(item.get("chunk_count") or 0),
                    ))
                except Exception as exc:
                    logger.exception("Selected-paper ingestion failed: job=%s paper=%s", job_id, paper_id)
                    self._update(job_id, lambda row, result, pid=paper_id, message=exc: self._set_paper(
                        row, result, pid, status="failed", stage="failed",
                        error=_safe_message(f"{type(message).__name__}: {message}"),
                    ))
            return self._finish(job_id)
        except LostIngestionLease:
            return self.store.get(job_id)
        except Exception as exc:
            logger.exception("Ingestion job failed: %s", job_id)
            return self._update(job_id, lambda row, result: self._fail_job(row, result, exc))
        finally:
            stop.set()
            heartbeat.join(timeout=2)

    def _options(self, job_id: str) -> dict[str, Any]:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            if row is None:
                raise KeyError(job_id)
            return dict(row.options)

    def _source_candidates(self, source_id: str) -> dict[str, dict[str, Any]]:
        with self.database.get_session() as session:
            row = session.get(ResearchJobModel, source_id)
            if row is None:
                raise KeyError(source_id)
            return self._candidates(row)

    @staticmethod
    def _set_paper(row: ResearchJobModel, result: dict[str, Any], paper_id: str, *,
                   status: str, stage: str, attempts_delta: int = 0,
                   error: str | None = None, chunk_count: int | None = None) -> None:
        item = next(entry for entry in result["papers"] if entry["paper_id"] == paper_id)
        item["status"] = status
        item["attempts"] += attempts_delta
        item["error"] = error
        if chunk_count is not None:
            item["chunk_count"] = chunk_count
        row.stage = stage
        completed = sum(entry["status"] in {"completed", "failed"} for entry in result["papers"])
        row.progress = min(99, int(100 * completed / len(result["papers"])))

    @staticmethod
    def _fail_job(row: ResearchJobModel, result: dict[str, Any], exc: Exception) -> None:
        row.status = "failed"
        row.stage = "failed"
        row.error = _safe_message(f"{type(exc).__name__}: {exc}")
        row.retryable = True
        row.finished_at = utc_now()
        for item in result.get("papers") or []:
            if item["status"] == "running":
                item.update(status="failed", error=row.error)

    def _finish(self, job_id: str) -> JobSnapshot:
        def finish(row: ResearchJobModel, result: dict[str, Any]) -> None:
            papers = result["papers"]
            if row.status == "cancelling":
                row.status = row.stage = "cancelled"
            else:
                done = sum(item["status"] == "completed" for item in papers)
                row.status = "completed" if done == len(papers) else "partial_success" if done else "failed"
                row.stage = row.status
            row.progress = 100
            row.retryable = any(item["status"] != "completed" and item["attempts"] < MAX_PAPER_ATTEMPTS
                                for item in papers)
            row.error = None if row.status == "completed" else "部分论文未入库，请查看逐篇状态。"
            row.finished_at = utc_now()
        return self._update(job_id, finish)

    def reconcile(self, *, stale_after_seconds: int = 120) -> list[str]:
        """Requeue lost workers and redispatch durable queued jobs."""
        cutoff = utc_now() - timedelta(seconds=stale_after_seconds)
        pending = []
        with self.database.get_session() as session:
            rows = session.scalars(select(ResearchJobModel).where(
                ResearchJobModel.job_type == "ingestion",
                ResearchJobModel.status.in_(("queued", "running", "cancelling")),
            ).limit(100)).all()
            for row in rows:
                if row.status in {"running", "cancelling"} and row.updated_at < cutoff:
                    if row.status == "cancelling":
                        row.status = row.stage = "cancelled"
                        row.finished_at = utc_now()
                        continue
                    result = deepcopy(row.result or {})
                    for item in result.get("papers") or []:
                        if item["status"] == "running":
                            item.update(status="queued", error="工作进程中断，等待恢复。")
                    row.result = result
                    row.status, row.stage = "queued", "queued"
                    row.error = "工作进程中断，正在恢复。"
                    row.updated_at = utc_now()
                if row.status == "queued":
                    pending.append(row.job_id)
            session.commit()
        for job_id in pending:
            self._try_dispatch(job_id)
        return pending
