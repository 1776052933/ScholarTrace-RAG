"""Celery transport for durable selected-paper ingestion jobs."""
from __future__ import annotations

import os
from pathlib import Path

from celery import Celery

from app.services.ingestions import IngestionJobService
from literature_database.database import DatabaseManager
from observability.tracing import flush_traces, trace_span


broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("scholartrace", broker=broker_url)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_ignore_result=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 10800},
    beat_schedule={
        "reconcile-durable-ingestions": {
            "task": "scholartrace.reconcile_ingestions", "schedule": 60.0,
        },
    },
)


def _service() -> IngestionJobService:
    root = Path(os.getenv("AUTO_PAPER_DATA_ROOT", "data/m2")).resolve()
    database_url = os.getenv("DATABASE_URL") or f"sqlite:///{(root / 'app.sqlite3').as_posix()}"
    database = DatabaseManager(database_url)
    return IngestionJobService(database, root)


@celery_app.task(name="scholartrace.run_ingestion")
def run_ingestion(job_id: str) -> str:
    service = _service()
    try:
        with trace_span("ingestion.job", input={"job_id": job_id}) as span:
            status = service.execute(job_id).status
            span.update(output={"job_id": job_id, "status": status})
            return status
    finally:
        service.database.dispose()
        flush_traces()


@celery_app.task(name="scholartrace.reconcile_ingestions")
def reconcile_ingestions() -> list[str]:
    service = _service()
    try:
        return service.reconcile()
    finally:
        service.database.dispose()
