"""FastAPI entry point for the durable M2 research workflow."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from hashlib import sha256
from uuid import uuid4
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from sqlalchemy import select, text

from app.api.schemas import (
    IngestionRequest, JobListItem, JobSnapshot, MultiPaperQuestionRequest,
    MultiPaperQuestionResponse, PaperSummary, QuestionRequest, QuestionResponse,
    ResearchOptions, ConversationCreate, ConversationMessageRequest,
    ConversationSummary, ConversationDetail, CollectionNameRequest, CollectionPaperRequest,
)
from app.config import ModelSettings
from app.services.jobs import JobBusyError, ResearchJobRunner
from app.services.ingestions import IngestionConflictError, IngestionJobService
from app.services.questions import QuestionService, PaperIndexUnavailableError
from app.services.conversations import ConversationService, ConversationBusyError
from app.services.library import PaperLibraryService, LibraryConflictError
from rag_reader.errors import RAGError
from rag_reader.config import RAGSettings
from rag_reader.store_factory import milvus_index_scope
from literature_database.database import DatabaseManager
from literature_database.models import PaperModel, PaperFileModel
import app.storage.models  # noqa: F401 - register M2 ORM tables


def create_app(
    *,
    database_url: str | None = None,
    data_root: str | Path | None = None,
    workflow_factory: Callable[[ResearchOptions], Any] | None = None,
    question_service: Any | None = None,
    ingestion_dispatcher: Callable[[str], None] | None = None,
) -> FastAPI:
    root = Path(data_root or os.getenv("AUTO_PAPER_DATA_ROOT", "data/m2")).resolve()
    if database_url is None:
        database_url = os.getenv("DATABASE_URL") or f"sqlite:///{(root / 'app.sqlite3').as_posix()}"
    database = DatabaseManager(database_url)
    runner = ResearchJobRunner(database, root, workflow_factory=workflow_factory)
    ingestions = IngestionJobService(database, root, dispatch=ingestion_dispatcher)
    questions = question_service or QuestionService(database, root)
    conversations = ConversationService(database, questions)
    library = PaperLibraryService(database, root, ingestions)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        root.mkdir(parents=True, exist_ok=True)
        try:
            with database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            runner.store.mark_interrupted()
        except Exception as exc:
            raise RuntimeError("M2 database is unavailable or not migrated; run `python -m alembic upgrade head`") from exc
        yield
        database.dispose()

    app = FastAPI(title="ScholarTrace Research API", version="4.2.0", lifespan=lifespan)
    app.state.database = database
    app.state.runner = runner
    app.state.ingestions = ingestions
    app.state.questions = questions
    app.state.conversations = conversations
    app.state.library = library
    app.state.data_root = root

    @app.get("/api/v1/health")
    def health() -> dict[str, bool]:
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "connected": True,
            "database": True,
            "deepseek": bool(ModelSettings.from_env().api_key.get_secret_value()),
            "serpapi": bool(os.getenv("SERPAPI_API_KEY")),
            "openalex": bool(os.getenv("OPENALEX_API_KEY")),
        }

    @app.post("/api/v1/jobs", response_model=JobSnapshot, status_code=202)
    def start_job(options: ResearchOptions) -> JobSnapshot:
        try:
            return runner.start(options)
        except JobBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/jobs", response_model=list[JobListItem])
    def list_jobs(limit: int = 20) -> list[JobListItem]:
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        return runner.store.list_recent(limit)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobSnapshot)
    def get_job(job_id: str) -> JobSnapshot:
        try:
            return runner.store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobSnapshot)
    def cancel_job(job_id: str) -> JobSnapshot:
        try:
            snapshot = runner.store.get(job_id)
            return ingestions.cancel(job_id) if snapshot.job_type == "ingestion" else runner.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/v1/ingestions", response_model=JobSnapshot, status_code=202)
    def start_ingestion(request: IngestionRequest) -> JobSnapshot:
        try:
            return ingestions.create(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="discovery job not found") from exc
        except IngestionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/ingestions/{job_id}/retry", response_model=JobSnapshot, status_code=202)
    def retry_ingestion(job_id: str) -> JobSnapshot:
        try:
            return ingestions.retry(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="ingestion job not found") from exc
        except IngestionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/papers", response_model=list[PaperSummary])
    def list_papers(limit: int = 100) -> list[PaperSummary]:
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        settings = RAGSettings.from_env()
        scope = milvus_index_scope(settings) + ":" if settings.vector_backend == "milvus" else None
        with database.get_session() as session:
            rows = session.scalars(select(PaperModel).order_by(PaperModel.rank_score.desc()).limit(limit)).all()
            files = session.scalars(select(PaperFileModel).where(PaperFileModel.file_type == "pdf")).all()
            pdf_ids = {item.paper_id for item in files if item.status.casefold() != "mismatch" and Path(item.file_path).is_file()}
            return [PaperSummary(
                paper_id=row.paper_id,
                title=row.title,
                authors=row.authors or [],
                year=row.year,
                availability=row.availability,
                source="uploaded" if row.paper_id.startswith("upload:") else "discovered",
                has_pdf=row.paper_id in pdf_ids,
                embedding_status=(
                    "REINDEX_REQUIRED"
                    if scope and row.embedding_status.casefold() == "completed"
                    and not (row.index_signature or "").startswith(scope)
                    else row.embedding_status
                ),
            ) for row in rows]

    @app.post("/api/v1/papers/upload", status_code=202)
    async def upload_paper(file: UploadFile = File(...), title: str | None = Form(default=None),
                           collection_id: str | None = Form(default=None),
                           paper_id: str | None = Form(default=None)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(422, "请选择 PDF 文件。")
        chosen_title = (title or Path(file.filename).stem).strip()
        if not 1 <= len(chosen_title) <= 500:
            raise HTTPException(422, "论文标题长度须为 1–500 个字符。")
        upload_dir = root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary = upload_dir / f".{uuid4().hex}.part"
        digest = sha256()
        total = 0
        try:
            with temporary.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > 50 * 1024 * 1024:
                        raise HTTPException(413, "PDF 不能超过 50 MB。")
                    digest.update(chunk)
                    output.write(chunk)
            if total < 8:
                raise HTTPException(422, "PDF 文件为空或损坏。")
            with temporary.open("rb") as input_file:
                if input_file.read(5) != b"%PDF-":
                    raise HTTPException(422, "文件内容不是 PDF。")
            import fitz
            try:
                with fitz.open(temporary) as document:
                    if document.is_encrypted or document.page_count < 1:
                        raise ValueError("encrypted or empty PDF")
            except Exception as exc:
                raise HTTPException(422, "PDF 无法读取、已加密或没有页面。") from exc
            try:
                paper_id, job_id = library.register_upload(temporary, digest.hexdigest(),
                                                            chosen_title, collection_id,
                                                            paper_id.strip() if paper_id else None)
            except KeyError as exc:
                raise HTTPException(404, "论文集合不存在。") from exc
            except LibraryConflictError as exc:
                raise HTTPException(409, str(exc)) from exc
            return {"paper_id": paper_id, "ingestion_job_id": job_id,
                    "status": "queued" if job_id else "already_indexed"}
        finally:
            temporary.unlink(missing_ok=True)
            await file.close()

    @app.get("/api/v1/papers/{paper_id}/pdf")
    def get_paper_pdf(paper_id: str) -> FileResponse:
        try:
            return FileResponse(library.pdf_path(paper_id), media_type="application/pdf",
                                filename="paper.pdf", content_disposition_type="inline")
        except KeyError as exc:
            raise HTTPException(404, "论文不存在。") from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "本地 PDF 不可用。") from exc

    @app.post("/api/v1/papers/{paper_id}/reindex", status_code=202)
    def reindex_paper(paper_id: str) -> dict[str, str]:
        try:
            return {"ingestion_job_id": library.reindex(paper_id)}
        except KeyError as exc:
            raise HTTPException(404, "论文不存在。") from exc
        except LibraryConflictError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/v1/papers/{paper_id}", status_code=204)
    def delete_paper(paper_id: str) -> Response:
        try:
            library.delete_uploaded(paper_id)
            return Response(status_code=204)
        except KeyError as exc:
            raise HTTPException(404, "论文不存在。") from exc
        except LibraryConflictError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/collections")
    def list_collections() -> list[dict[str, Any]]:
        return library.collections()

    @app.post("/api/v1/collections", status_code=201)
    def create_collection(body: CollectionNameRequest) -> dict[str, Any]:
        return library.create_collection(body.name)

    @app.get("/api/v1/collections/{collection_id}")
    def get_collection(collection_id: str) -> dict[str, Any]:
        try:
            return library.get_collection(collection_id)
        except KeyError as exc:
            raise HTTPException(404, "论文集合不存在。") from exc

    @app.patch("/api/v1/collections/{collection_id}")
    def rename_collection(collection_id: str, body: CollectionNameRequest) -> dict[str, Any]:
        try:
            return library.rename_collection(collection_id, body.name)
        except KeyError as exc:
            raise HTTPException(404, "论文集合不存在。") from exc

    @app.delete("/api/v1/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: str) -> Response:
        try:
            library.delete_collection(collection_id)
            return Response(status_code=204)
        except KeyError as exc:
            raise HTTPException(404, "论文集合不存在。") from exc

    @app.post("/api/v1/collections/{collection_id}/papers")
    def add_collection_paper(collection_id: str, body: CollectionPaperRequest) -> dict[str, Any]:
        try:
            return library.add_paper(collection_id, body.paper_id)
        except KeyError as exc:
            raise HTTPException(404, "论文集合或论文不存在。") from exc

    @app.delete("/api/v1/collections/{collection_id}/papers/{paper_id}")
    def remove_collection_paper(collection_id: str, paper_id: str) -> dict[str, Any]:
        try:
            return library.remove_paper(collection_id, paper_id)
        except KeyError as exc:
            raise HTTPException(404, "论文集合不存在。") from exc

    @app.post("/api/v1/questions", response_model=QuestionResponse)
    def ask_question(request: QuestionRequest) -> QuestionResponse:
        try:
            answer_id, answer = questions.ask(request.paper_id, request.question)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="paper not found") from exc
        except PaperIndexUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return QuestionResponse(answer_id=answer_id, paper_id=request.paper_id, answer=answer)

    @app.post("/api/v1/questions/multi", response_model=MultiPaperQuestionResponse)
    def ask_multiple_questions(request: MultiPaperQuestionRequest) -> MultiPaperQuestionResponse:
        try:
            answer_id, answer = questions.ask_many(request.paper_ids, request.question)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"paper not found: {exc}") from exc
        except PaperIndexUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MultiPaperQuestionResponse(answer_id=answer_id, paper_ids=request.paper_ids, answer=answer)

    @app.get("/api/v1/answers/{answer_id}/evidence", response_model=list[dict[str, Any]])
    def get_evidence(answer_id: str) -> list[dict[str, Any]]:
        try:
            return questions.evidence(answer_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="answer not found") from exc

    @app.get("/api/v1/answers/{answer_id}/export")
    def export_answer(answer_id: str) -> Response:
        try:
            report = questions.export(answer_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="answer not found") from exc
        return Response(content=report, media_type="text/markdown; charset=utf-8")

    @app.post("/api/v1/conversations", status_code=201, response_model=ConversationSummary)
    def create_conversation(body: ConversationCreate) -> dict[str, Any]:
        try:
            return conversations.create(body.paper_ids, body.title)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="paper not found") from exc
        except PaperIndexUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/conversations", response_model=list[ConversationSummary])
    def list_conversations(paper_id: list[str] | None = Query(default=None),
                           limit: int = Query(default=20, ge=1, le=100)):
        return conversations.recent(paper_id, limit)

    @app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail)
    def get_conversation(conversation_id: str, limit: int = Query(default=50, ge=1, le=100),
                         before: int | None = Query(default=None, ge=1)):
        try:
            return conversations.get(conversation_id, limit=limit, before=before)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="conversation not found") from exc

    @app.post("/api/v1/conversations/{conversation_id}/messages/stream", response_class=StreamingResponse, responses={
        200: {"description": "meta/status/delta/done/error SSE events; only done confirms persistence",
              "content": {"text/event-stream": {"schema": {"type": "string"}}}},
        409: {"description": "Conversation busy or paper index unavailable"},
    })
    async def stream_conversation(conversation_id: str, body: ConversationMessageRequest,
                                  request: Request):
        try:
            turn = await run_in_threadpool(conversations.begin, conversation_id, body.question)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="conversation or paper not found") from exc
        except (ConversationBusyError, PaperIndexUnavailableError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        iterator = conversations.stream(turn)

        async def events():
            finished = False
            try:
                while True:
                    event = await run_in_threadpool(_next_event, iterator)
                    if event is None or await request.is_disconnected():
                        break
                    if event["event"] == "done":
                        finished = True
                    yield _sse(event)
            except Exception as exc:
                logging.getLogger(__name__).exception("Conversation stream failed: %s", turn.turn_id)
                message = str(exc) if isinstance(exc, (RAGError, ConversationBusyError)) else "生成失败，请重试。"
                yield _sse({"event": "error", "data": {
                    "turn_id": turn.turn_id, "message": message, "retryable": True,
                }})
            finally:
                # Starlette cancels response tasks on disconnect. Shield cleanup so an abandoned
                # response cannot leave the conversation locked or enter successful history.
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(iterator.close)
                    if not finished:
                        await run_in_threadpool(conversations.fail, turn, "cancelled", "连接已断开，本轮未完成。")

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    return app


def _next_event(iterator):
    return next(iterator, None)


def _sse(event):
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


app = create_app()
