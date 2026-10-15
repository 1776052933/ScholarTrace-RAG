"""Streamlit's HTTP-only client for the M2 API."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from app.api.schemas import ResearchOptions
from pydantic import ValidationError


@dataclass(frozen=True)
class RunOptions:
    query: str
    max_papers: int = 3
    max_search_rounds: int = 1
    download_papers: bool = True
    enable_rag: bool = True
    summarize: bool = True
    extract_knowledge: bool = True
    question: str | None = None
    use_query_llm: bool = True
    enable_citation_tracking: bool = True
    publication_date_to: str | None = None
    download_engine: str = "academic_oa"
    download_timeout: int = 30
    retry_failed_downloads: bool = False


class WorkflowRunError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.log_path = None


def _base_url() -> str:
    return os.getenv("AUTO_PAPER_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _request(method: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    try:
        with httpx.Client(base_url=_base_url(), timeout=timeout) as client:
            response = client.request(method, path, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise WorkflowRunError(str(detail or f"API 返回 HTTP {exc.response.status_code}")) from exc
    except (httpx.RequestError, ValueError) as exc:
        raise WorkflowRunError(f"无法连接研究 API：{exc}") from exc


def api_status() -> dict[str, bool]:
    try:
        status = _request("GET", "/api/v1/health", timeout=2.0)
    except WorkflowRunError:
        return {"connected": False, "database": False, "deepseek": False, "serpapi": False, "openalex": False}
    return {name: bool(status.get(name)) for name in ("connected", "database", "deepseek", "serpapi", "openalex")}


def start_workflow(options: RunOptions) -> str:
    try:
        payload = ResearchOptions.model_validate(asdict(options)).model_dump(mode="json")
    except ValidationError as exc:
        raise WorkflowRunError(str(exc)) from exc
    snapshot = _request("POST", "/api/v1/jobs", payload=payload)
    return str(snapshot["job_id"])


def poll_workflow(job_id: str) -> dict[str, Any]:
    return dict(_request("GET", f"/api/v1/jobs/{job_id}"))


def cancel_workflow(job_id: str) -> dict[str, Any]:
    return dict(_request("POST", f"/api/v1/jobs/{job_id}/cancel"))


def recent_workflows(limit: int = 10) -> list[dict[str, Any]]:
    return list(_request("GET", f"/api/v1/jobs?limit={limit}"))


def ask_paper_question(paper_id: str, question: str, *, deepseek_api_key: str | None = None) -> dict[str, Any]:
    if deepseek_api_key:
        raise WorkflowRunError("M2 的模型密钥在 API 服务端配置")
    response = _request("POST", "/api/v1/questions", payload={"paper_id": paper_id, "question": question}, timeout=240.0)
    return dict(response["answer"])
