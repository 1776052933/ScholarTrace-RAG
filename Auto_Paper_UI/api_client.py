"""Streamlit's HTTP-only client for the M2 API."""
from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

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


def start_selected_ingestion(
    discovery_job_id: str, paper_ids: list[str], *,
    download_engine: str = "academic_oa", download_timeout: int = 30,
) -> dict[str, Any]:
    return dict(_request("POST", "/api/v1/ingestions", payload={
        "discovery_job_id": discovery_job_id,
        "paper_ids": paper_ids,
        "download_engine": download_engine,
        "download_timeout": download_timeout,
    }))


def retry_selected_ingestion(job_id: str) -> dict[str, Any]:
    return dict(_request("POST", f"/api/v1/ingestions/{job_id}/retry"))


def paper_library(limit: int = 200) -> list[dict[str, Any]]:
    return list(_request("GET", f"/api/v1/papers?limit={limit}"))


def ask_multiple_papers(paper_ids: list[str], question: str) -> dict[str, Any]:
    return dict(_request("POST", "/api/v1/questions/multi", payload={
        "paper_ids": paper_ids, "question": question,
    }, timeout=240.0))


def export_answer(answer_id: str) -> str:
    try:
        with httpx.Client(base_url=_base_url(), timeout=15.0) as client:
            response = client.get(f"/api/v1/answers/{answer_id}/export")
            response.raise_for_status()
            return response.text
    except (httpx.HTTPError, ValueError) as exc:
        raise WorkflowRunError(f"无法导出问答：{exc}") from exc


def ask_paper_question(paper_id: str, question: str, *, deepseek_api_key: str | None = None) -> dict[str, Any]:
    if deepseek_api_key:
        raise WorkflowRunError("M2 的模型密钥在 API 服务端配置")
    response = _request("POST", "/api/v1/questions", payload={"paper_id": paper_id, "question": question}, timeout=240.0)
    return dict(response["answer"])


def create_conversation(paper_ids: list[str]) -> dict[str, Any]:
    return dict(_request("POST", "/api/v1/conversations", payload={"paper_ids": paper_ids}))


def recent_conversations(paper_ids: list[str]) -> list[dict[str, Any]]:
    query = urlencode([("paper_id", value) for value in paper_ids])
    return list(_request("GET", f"/api/v1/conversations?{query}"))


def get_conversation(conversation_id: str, *, before: int | None = None) -> dict[str, Any]:
    suffix = f"?before={before}" if before is not None else ""
    return dict(_request("GET", f"/api/v1/conversations/{conversation_id}{suffix}"))


def stream_conversation(conversation_id: str, question: str):
    """Consume POST SSE as it arrives; a missing final done event is a failed exchange."""
    done = False
    try:
        timeout = httpx.Timeout(240.0, connect=10.0)
        with httpx.Client(base_url=_base_url(), timeout=timeout) as client:
            with client.stream("POST", f"/api/v1/conversations/{conversation_id}/messages/stream",
                               json={"question": question}, headers={"Accept": "text/event-stream"}) as response:
                if response.is_error:
                    response.read()
                    try:
                        detail = response.json().get("detail")
                    except ValueError:
                        detail = None
                    raise WorkflowRunError(str(detail or f"API 返回 HTTP {response.status_code}"))
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise WorkflowRunError("API 未返回有效的问答事件流。")
                event, data = "message", []
                for line in response.iter_lines():
                    if line.startswith(":"):
                        continue
                    if not line:
                        if data:
                            payload = json.loads("\n".join(data))
                            if not isinstance(payload, dict):
                                raise ValueError("invalid SSE event payload")
                            if event == "error":
                                raise WorkflowRunError(str(payload.get("message") or "回答生成失败"))
                            done = event == "done"
                            yield {"event": event, "data": payload}
                            if done:
                                return
                        event, data = "message", []
                    elif line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
        if not done:
            raise WorkflowRunError("回答连接提前结束，本轮未确认完成，请刷新会话后重试。")
    except (httpx.HTTPError, ValueError) as exc:
        raise WorkflowRunError(f"回答流连接失败：{exc}") from exc
