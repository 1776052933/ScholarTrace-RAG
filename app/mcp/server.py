"""MCP tools backed by the same FastAPI domain services used by Streamlit."""
from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from mcp.server import MCPServer


mcp = MCPServer(
    "ScholarTrace",
    instructions="Search academic papers, ingest selected PDFs, and ask cited questions about indexed papers.",
)


def _api(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    base_url = os.getenv("SCHOLARTRACE_API_URL", "http://127.0.0.1:8000").rstrip("/")
    response = requests.request(
        method, f"{base_url}{path}", json=payload, timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.reason
        raise RuntimeError(f"ScholarTrace API {response.status_code}: {detail}")
    return response.json()


@mcp.tool()
def search_papers(query: str, max_papers: int = 3, max_search_rounds: int = 2) -> dict[str, Any]:
    """Start an academic search. Poll get_job_status for ranked candidate papers."""
    job = _api("POST", "/api/v1/jobs", {
        "query": query, "max_papers": max_papers, "max_search_rounds": max_search_rounds,
        "download_papers": False, "enable_rag": False,
        "summarize": False, "extract_knowledge": False,
    })
    return {"job_id": job["job_id"], "status": job["status"], "stage": job.get("stage")}


@mcp.tool()
def get_job_status(job_id: str) -> dict[str, Any]:
    """Get a research or ingestion job's progress and selected-paper results."""
    job = _api("GET", f"/api/v1/jobs/{job_id}")
    stages = (job.get("result") or {}).get("stages") or {}
    ranking = (stages.get("ranking_agent") or {}).get("result") or {}
    return {
        "job_id": job["job_id"], "job_type": job["job_type"],
        "status": job["status"], "stage": job.get("stage"),
        "progress": job.get("progress"), "error": job.get("error"),
        "retryable": job.get("retryable"),
        "candidates": ranking.get("papers") or [],
        "papers": (job.get("result") or {}).get("papers") or [],
    }


@mcp.tool()
def start_ingestion(discovery_job_id: str, paper_ids: list[str],
                    download_engine: str = "academic_oa") -> dict[str, Any]:
    """Download and index selected candidate papers in a durable background job."""
    job = _api("POST", "/api/v1/ingestions", {
        "discovery_job_id": discovery_job_id, "paper_ids": paper_ids,
        "download_engine": download_engine,
    })
    return {"job_id": job["job_id"], "status": job["status"],
            "papers": (job.get("result") or {}).get("papers") or []}


@mcp.tool()
def ask_papers(paper_ids: list[str], question: str) -> dict[str, Any]:
    """Ask a cited question about one to five papers already indexed in Milvus."""
    if len(paper_ids) == 1:
        result = _api("POST", "/api/v1/questions", {
            "paper_id": paper_ids[0], "question": question,
        })
    else:
        result = _api("POST", "/api/v1/questions/multi", {
            "paper_ids": paper_ids, "question": question,
        })
    return {"answer_id": result["answer_id"], "paper_ids": paper_ids,
            "answer": result["answer"]}


@mcp.tool()
def get_evidence(answer_id: str) -> list[dict[str, Any]]:
    """Return the original paper excerpts and PDF pages cited by an answer."""
    return _api("GET", f"/api/v1/answers/{answer_id}/evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"],
                        default="streamable-http")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port,
                stateless_http=True, json_response=True)


if __name__ == "__main__":
    main()
