"""Subprocess adapter for the unchanged AutoPaper command-line workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4


UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parent
RUNTIME_ROOT = UI_ROOT / "runtime"
OUTPUT_ROOT = UI_ROOT / "outputs"


@dataclass(frozen=True)
class RunOptions:
    query: str
    max_papers: int = 3
    max_search_rounds: int = 1
    download_papers: bool = True
    enable_rag: bool = True
    summarize: bool = True
    extract_knowledge: bool = True
    question: Optional[str] = None
    use_query_llm: bool = True
    enable_citation_tracking: bool = True
    publication_date_to: Optional[str] = None
    download_engine: str = "academic_oa"
    download_timeout: int = 30
    retry_failed_downloads: bool = False
    log_level: str = "INFO"
    deepseek_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None


@dataclass(frozen=True)
class WorkflowRun:
    result: dict
    output_path: Path
    log_path: Path
    command: list[str]
    logs: list[str]


class WorkflowRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        logs: list[str],
        log_path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.logs = logs
        self.log_path = log_path


@dataclass
class _WorkflowJob:
    job_id: str
    options: RunOptions
    output_path: Path
    log_path: Path
    command: list[str]
    process: subprocess.Popen[str]
    secret_values: list[str]
    started_at: datetime
    started_monotonic: float
    initial_partial_files: set[str]
    logs: list[str] = field(default_factory=list)
    status: str = "running"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    reader_thread: Optional[threading.Thread] = None
    lock: threading.RLock = field(default_factory=threading.RLock)


_WORKFLOW_JOBS: dict[str, _WorkflowJob] = {}
_WORKFLOW_JOBS_LOCK = threading.RLock()


def ask_paper_question(
    paper_id: str,
    question: str,
    *,
    deepseek_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Ask a new question against the vector index created by the latest run."""
    paper_id = paper_id.strip()
    question = question.strip()
    if not paper_id:
        raise ValueError("paper_id must not be empty")
    if not question:
        raise ValueError("question must not be empty")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from rag_reader.config import RAGSettings
    from rag_reader.llm_client import DeepSeekClient
    from rag_reader.pipeline import RAGReader
    from rag_reader.vector_store import VectorStore

    settings = RAGSettings.from_env()
    llm_client = DeepSeekClient(
        api_key=deepseek_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
    reader = RAGReader(
        vector_store=VectorStore(
            persist_directory=str(RUNTIME_ROOT / "chroma"),
        ),
        llm_client=llm_client,
        settings=settings,
    )
    answer = reader.ask(paper_id, question)
    if hasattr(answer, "model_dump"):
        return dict(answer.model_dump(mode="json"))
    if isinstance(answer, dict):
        return dict(answer)
    raise TypeError("RAG Reader returned an invalid answer")


def start_workflow(options: RunOptions) -> str:
    """Start a controllable workflow process and return its job identifier."""
    _prepare_directories()
    output_path = _new_output_path()
    log_path = output_path.with_suffix(".log")
    command = build_command(options, output_path)
    environment = _build_environment(options)
    secret_values = _secret_values(options, environment)
    initial_partial_files = {
        str(path.resolve())
        for path in (RUNTIME_ROOT / "papers").rglob("*.part")
        if path.is_file()
    }
    popen_options: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "env": environment,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_options["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        popen_options["start_new_session"] = True

    try:
        process = subprocess.Popen(command, **popen_options)
    except OSError as exc:
        message = _redact_log_line(f"无法启动研究进程：{exc}", secret_values)
        log_path.write_text(message + "\n", encoding="utf-8")
        _prune_output_history()
        raise WorkflowRunError(message, [message], log_path) from exc

    job = _WorkflowJob(
        job_id=uuid4().hex,
        options=options,
        output_path=output_path,
        log_path=log_path,
        command=command,
        process=process,
        secret_values=secret_values,
        started_at=datetime.now(),
        started_monotonic=time.monotonic(),
        initial_partial_files=initial_partial_files,
    )
    with _WORKFLOW_JOBS_LOCK:
        _WORKFLOW_JOBS[job.job_id] = job
    job.reader_thread = threading.Thread(
        target=_capture_job_output,
        args=(job,),
        name=f"autopaper-log-{job.job_id[:8]}",
        daemon=True,
    )
    job.reader_thread.start()
    return job.job_id


def poll_workflow(job_id: str) -> dict[str, Any]:
    """Return the latest state of a background workflow."""
    job = _get_workflow_job(job_id)
    if job.process.poll() is not None:
        _finalize_workflow_job(job)
    return _job_snapshot(job)


def cancel_workflow(job_id: str) -> dict[str, Any]:
    """Terminate a workflow process tree and retain a cancellation record."""
    job = _get_workflow_job(job_id)
    if job.process.poll() is not None:
        _finalize_workflow_job(job)
        return _job_snapshot(job)

    with job.lock:
        if job.status != "running":
            return _job_snapshot(job)
        job.status = "cancelling"

    _terminate_process_tree(job.process)
    if job.reader_thread is not None:
        job.reader_thread.join(timeout=5)

    cancelled_at = datetime.now()
    elapsed = max(0.0, time.monotonic() - job.started_monotonic)
    message = (
        "用户已取消本次检索。已完成并核验的 PDF 与有效缓存已保留，"
        "未完成的临时下载已清理。"
    )
    log_line = f"{cancelled_at:%Y-%m-%d %H:%M:%S} | WARNING | ui | {message}"
    sanitized = _redact_log_line(log_line, job.secret_values)
    with job.lock:
        job.logs.append(sanitized)
        job.status = "cancelled"
        job.error = None
    with job.log_path.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(sanitized + "\n")

    removed_partials = _cleanup_new_partial_downloads(job.initial_partial_files)
    cancellation_result = {
        "status": "cancelled",
        "query": job.options.query,
        "started_at": job.started_at.isoformat(timespec="seconds"),
        "cancelled_at": cancelled_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "removed_partial_downloads": removed_partials,
        "message": message,
        "stages": {},
    }
    job.output_path.write_text(
        json.dumps(cancellation_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with job.lock:
        job.result = cancellation_result
    _prune_output_history()
    return _job_snapshot(job)


def _get_workflow_job(job_id: str) -> _WorkflowJob:
    with _WORKFLOW_JOBS_LOCK:
        job = _WORKFLOW_JOBS.get(job_id)
    if job is None:
        raise KeyError(f"找不到运行任务：{job_id}")
    return job


def _capture_job_output(job: _WorkflowJob) -> None:
    suppress_json = False
    with job.log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        if job.process.stdout is None:
            return
        for raw_line in job.process.stdout:
            line = _redact_log_line(raw_line.rstrip("\r\n"), job.secret_values)
            if not suppress_json and line.strip() == "{":
                suppress_json = True
            if suppress_json or not line:
                continue
            with job.lock:
                job.logs.append(line)
            log_file.write(line + "\n")
            log_file.flush()


def _finalize_workflow_job(job: _WorkflowJob) -> None:
    with job.lock:
        if job.status not in {"running", "finalizing"}:
            return
        if job.status == "finalizing":
            return
        job.status = "finalizing"

    if job.reader_thread is not None:
        job.reader_thread.join(timeout=5)
    return_code = job.process.wait()
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    if return_code != 0:
        error = f"工作流运行失败，进程退出码为 {return_code}"
    elif not job.output_path.exists():
        error = "工作流结束，但没有生成结果文件"
    else:
        try:
            result = json.loads(job.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            error = f"无法读取结果文件：{exc}"

    with job.lock:
        job.result = result
        job.error = error
        job.status = "failed" if error else "completed"
    _prune_output_history()


def _job_snapshot(job: _WorkflowJob) -> dict[str, Any]:
    with job.lock:
        status = "running" if job.status in {"running", "finalizing"} else job.status
        return {
            "job_id": job.job_id,
            "status": status,
            "result": job.result,
            "error": job.error,
            "output_path": job.output_path,
            "log_path": job.log_path,
            "elapsed_seconds": max(0.0, time.monotonic() - job.started_monotonic),
            "logs": list(job.logs),
        }


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _cleanup_new_partial_downloads(initial_files: set[str]) -> int:
    removed = 0
    papers_root = RUNTIME_ROOT / "papers"
    if not papers_root.exists():
        return removed
    for path in papers_root.rglob("*.part"):
        if not path.is_file() or str(path.resolve()) in initial_files:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def run_workflow(
    options: RunOptions,
    on_log: Optional[Callable[[str], None]] = None,
) -> WorkflowRun:
    """Run main.py without changing its behavior and return its JSON output."""
    _prepare_directories()
    output_path = _new_output_path()
    log_path = output_path.with_suffix(".log")
    command = build_command(options, output_path)
    environment = _build_environment(options)
    secret_values = _secret_values(options, environment)

    logs: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        message = _redact_log_line(f"无法启动研究进程：{exc}", secret_values)
        log_path.write_text(message + "\n", encoding="utf-8")
        _prune_output_history()
        raise WorkflowRunError(message, [message], log_path) from exc

    suppress_json = False
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = _redact_log_line(raw_line.rstrip("\r\n"), secret_values)
            if not suppress_json and line.strip() == "{":
                suppress_json = True
            if suppress_json:
                continue
            if line:
                logs.append(line)
                log_file.write(line + "\n")
                log_file.flush()
                if on_log is not None:
                    on_log(line)

    return_code = process.wait()
    if return_code != 0:
        _prune_output_history()
        raise WorkflowRunError(
            f"工作流运行失败，进程退出码为 {return_code}",
            logs,
            log_path,
        )
    if not output_path.exists():
        _prune_output_history()
        raise WorkflowRunError("工作流结束，但没有生成结果文件", logs, log_path)

    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _prune_output_history()
        raise WorkflowRunError(f"无法读取结果文件：{exc}", logs, log_path) from exc

    _prune_output_history()
    return WorkflowRun(
        result=result,
        output_path=output_path,
        log_path=log_path,
        command=command,
        logs=logs,
    )


def build_command(options: RunOptions, output_path: Path) -> list[str]:
    database_path = (RUNTIME_ROOT / "literature.db").resolve().as_posix()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        options.query.strip(),
        "--max-papers",
        str(options.max_papers),
        "--max-search-rounds",
        str(options.max_search_rounds),
        "--papers-directory",
        str(RUNTIME_ROOT / "papers"),
        "--text-directory",
        str(RUNTIME_ROOT / "papers" / "text"),
        "--chroma-directory",
        str(RUNTIME_ROOT / "chroma"),
        "--database-url",
        f"sqlite:///{database_path}",
        "--download-engine",
        options.download_engine,
        "--download-timeout",
        str(options.download_timeout),
        "--log-level",
        options.log_level,
        "--output",
        str(output_path),
    ]

    if not options.download_papers:
        command.append("--skip-download")
    elif not options.enable_rag:
        command.append("--skip-rag")
    if not options.use_query_llm:
        command.append("--no-query-llm")
    if not options.enable_citation_tracking:
        command.append("--skip-citation-tracking")
    if options.publication_date_to:
        command.extend(["--publication-date-to", options.publication_date_to])
    if options.retry_failed_downloads:
        command.append("--retry-failed-downloads")
    if options.enable_rag and options.summarize:
        command.append("--summarize")
    if options.enable_rag and options.extract_knowledge:
        command.append("--extract-knowledge")
    if options.enable_rag and options.question:
        command.extend(["--question", options.question.strip()])
    return command


def api_status() -> dict[str, bool]:
    """Report environment-key availability without exposing secret values."""
    return {
        "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
        "serpapi": bool(os.getenv("SERPAPI_API_KEY")),
        "openalex": bool(_configured_key("openalex", "OPENALEX_API_KEY")),
    }


def _prepare_directories() -> None:
    for path in (
        OUTPUT_ROOT,
        RUNTIME_ROOT / "papers" / "text",
        RUNTIME_ROOT / "chroma",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _new_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"result_{timestamp}_{uuid4().hex[:6]}.json"


def _build_environment(options: RunOptions) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["RANKING_CACHE_PATH"] = str(RUNTIME_ROOT / "ranking_evaluations.sqlite3")
    if options.deepseek_api_key:
        environment["DEEPSEEK_API_KEY"] = options.deepseek_api_key.strip()
    if options.serpapi_api_key:
        environment["SERPAPI_API_KEY"] = options.serpapi_api_key.strip()
    openalex_api_key = _configured_key("openalex", "OPENALEX_API_KEY")
    if openalex_api_key:
        environment["OPENALEX_API_KEY"] = openalex_api_key
    return environment


def _configured_key(name: str, environment_name: str) -> str:
    value = str(os.getenv(environment_name) or "").strip()
    if value:
        return value
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from API_key.key_manager import key_get

        return str(key_get(name) or "").strip()
    except (ImportError, OSError, TypeError, ValueError):
        return ""


def _secret_values(options: RunOptions, environment: dict[str, str]) -> list[str]:
    values = [
        options.deepseek_api_key,
        options.serpapi_api_key,
        environment.get("DEEPSEEK_API_KEY"),
        environment.get("SERPAPI_API_KEY"),
        environment.get("OPENALEX_API_KEY"),
    ]
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from API_key.key_manager import key_get

        values.extend([key_get("deepseek"), key_get("serpapi"), key_get("openalex")])
    except (ImportError, OSError, TypeError, ValueError):
        pass
    return sorted(
        {str(value).strip() for value in values if value and str(value).strip()},
        key=len,
        reverse=True,
    )


def _redact_log_line(line: str, secret_values: list[str]) -> str:
    sanitized = line
    for value in secret_values:
        sanitized = sanitized.replace(value, "[REDACTED]")
    sanitized = re.sub(
        r"(?i)((?:DEEPSEEK|SERPAPI|OPENALEX)_API_KEY\s*[=:]\s*)\S+",
        r"\1[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", sanitized)
    return sanitized


def _prune_output_history(max_runs: int = 30) -> None:
    files = [
        path for path in OUTPUT_ROOT.glob("result_*.*")
        if path.is_file() and path.suffix.casefold() in {".json", ".log"}
    ]
    groups: dict[str, list[Path]] = {}
    for path in files:
        groups.setdefault(path.stem, []).append(path)
    ordered = sorted(
        groups.items(),
        key=lambda item: max(path.stat().st_mtime for path in item[1]),
        reverse=True,
    )
    for _, paths in ordered[max_runs:]:
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass
