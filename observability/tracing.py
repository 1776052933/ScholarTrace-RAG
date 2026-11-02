"""Optional LangSmith tracing for application spans and LangChain runs.

LangChain/LangGraph create their own child runs when LANGSMITH_TRACING is enabled.
Application spans add job boundaries and small, structured summaries around them.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from typing import Any, Iterator


logger = logging.getLogger(__name__)
_client_instance: Any | None = None


class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass


class _Span:
    def __init__(self, run: Any) -> None:
        self.run = run

    def update(self, **kwargs: Any) -> None:
        output = kwargs.get("output")
        if output is not None:
            self.run.add_outputs(output)
        metadata = {key: value for key, value in kwargs.items() if key != "output"}
        if metadata:
            self.run.add_metadata(metadata)


def _enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def _client() -> Any | None:
    global _client_instance
    if not _enabled():
        return None
    if _client_instance is None:
        try:
            from langsmith import Client
            _client_instance = Client()
        except Exception:
            logger.warning("LangSmith 初始化失败，本次业务继续运行", exc_info=True)
            return None
    return _client_instance


@contextmanager
def trace_span(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    client = _client()
    if client is None:
        yield _NoopSpan()
        return
    from langsmith import trace

    # Native LangChain model calls are LLM child runs; this application
    # boundary stays a chain run to avoid counting the model usage twice.
    run_type = "chain" if as_type in {"span", "generation"} else as_type
    inputs = kwargs.pop("input", None)
    with trace(name, run_type=run_type, inputs=inputs, metadata=kwargs, client=client) as run:
        yield _Span(run)


def flush_traces() -> None:
    client = _client()
    if client is not None:
        try:
            # LangChain's tracer and the application spans may use separate queues.
            from langchain_core.tracers.langchain import wait_for_all_tracers
            wait_for_all_tracers()
            client.flush(timeout=5)
        except Exception:
            logger.warning("LangSmith 提交失败，本次业务结果已保留", exc_info=True)
