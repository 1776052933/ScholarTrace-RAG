"""Langfuse tracing with a zero-configuration local fallback."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from typing import Any, Iterator


logger = logging.getLogger(__name__)


class _NoopSpan:
    def update(self, **kwargs: Any) -> None:
        pass


def _client() -> Any | None:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception:
        logger.warning("Langfuse 初始化失败，本次调用继续运行", exc_info=True)
        return None


@contextmanager
def trace_span(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    client = _client()
    if client is None:
        yield _NoopSpan()
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as span:
        yield span


def flush_traces() -> None:
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            logger.warning("Langfuse 提交失败，本次业务结果已保留", exc_info=True)
