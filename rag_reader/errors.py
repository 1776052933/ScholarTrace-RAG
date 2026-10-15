"""Stable error categories shared by CLI and future API/MCP adapters."""

from __future__ import annotations

from typing import Any


class RAGError(RuntimeError):
    code = "rag_error"
    retryable = False


class RAGDocumentError(RAGError):
    code = "document_error"


class RAGModelError(RAGError):
    code = "model_error"


class RAGRetrievalError(RAGError):
    code = "retrieval_error"


class IndexCompatibilityError(RAGRetrievalError):
    code = "index_incompatible"

class RAGGenerationError(RAGError):
    code = "generation_error"
    retryable = True


class RAGNotFoundError(RAGError):
    code = "not_found"


def error_info(exc: Exception) -> dict[str, Any]:
    """Turn a known domain failure into a stable response field set."""
    if isinstance(exc, RAGError):
        return {"code": exc.code, "type": type(exc).__name__,
                "message": str(exc), "retryable": exc.retryable}
    return {"code": "internal_error", "type": type(exc).__name__,
            "message": str(exc), "retryable": False}
