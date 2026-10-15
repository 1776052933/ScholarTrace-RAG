"""Construct the explicitly selected RAG index backend."""
from __future__ import annotations

from .config import RAGSettings
from .vector_store import VectorStore


def milvus_pipeline_signature(settings: RAGSettings) -> str:
    return (
        f"page-section-{settings.chunk_size}-{settings.chunk_overlap}-"
        f"{settings.pdf_parser_backend}-v1"
    )


def milvus_index_scope(settings: RAGSettings) -> str:
    return f"milvus:{settings.collection_name}:{milvus_pipeline_signature(settings)}"


def create_vector_store(
    settings: RAGSettings, *, chroma_directory: str | None = None
):
    if settings.vector_backend == "milvus":
        from .milvus_store import MilvusVectorStore

        return MilvusVectorStore(
            collection_name=settings.collection_name,
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            pipeline_signature=milvus_pipeline_signature(settings),
        )
    if settings.vector_backend == "chroma":
        return VectorStore(
            collection_name=settings.collection_name,
            persist_directory=chroma_directory,
        )
    raise ValueError(f"unsupported vector backend: {settings.vector_backend}")
