"""Validated RAG model, splitting, retrieval and provider settings."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RAGSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str = Field(default="BAAI/bge-m3", min_length=1)
    embedding_device: str | None = None
    embedding_batch_size: int = Field(default=32, ge=1)
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1)
    collection_name: str = Field(default="paper_chunks_m1", min_length=1)
    llm_model: str = Field(default="deepseek-chat", min_length=1)
    llm_base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def valid_overlap(self) -> "RAGSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @classmethod
    def from_env(cls) -> "RAGSettings":
        mapping = {
            "embedding_model": "RAG_EMBEDDING_MODEL",
            "embedding_device": "RAG_EMBEDDING_DEVICE",
            "embedding_batch_size": "RAG_EMBEDDING_BATCH_SIZE",
            "chunk_size": "RAG_CHUNK_SIZE",
            "chunk_overlap": "RAG_CHUNK_OVERLAP",
            "retrieval_top_k": "RAG_TOP_K",
            "collection_name": "RAG_COLLECTION_NAME",
            "llm_model": "DEEPSEEK_MODEL",
            "llm_base_url": "DEEPSEEK_BASE_URL",
            "llm_timeout_seconds": "DEEPSEEK_TIMEOUT",
        }
        values: dict[str, Any] = {}
        for field, name in mapping.items():
            value = os.getenv(name)
            if value is not None and value.strip():
                values[field] = value.strip()
        return cls.model_validate(values)
