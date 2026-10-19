"""Scoped LangChain retrieval with lossless paper provenance conversion."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, model_validator

from .schema import DocumentChunk


def to_document(chunk: DocumentChunk) -> Document:
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata={
        **chunk.metadata,
        "paper_id": chunk.paper_id, "chunk_id": chunk.chunk_id,
        "page": chunk.page, "section": chunk.section,
        "chunk_metadata": dict(chunk.metadata),
    })


def to_chunk(document: Document) -> DocumentChunk:
    meta = document.metadata
    return DocumentChunk(
        chunk_id=meta["chunk_id"], paper_id=meta["paper_id"], text=document.page_content,
        page=meta.get("page"), section=meta.get("section", ""),
        metadata=dict(meta.get("chunk_metadata", {})),
    )


class ScopedPaperRetriever(BaseRetriever):
    backend: Any = Field(exclude=True, repr=False)
    paper_ids: tuple[str, ...]
    top_k: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_scope(self):
        if not self.paper_ids or any(not p.strip() for p in self.paper_ids):
            raise ValueError("retrieval requires a non-empty paper scope")
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("retrieval scope contains duplicate papers")
        return self

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        chunks = (
            self.backend.retrieve(self.paper_ids[0], query, top_k=self.top_k)
            if len(self.paper_ids) == 1 else
            self.backend.retrieve_multiple(list(self.paper_ids), query, top_k=self.top_k)
        )
        # A backend bug must never leak evidence outside the user's selected scope.
        if any(chunk.paper_id not in self.paper_ids for chunk in chunks):
            raise ValueError("retriever returned evidence outside the selected paper scope")
        return [to_document(chunk) for chunk in chunks]
