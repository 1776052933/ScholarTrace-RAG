"""Retrieve paper passages with optional Milvus hybrid search and cross-encoder."""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from .embedding import EmbeddingModel, embedding_fingerprint
from .schema import DocumentChunk
from .vector_store import VectorStore


class PaperRetriever:
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        default_top_k: int = 5,
        retrieval_mode: str = "dense",
        candidate_k: int = 20,
        reranker: Any = None,
    ) -> None:
        if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer")
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("invalid retrieval mode")
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k
        self.retrieval_mode = retrieval_mode
        self.candidate_k = candidate_k
        self.reranker = reranker

    def _search(
        self, question: str, vector: list[float] | None, *, top_k: int,
        paper_id: str | None = None, paper_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_limit = max(top_k, self.candidate_k) if self.reranker else top_k
        if getattr(self.vector_store, "supports_hybrid", False):
            return self.vector_store.search(
                vector, top_k=candidate_limit, paper_id=paper_id,
                paper_ids=paper_ids, query_text=question,
                mode=self.retrieval_mode, candidate_k=max(candidate_limit, self.candidate_k),
            )
        if paper_ids is not None:
            merged = []
            for item in paper_ids:
                merged.extend(self.vector_store.search(vector, top_k=candidate_limit, paper_id=item))
            merged.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
            seen = set()
            unique = []
            for row in merged:
                chunk_id = row.get("chunk_id")
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    unique.append(row)
                if len(unique) >= candidate_limit:
                    break
            return unique
        return self.vector_store.search(vector, top_k=candidate_limit, paper_id=paper_id)

    def _retrieve(
        self, question: str, limit: int, *,
        paper_id: str | None = None, paper_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        if getattr(self.vector_store, "supports_hybrid", False) and self.retrieval_mode == "bm25":
            self.vector_store.prepare_bm25()
            vector = None
        else:
            vector = self.embedding_model.embed(question)
            if hasattr(self.vector_store, "ensure_embedding_compatible"):
                self.vector_store.ensure_embedding_compatible(
                    embedding_fingerprint(self.embedding_model), len(vector)
                )
        results = self._search(question, vector, top_k=limit, paper_id=paper_id, paper_ids=paper_ids)
        if self.reranker is not None:
            results = self.reranker.rerank(question, results, top_k=limit)
        return _chunks(results[:limit], fallback_paper_id=paper_id)

    def retrieve(
        self, paper_id: Optional[str], question: str, *, top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        question = _question(question)
        limit = _top_k(top_k, self.default_top_k)
        if paper_id is not None:
            paper_id = paper_id.strip()
            if not paper_id:
                raise ValueError("paper_id must not be empty")
        return self._retrieve(question, limit, paper_id=paper_id)

    def retrieve_multiple(
        self, paper_ids: Iterable[str], question: str, *, top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        question = _question(question)
        limit = _top_k(top_k, self.default_top_k)
        ids = _paper_ids(paper_ids)
        if not ids:
            return []
        return self._retrieve(question, limit, paper_ids=ids)

    def retrieve_all(self, question: str, *, top_k: Optional[int] = None) -> List[DocumentChunk]:
        return self.retrieve(None, question, top_k=top_k)

    def search(
        self, question: str, *, paper_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        return self.retrieve(paper_id, question, top_k=top_k)


def _chunks(
    results: Iterable[Mapping[str, Any]], *, fallback_paper_id: Optional[str] = None
) -> List[DocumentChunk]:
    chunks = []
    for result in results:
        metadata = dict(result.get("metadata") or {})
        paper_id = str(metadata.get("paper_id") or fallback_paper_id or "").strip()
        chunk_id = str(result.get("chunk_id") or "").strip()
        text = str(result.get("text") or "").strip()
        if not paper_id or not chunk_id or not text:
            continue
        page = metadata.get("page")
        section = str(metadata.get("section") or "")
        metadata["paper_id"] = paper_id
        metadata["retrieval_score"] = float(result.get("score") or 0.0)
        metadata["distance"] = float(result.get("distance") or 0.0)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id, paper_id=paper_id, text=text,
            page=page, section=section, metadata=metadata,
        ))
    return chunks


def _question(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("question must be a string")
    value = value.strip()
    if not value:
        raise ValueError("question must not be empty")
    return value


def _top_k(value: Optional[int], default: int) -> int:
    result = default if value is None else value
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise ValueError("top_k must be a positive integer")
    return result


def _paper_ids(values: Iterable[str]) -> List[str]:
    if isinstance(values, (str, bytes)):
        values = [str(values)]
    result, seen = [], set()
    for raw in values:
        if not isinstance(raw, str):
            raise TypeError("paper_ids must contain only strings")
        value = raw.strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
