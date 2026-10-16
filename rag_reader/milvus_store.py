"""Milvus 2.6 document index with dense, built-in BM25, and RRF search."""
from __future__ import annotations

import json
import math
from threading import RLock
from typing import Any, Iterable, Sequence

from .errors import IndexCompatibilityError
from .schema import DocumentChunk
from .vector_store import _metadata, _vector

INDEX_SCHEMA_VERSION = "scholartrace-m3-v1"
OUTPUT_FIELDS = ["paper_id", "text", "page", "section", "metadata"]


class MilvusVectorStore:
    """Keep paper chunks in one versioned Milvus collection.

    The collection description contains the embedding and parsing signature. A
    different signature is rejected before reads or writes; no implicit fallback
    to a Chroma index or a different embedding model is possible.
    """

    supports_hybrid = True

    def __init__(
        self,
        collection_name: str = "paper_chunks_m3",
        *,
        uri: str = "http://127.0.0.1:19530",
        token: str | None = None,
        pipeline_signature: str = "page-section-1000-200-pymupdf-v1",
        client: Any = None,
    ) -> None:
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name is required")
        if not pipeline_signature or not pipeline_signature.strip():
            raise ValueError("pipeline_signature is required")
        self.collection_name = collection_name.strip()
        self.pipeline_signature = pipeline_signature.strip()
        self._lock = RLock()
        self._dimension: int | None = None
        self._fingerprint: str | None = None
        self._loaded = False
        self.index_scope = f"milvus:{self.collection_name}:{self.pipeline_signature}"
        self.index_signature: str | None = None
        if client is None:
            try:
                from pymilvus import MilvusClient
            except ImportError as exc:
                raise ImportError("Milvus mode requires pymilvus>=2.6,<3.0") from exc
            arguments: dict[str, Any] = {"uri": uri}
            if token:
                arguments["token"] = token
            try:
                client = MilvusClient(**arguments)
            except Exception as exc:
                raise ConnectionError(f"cannot connect to Milvus at {uri}") from exc
        self.client = client

    def ensure_embedding_compatible(self, fingerprint: str, dimension: int) -> None:
        if not fingerprint or not fingerprint.strip() or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("fingerprint and positive dimension are required")
        signature = json.dumps({
            "schema": INDEX_SCHEMA_VERSION,
            "embedding": fingerprint.strip(),
            "dimension": dimension,
            "pipeline": self.pipeline_signature,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.client.has_collection(self.collection_name):
                actual = self.client.describe_collection(self.collection_name)
                if actual.get("description") != signature:
                    raise IndexCompatibilityError(
                        f"Milvus collection {self.collection_name} has a different index signature; "
                        "create a new collection and rebuild the papers"
                    )
            else:
                self._create_collection(dimension, signature)
            if not self._loaded:
                self.client.load_collection(self.collection_name)
                self._loaded = True
            self._fingerprint = fingerprint.strip()
            self._dimension = dimension
            self.index_signature = f"{self.index_scope}:{fingerprint.strip()}:{dimension}"

    def prepare_bm25(self) -> None:
        """Load the lexical index without running a dense embedding model."""
        with self._lock:
            if not self.client.has_collection(self.collection_name):
                raise IndexCompatibilityError(f"Milvus collection {self.collection_name} does not exist")
            raw = self.client.describe_collection(self.collection_name).get("description") or ""
            try:
                signature = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise IndexCompatibilityError("Milvus collection has no valid index signature") from exc
            if signature.get("schema") != INDEX_SCHEMA_VERSION or signature.get("pipeline") != self.pipeline_signature:
                raise IndexCompatibilityError("Milvus BM25 index uses a different parsing or chunking version")
            dimension = signature.get("dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise IndexCompatibilityError("Milvus collection has an invalid embedding dimension")
            if not self._loaded:
                self.client.load_collection(self.collection_name)
                self._loaded = True
            self._dimension = dimension

    def _create_collection(self, dimension: int, signature: str) -> None:
        try:
            from pymilvus import DataType, Function, FunctionType, MilvusClient
        except ImportError as exc:
            raise ImportError("Milvus mode requires pymilvus>=2.6,<3.0") from exc
        schema = MilvusClient.create_schema(
            auto_id=False, enable_dynamic_field=False, description=signature
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
        schema.add_field(field_name="paper_id", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=16384, enable_analyzer=True)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        ))
        indexes = self.client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
            )
        except Exception:
            # Another worker may have created it at the same time. Recheck the
            # signature instead of deleting or mutating an unknown collection.
            if not self.client.has_collection(self.collection_name):
                raise
            actual = self.client.describe_collection(self.collection_name)
            if actual.get("description") != signature:
                raise IndexCompatibilityError(
                    f"Milvus collection {self.collection_name} was created with a different signature"
                )

    def add_documents(
        self, chunks: Iterable[DocumentChunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        documents, vectors = list(chunks), list(embeddings)
        if len(documents) != len(vectors):
            raise ValueError("chunks and embeddings must have the same length")
        if not documents:
            return 0
        if self._dimension is None:
            raise IndexCompatibilityError("call ensure_embedding_compatible before indexing")
        rows = []
        seen: set[str] = set()
        for chunk, vector in zip(documents, vectors):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("chunks must contain DocumentChunk values")
            if chunk.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
            seen.add(chunk.chunk_id)
            checked = _vector(vector, "embedding")
            if len(checked) != self._dimension:
                raise IndexCompatibilityError("embedding dimension differs from Milvus collection")
            row = {
                "chunk_id": chunk.chunk_id,
                "paper_id": chunk.paper_id,
                "text": chunk.text,
                "page": chunk.page or 0,
                "section": chunk.section,
                "metadata": _metadata(chunk),
                "dense": checked,
            }
            for name, max_bytes in (("chunk_id", 512), ("paper_id", 512), ("text", 16384), ("section", 1024)):
                if len(str(row[name]).encode("utf-8")) > max_bytes:
                    raise ValueError(f"{name} exceeds Milvus VARCHAR byte limit")
            rows.append(row)
        written = 0
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            result = self.client.upsert(collection_name=self.collection_name, data=batch)
            written += int(result.get("upsert_count", len(batch))) if isinstance(result, dict) else len(batch)
        return written

    def verify_documents(self, chunks: Sequence[DocumentChunk]) -> bool:
        """Confirm that every new primary key is readable before marking READY."""
        if not chunks:
            return False
        for start in range(0, len(chunks), 128):
            batch = chunks[start:start + 128]
            rows = self.client.get(
                collection_name=self.collection_name,
                ids=[chunk.chunk_id for chunk in batch],
                output_fields=["chunk_id", "paper_id"],
            )
            found = {row["chunk_id"] for row in rows}
            if found != {chunk.chunk_id for chunk in batch}:
                return False
        return True

    def search(
        self,
        query_embedding: Sequence[float] | None,
        *,
        top_k: int = 5,
        paper_id: str | None = None,
        paper_ids: Sequence[str] | None = None,
        query_text: str | None = None,
        mode: str = "hybrid",
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self._dimension is None:
            raise IndexCompatibilityError("call ensure_embedding_compatible before querying")
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("mode must be dense, bm25, or hybrid")
        vector = None
        if mode != "bm25":
            vector = _vector(query_embedding, "query_embedding")
            if len(vector) != self._dimension:
                raise IndexCompatibilityError("query embedding dimension differs from Milvus collection")
        if mode != "dense" and not (query_text or "").strip():
            raise ValueError("BM25 and hybrid retrieval require query_text")
        if paper_id is not None and paper_ids is not None:
            raise ValueError("paper_id and paper_ids cannot both be set")
        expression = None
        if paper_ids is not None:
            ids = [item.strip() for item in paper_ids if isinstance(item, str) and item.strip()]
            if not ids:
                return []
            expression = f"paper_id in {json.dumps(ids, ensure_ascii=False)}"
        if paper_id is not None:
            if not paper_id.strip():
                raise ValueError("paper_id must not be empty")
            expression = f"paper_id == {json.dumps(paper_id.strip(), ensure_ascii=False)}"
        if mode == "hybrid":
            from pymilvus import AnnSearchRequest, RRFRanker
            limit = max(top_k, candidate_k or top_k * 2)
            requests = [
                AnnSearchRequest(data=[vector], anns_field="dense", param={"metric_type": "COSINE"}, limit=limit, expr=expression),
                AnnSearchRequest(data=[query_text.strip()], anns_field="sparse", param={"metric_type": "BM25"}, limit=limit, expr=expression),
            ]
            raw = self.client.hybrid_search(
                collection_name=self.collection_name, reqs=requests,
                ranker=RRFRanker(), limit=top_k, output_fields=OUTPUT_FIELDS,
            )
        else:
            raw = self.client.search(
                collection_name=self.collection_name,
                data=[vector] if mode == "dense" else [query_text.strip()],
                anns_field="dense" if mode == "dense" else "sparse",
                search_params={"metric_type": "COSINE" if mode == "dense" else "BM25"},
                filter=expression or "",
                limit=top_k,
                output_fields=OUTPUT_FIELDS,
            )
        results = []
        for hit in raw[0] if raw else []:
            entity = hit.get("entity") or {}
            chunk_id = hit.get("chunk_id", hit.get("id"))
            if chunk_id is None:
                raise KeyError("Milvus search result has no primary key")
            raw_score = float(hit.get("distance", hit.get("score", 0.0)))
            if not math.isfinite(raw_score):
                continue
            score = (
                max(0.0, raw_score) / (1.0 + max(0.0, raw_score))
                if mode == "bm25" else max(0.0, min(1.0, raw_score))
            )
            metadata = dict(entity.get("metadata") or {})
            metadata.update({
                "paper_id": entity.get("paper_id"),
                "page": entity.get("page") or None,
                "section": entity.get("section") or "",
                "retrieval_mode": mode,
                "raw_retrieval_score": raw_score,
            })
            results.append({
                "chunk_id": str(chunk_id),
                "text": entity.get("text") or "",
                "metadata": metadata,
                "distance": 1.0 - score,
                "score": score,
            })
        return results

    def has_paper(self, paper_id: str) -> bool:
        if not paper_id or not paper_id.strip():
            raise ValueError("paper_id is required")
        if not self.client.has_collection(self.collection_name):
            return False
        self.prepare_bm25()
        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f"paper_id == {json.dumps(paper_id.strip(), ensure_ascii=False)}",
            output_fields=["chunk_id"],
            limit=1,
        )
        return bool(rows)

    def delete(
        self, chunk_ids: Sequence[str] | None = None, *, paper_id: str | None = None
    ) -> int:
        ids = [value.strip() for value in chunk_ids or [] if value and value.strip()]
        if not ids and paper_id is None:
            raise ValueError("chunk_ids or paper_id is required")
        terms = []
        if ids:
            terms.append(f"chunk_id in {json.dumps(ids, ensure_ascii=False)}")
        if paper_id is not None:
            if not paper_id.strip():
                raise ValueError("paper_id must not be empty")
            terms.append(f"paper_id == {json.dumps(paper_id.strip(), ensure_ascii=False)}")
        if not self.client.has_collection(self.collection_name):
            return 0
        result = self.client.delete(collection_name=self.collection_name, filter=" and ".join(terms))
        return int(result.get("delete_count", 0)) if isinstance(result, dict) else 0
