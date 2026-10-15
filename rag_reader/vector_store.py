"""基于 Chroma 的论文 Chunk 向量存储。"""

import json
import logging
import math
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .errors import IndexCompatibilityError
from .schema import DocumentChunk


logger = logging.getLogger(__name__)


class VectorStore:
    """保存 Chunk embedding、正文及 paper/page/section metadata。"""

    def __init__(
        self,
        collection_name: str = "paper_chunks_m1",
        *,
        persist_directory: Optional[str] = None,
        client: Any = None,
        collection: Any = None,
    ) -> None:
        collection_name = collection_name.strip()
        if not collection_name:
            raise ValueError("collection_name must not be empty")
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._configuration_lock = RLock()

        if collection is not None:
            self.client = client
            self.collection = collection
            return
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "VectorStore requires chromadb; install it with: pip install chromadb"
            ) from exc

        if client is None:
            if persist_directory:
                path = str(Path(persist_directory).expanduser())
                client = chromadb.PersistentClient(path=path)
            else:
                client = chromadb.Client()
        self.client = client
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_embedding_compatible(self, fingerprint: str, dimension: int) -> None:
        """Bind an empty collection, or reject a mismatched/legacy vector index."""
        fingerprint = fingerprint.strip()
        if not fingerprint or dimension < 1:
            raise ValueError("fingerprint and positive dimension are required")
        with self._configuration_lock:
            metadata = dict(self.collection.metadata or {})
            saved_fingerprint = metadata.get("embedding_fingerprint")
            saved_dimension = metadata.get("embedding_dimension")
            if saved_fingerprint is None:
                if self.collection.count() > 0:
                    raise IndexCompatibilityError(
                        f"collection {self.collection_name} has unversioned vectors; "
                        "rebuild into a new collection before querying or indexing"
                    )
                self.collection.modify(metadata={
                    **{key: value for key, value in metadata.items() if key != "hnsw:space"},
                    "embedding_fingerprint": fingerprint,
                    "embedding_dimension": dimension,
                })
                return
            if saved_fingerprint != fingerprint or saved_dimension != dimension:
                raise IndexCompatibilityError(
                    f"collection {self.collection_name} was built with "
                    f"{saved_fingerprint}/{saved_dimension}, requested {fingerprint}/{dimension}"
                )

    def add_documents(
        self,
        chunks: Iterable[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """批量新增或更新 Chunk，返回写入数量。"""
        documents = list(chunks)
        vectors = list(embeddings)
        if len(documents) != len(vectors):
            raise ValueError("chunks and embeddings must have the same length")
        if not documents:
            return 0

        ids, texts, metadatas, checked_vectors = [], [], [], []
        seen_ids = set()
        dimension = None
        for index, (chunk, vector) in enumerate(zip(documents, vectors)):
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(f"chunks[{index}] must be a DocumentChunk")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id in batch: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            checked = _vector(vector, f"embeddings[{index}]")
            dimension = dimension or len(checked)
            if len(checked) != dimension:
                raise ValueError("all embeddings must have the same dimension")
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(_metadata(chunk))
            checked_vectors.append(checked)

        self.collection.upsert(
            ids=ids,
            embeddings=checked_vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("写入 Chroma：collection=%s, chunks=%d", self.collection_name, len(ids))
        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        paper_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按余弦距离检索 Chunk，可限定单篇论文。"""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query = _vector(query_embedding, "query_embedding")
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if paper_id is not None:
            paper_id = paper_id.strip()
            if not paper_id:
                raise ValueError("paper_id must not be empty")
            kwargs["where"] = {"paper_id": paper_id}
        response = self.collection.query(**kwargs)
        return _search_results(response)

    def delete(
        self,
        chunk_ids: Optional[Sequence[str]] = None,
        *,
        paper_id: Optional[str] = None,
    ) -> int:
        """按 Chunk ID、paper_id 或两者交集删除，返回匹配记录数。"""
        ids = [value.strip() for value in (chunk_ids or []) if value and value.strip()]
        where = None
        if paper_id is not None:
            paper_id = paper_id.strip()
            if not paper_id:
                raise ValueError("paper_id must not be empty")
            where = {"paper_id": paper_id}
        if not ids and where is None:
            raise ValueError("chunk_ids or paper_id is required")

        get_kwargs: Dict[str, Any] = {"include": []}
        delete_kwargs: Dict[str, Any] = {}
        if ids:
            get_kwargs["ids"] = ids
            delete_kwargs["ids"] = ids
        if where is not None:
            get_kwargs["where"] = where
            delete_kwargs["where"] = where
        matched = self.collection.get(**get_kwargs).get("ids") or []
        if matched:
            self.collection.delete(**delete_kwargs)
        logger.info("删除 Chroma 记录：collection=%s, chunks=%d", self.collection_name, len(matched))
        return len(matched)


def _metadata(chunk: DocumentChunk) -> Dict[str, Any]:
    metadata = {
        key: _metadata_value(value)
        for key, value in chunk.metadata.items()
        if value is not None
    }
    metadata["paper_id"] = chunk.paper_id
    metadata["section"] = chunk.section
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _metadata_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _vector(value: Sequence[float], name: str) -> List[float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a numeric vector")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric vector") from exc
    if not values:
        raise ValueError(f"{name} must not be empty")
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{name} must contain only real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        result.append(number)
    return result


def _search_results(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    results = []
    for index, chunk_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 0.0
        results.append({
            "chunk_id": chunk_id,
            "text": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "distance": distance,
            "score": max(0.0, min(1.0, 1.0 - distance)),
        })
    return results
