"""根据用户问题检索单篇或多篇论文的相关文本片段。"""

from typing import Any, Iterable, List, Mapping, Optional

from .embedding import EmbeddingModel
from .schema import DocumentChunk
from .vector_store import VectorStore


class PaperRetriever:
    """组合查询向量模型和 Chroma VectorStore 的论文检索器。"""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        default_top_k: int = 5,
    ) -> None:
        if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer")
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k

    def retrieve(
        self,
        paper_id: Optional[str],
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """检索单篇论文；paper_id 为 ``None`` 时检索全部论文。"""
        question = _question(question)
        limit = _top_k(top_k, self.default_top_k)
        if paper_id is not None:
            paper_id = paper_id.strip()
            if not paper_id:
                raise ValueError("paper_id must not be empty")
        query_vector = self.embedding_model.embed(question)
        results = self.vector_store.search(
            query_vector,
            top_k=limit,
            paper_id=paper_id,
        )
        return _chunks(results, fallback_paper_id=paper_id)

    def retrieve_multiple(
        self,
        paper_ids: Iterable[str],
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """检索指定的多篇论文，并对结果进行全局 Top-K 排序。"""
        question = _question(question)
        limit = _top_k(top_k, self.default_top_k)
        ids = _paper_ids(paper_ids)
        if not ids:
            return []
        query_vector = self.embedding_model.embed(question)
        merged = []
        for paper_id in ids:
            merged.extend(self.vector_store.search(
                query_vector,
                top_k=limit,
                paper_id=paper_id,
            ))
        merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        unique, seen = [], set()
        for item in merged:
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id and chunk_id not in seen:
                unique.append(item)
                seen.add(chunk_id)
            if len(unique) >= limit:
                break
        return _chunks(unique)

    def retrieve_all(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """跨向量库中的全部论文检索。"""
        return self.retrieve(None, question, top_k=top_k)

    def search(
        self,
        question: str,
        *,
        paper_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """以 question 为首参数的兼容接口。"""
        return self.retrieve(paper_id, question, top_k=top_k)


def _chunks(
    results: Iterable[Mapping[str, Any]],
    *,
    fallback_paper_id: Optional[str] = None,
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
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            page=page,
            section=section,
            metadata=metadata,
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
