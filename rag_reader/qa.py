"""基于检索论文片段和 DeepSeek 的单篇论文问答。"""

import json
import logging
import re
from threading import Lock
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .llm_client import DeepSeekClient
from .errors import RAGGenerationError
from .prompt_adapter import build_messages
from .prompts import RAG_QA_PROMPT
from .retriever import PaperRetriever
from .schema import EvidenceCitation, RAGAnswer


logger = logging.getLogger(__name__)


class RAGAnswerParseError(RAGGenerationError, ValueError):
    """DeepSeek 问答输出不是合法 RAGAnswer。"""


class PaperQA:
    """执行“问题→Retriever→Top-K Chunk→DeepSeek→回答”流程。"""

    def __init__(
        self,
        retriever: PaperRetriever,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        top_k: int = 5,
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self._client_lock = Lock()

    def ask(self, paper_id: str, question: str) -> RAGAnswer:
        """基于指定论文的 Top-K 证据片段回答问题。"""
        paper_id = _required_text(paper_id, "paper_id")
        question = _required_text(question, "question")
        chunks = self.retriever.retrieve(paper_id, question, top_k=self.top_k)
        if not chunks:
            return RAGAnswer(
                question=question,
                answer="根据提供的论文片段无法确定。",
                sources=[],
                confidence=0.0,
            )

        payload = {
            "paper_id": paper_id,
            "question": question,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "paper_id": chunk.paper_id,
                    "page": chunk.page,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        }
        client = self._get_client()
        logger.info("调用 DeepSeek 论文问答：paper_id=%s, chunks=%d", paper_id, len(chunks))
        content = client.chat(
            build_messages(RAG_QA_PROMPT, payload),
            temperature=0.0,
            max_tokens=2048,
        )
        try:
            data = _extract_json(content)
            data["question"] = question
            valid_sources = {chunk.chunk_id for chunk in chunks}
            raw_sources = data.get("sources") or []
            if not isinstance(raw_sources, list):
                raise TypeError("sources must be a list")
            sources = list(dict.fromkeys(
                str(source) for source in raw_sources if str(source) in valid_sources
            ))
            data["sources"] = sources
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            data["citations"] = [
                EvidenceCitation(
                    paper_id=by_id[source].paper_id,
                    chunk_id=source,
                    title=str(by_id[source].metadata.get("title") or ""),
                    page=by_id[source].page,
                    section=by_id[source].section,
                    excerpt=by_id[source].text,
                    retrieval_score=by_id[source].metadata.get("retrieval_score"),
                )
                for source in sources
            ]
            if not sources:
                data["confidence"] = min(float(data.get("confidence") or 0.0), 0.5)
            return RAGAnswer.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("DeepSeek 论文问答 JSON 解析失败：paper_id=%s", paper_id)
            raise RAGAnswerParseError(f"invalid RAG answer JSON for paper {paper_id}") from exc

    def _get_client(self) -> DeepSeekClient:
        client = self.llm_client
        if client is not None:
            return client
        with self._client_lock:
            if self.llm_client is None:
                self.llm_client = DeepSeekClient()
            return self.llm_client


def _extract_json(content: str) -> Dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise RAGAnswerParseError("DeepSeek returned empty answer")
    text = content.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RAGAnswerParseError("no valid JSON object found in DeepSeek response")


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result
