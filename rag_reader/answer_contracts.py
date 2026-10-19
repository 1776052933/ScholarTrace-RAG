"""Answer validation and context budgets, independent of workflow orchestration."""
import json
import logging
import re
from typing import Any, Dict, Sequence

from pydantic import ValidationError

from .errors import RAGGenerationError
from .prompt_adapter import build_messages
from .schema import EvidenceCitation, RAGAnswer

logger = logging.getLogger(__name__)


class RAGAnswerParseError(RAGGenerationError, ValueError):
    """The provider returned an invalid structured answer."""


def _parse_answer(content: str, question: str, chunks: Sequence[Any], *,
                  evidence_sufficient: bool = True) -> RAGAnswer:
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
        if not evidence_sufficient:
            data["confidence"] = min(float(data.get("confidence") or 0.0), 0.5)
        return RAGAnswer.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        logger.warning("DeepSeek 论文问答 JSON 解析失败")
        raise RAGAnswerParseError("invalid RAG answer JSON") from exc

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


def _fit_evidence(prompt, payload, history, chunks, max_chars):
    """Bound the entire serialized input, including history, JSON escaping and evidence."""
    used = []

    def render():
        payload["chunks"] = [
            {"chunk_id": c.chunk_id, "paper_id": c.paper_id, "page": c.page,
             "section": c.section, "text": c.text} for c in used
        ]
        return build_messages(prompt, payload, history=history)

    if len(json.dumps(render(), ensure_ascii=False)) > max_chars:
        raise RAGGenerationError("对话上下文超过输入预算，请缩短问题或新建会话。")
    for chunk in chunks:
        used.append(chunk)
        if len(json.dumps(render(), ensure_ascii=False)) <= max_chars:
            continue
        used.pop()
        low, high = 0, len(chunk.text)
        while low < high:
            mid = (low + high + 1) // 2
            used.append(chunk.model_copy(update={"text": chunk.text[:mid]}))
            fits = len(json.dumps(render(), ensure_ascii=False)) <= max_chars
            used.pop()
            if fits:
                low = mid
            else:
                high = mid - 1
        if low >= 80:
            used.append(chunk.model_copy(update={"text": chunk.text[:low]}))
        break
    # Dropping all evidence for one paper must never silently turn a comparison into a single-paper answer.
    if {c.paper_id for c in used} != {c.paper_id for c in chunks}:
        raise RAGGenerationError("输入预算不足以容纳所有论文的证据，请减少论文数量或新建会话。")
    return render(), used
