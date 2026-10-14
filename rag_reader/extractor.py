"""使用 DeepSeek 从论文正文抽取结构化科研知识。"""

import json
import logging
import re
from threading import Lock
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .llm_client import DeepSeekClient
from .prompts import PAPER_EXTRACTION_PROMPT
from .schema import PaperDocument, PaperKnowledge


logger = logging.getLogger(__name__)


class PaperKnowledgeParseError(ValueError):
    """DeepSeek 返回内容无法解析为 ``PaperKnowledge``。"""


class PaperKnowledgeExtractor:
    """依据论文全文抽取研究问题、方法、实验、结果及局限。"""

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        max_input_chars: int = 60000,
    ) -> None:
        if (
            isinstance(max_input_chars, bool)
            or not isinstance(max_input_chars, int)
            or max_input_chars <= 0
        ):
            raise ValueError("max_input_chars must be a positive integer")
        self.llm_client = llm_client
        self.max_input_chars = max_input_chars
        self._client_lock = Lock()

    def extract(self, document: PaperDocument) -> PaperKnowledge:
        """从 ``PaperDocument`` 中抽取严格校验的结构化知识。"""
        if not isinstance(document, PaperDocument):
            raise TypeError("document must be a PaperDocument")
        content = document.content.strip()
        if not content:
            raise ValueError("document content must not be empty")
        if len(content) > self.max_input_chars:
            logger.warning(
                "知识抽取输入超过限制，截断正文：paper_id=%s, characters=%d, limit=%d",
                document.paper_id,
                len(content),
                self.max_input_chars,
            )
            content = content[:self.max_input_chars]

        client = self._get_client()
        payload = {"title": document.title, "content": content}
        logger.info("调用 DeepSeek 抽取论文知识：paper_id=%s", document.paper_id)
        response = client.chat(
            [
                {"role": "system", "content": PAPER_EXTRACTION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=3000,
        )
        try:
            data = _extract_json(response)
            # 标题以输入论文元数据为准，避免模型改写或漏填。
            data["title"] = document.title
            return PaperKnowledge.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("DeepSeek 论文知识 JSON 解析失败：paper_id=%s", document.paper_id)
            raise PaperKnowledgeParseError(
                f"invalid paper knowledge JSON: {document.paper_id}"
            ) from exc

    def _get_client(self) -> DeepSeekClient:
        client = self.llm_client
        if client is not None:
            return client
        with self._client_lock:
            if self.llm_client is None:
                self.llm_client = DeepSeekClient()
            return self.llm_client


# 简短别名，便于外部 Agent 按职责名称注入。
KnowledgeExtractor = PaperKnowledgeExtractor


def _extract_json(content: str) -> Dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise PaperKnowledgeParseError("DeepSeek returned empty knowledge")
    text = content.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PaperKnowledgeParseError("no valid JSON object found in DeepSeek response")
