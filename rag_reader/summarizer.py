"""使用 DeepSeek 生成结构化论文总结。"""

import json
import logging
import re
from threading import Lock
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from .llm_client import DeepSeekClient
from .prompts import SUMMARY_PROMPT
from .schema import PaperDocument


logger = logging.getLogger(__name__)


class PaperSummaryParseError(ValueError):
    """DeepSeek 总结输出无法解析为指定结构。"""


class _PaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    background: str
    method: str
    experiment: str
    result: str
    limitation: str
    conclusion: str


class PaperSummarizer:
    """调用 DeepSeek 生成背景、方法、实验、结果、局限和结论。"""

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

    def summarize(self, document: PaperDocument) -> Dict[str, str]:
        """总结 PaperDocument 并返回六字段字典。"""
        if not isinstance(document, PaperDocument):
            raise TypeError("document must be a PaperDocument")
        return self.summarize_text(document.title, document.content)

    def summarize_text(self, title: str, content: str) -> Dict[str, str]:
        """直接根据论文标题和正文生成总结。"""
        title = _required_text(title, "title")
        content = _required_text(content, "content")
        if len(content) > self.max_input_chars:
            logger.warning(
                "论文总结输入超过限制，截断正文：characters=%d, limit=%d",
                len(content),
                self.max_input_chars,
            )
            content = content[:self.max_input_chars]

        client = self._get_client()
        payload = {"title": title, "content": content}
        logger.info("调用 DeepSeek 生成论文总结：title=%s", title)
        response = client.chat(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        try:
            summary = _PaperSummary.model_validate(_extract_json(response))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("DeepSeek 论文总结 JSON 解析失败：title=%s", title)
            raise PaperSummaryParseError(f"invalid paper summary JSON: {title}") from exc
        return summary.model_dump(mode="json")

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
        raise PaperSummaryParseError("DeepSeek returned empty summary")
    text = content.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PaperSummaryParseError("no valid JSON object found in DeepSeek response")


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result
