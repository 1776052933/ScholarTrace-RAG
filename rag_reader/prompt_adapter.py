"""Compose versionable chat prompts with LangChain Core without changing the provider API."""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any, Mapping

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


@lru_cache(maxsize=16)
def _template(system_prompt: str) -> ChatPromptTemplate:
    # System prompts contain JSON examples with braces; a literal message avoids
    # treating them as template variables. Only the user payload is formatted.
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("human", "{payload}"),
    ])


def build_messages(system_prompt: str, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render a project prompt to the existing DeepSeek chat message contract."""
    formatted = _template(system_prompt).format_messages(
        payload=json.dumps(dict(payload), ensure_ascii=False)
    )
    return [
        {"role": "system" if index == 0 else "user", "content": str(message.content)}
        for index, message in enumerate(formatted)
    ]
