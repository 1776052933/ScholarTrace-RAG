"""Shared OpenAI-compatible connection and request mechanics.

Domain clients retain their own prompts, JSON parsing, and public exceptions.
This module deliberately adds no retry: a streamed partial answer is not safe
to replay, and callers already decide how to handle provider failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, Mapping, Sequence


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    base_url: str
    model: str


def resolve_provider(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    *,
    key_fallback: Callable[[], str] | None = None,
    url_fallback: Callable[[], str] | None = None,
    model_fallback: Callable[[], str] | None = None,
) -> ProviderConfig:
    """Explicit argument > environment > optional legacy key store > default."""
    def selected(explicit: str | None, env: str, fallback: Callable[[], str] | None, default: str) -> str:
        return str(explicit or os.getenv(env) or (fallback() if fallback else "") or default).strip()

    return ProviderConfig(
        api_key=selected(api_key, "DEEPSEEK_API_KEY", key_fallback, ""),
        base_url=selected(base_url, "DEEPSEEK_BASE_URL", url_fallback, DEFAULT_BASE_URL).rstrip("/"),
        model=selected(model, "DEEPSEEK_MODEL", model_fallback, DEFAULT_MODEL),
    )


def create_client(factory: Any, config: ProviderConfig, *, timeout: float, missing_sdk: str) -> Any:
    if factory is None:
        raise ImportError(missing_sdk)
    return factory(api_key=config.api_key, base_url=config.base_url, timeout=timeout)


def chat_response(
    client: Any,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    timeout: float,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Send one nonstreaming request without changing domain-specific validation."""
    options: dict[str, Any] = {"model": model, "messages": messages, "timeout": timeout}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return client.chat.completions.create(**options)


def assistant_content(response: Any) -> Any:
    return response.choices[0].message.content
