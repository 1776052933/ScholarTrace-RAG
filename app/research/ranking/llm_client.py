"""Ranking Agent 使用的 DeepSeek OpenAI-compatible 客户端。"""

from typing import Any, List, Mapping, Optional

from ai_components.provider import assistant_content, chat_response, create_client, resolve_provider

try:
    from openai import OpenAI
except ImportError:  # 允许未启用 LLM 时导入 ranking_agent。
    OpenAI = None  # type: ignore[assignment]


class DeepSeekClient:
    """发送 DeepSeek Chat Completions 请求并返回文本。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        *,
        _client: Any = None,
    ) -> None:
        config = resolve_provider(api_key, base_url, model, key_fallback=_project_key)
        self.api_key, self.base_url, self.model = config.api_key, config.base_url, config.model
        self.timeout = float(timeout)
        if not self.api_key:
            raise ValueError("DeepSeek API key is missing")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if _client is not None:
            self._client = _client
        else:
            self._client = create_client(OpenAI, config, timeout=self.timeout,
                                         missing_sdk="DeepSeekClient requires openai: pip install openai")

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """调用 DeepSeek 并返回 assistant 文本。"""
        response = chat_response(self._client, self.model, messages, self.timeout,
                                 temperature=temperature, max_tokens=max_tokens)
        try:
            content = assistant_content(response)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek returned an invalid response structure") from exc
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        return str(content)


def _project_key() -> str:
    try:
        from API_key.key_manager import key_get

        return str(key_get("deepseek") or "")
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""
