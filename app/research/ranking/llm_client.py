"""Ranking Agent 使用的 DeepSeek OpenAI-compatible 客户端。"""

import os
from typing import Any, List, Mapping, Optional

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
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or _project_key()).strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout = float(timeout)
        if not self.api_key:
            raise ValueError("DeepSeek API key is missing")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if _client is not None:
            self._client = _client
        else:
            if OpenAI is None:
                raise ImportError("DeepSeekClient requires openai: pip install openai")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """调用 DeepSeek 并返回 assistant 文本。"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        try:
            content = response.choices[0].message.content
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
