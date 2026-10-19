"""使用 OpenAI Python SDK 封装 DeepSeek Chat API。"""

import json
from ai_components.chains import generate_text
import os
import re
from typing import Any, Dict, List, Mapping, Optional

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
except ImportError:  # 允许未安装 SDK 时导入离线拆解功能。
    OpenAI = None  # type: ignore[assignment]

    class OpenAIError(Exception):
        pass

    class APIConnectionError(OpenAIError):
        pass

    class APITimeoutError(OpenAIError):
        pass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048


class DeepSeekError(RuntimeError):
    """DeepSeek 客户端异常基类。"""


class DeepSeekAPIError(DeepSeekError):
    """DeepSeek API 返回认证、限流、参数或服务端错误。"""


class DeepSeekNetworkError(DeepSeekError):
    """连接失败、网络中断或请求超时。"""


class DeepSeekJSONError(DeepSeekError):
    """API 响应或模型文本无法解析为期望的 JSON。"""


# 保留旧异常名称，避免模块内部或已有调用方失效。
DecompositionLLMError = DeepSeekError


class DeepSeekClient:
    """同步 DeepSeek OpenAI-compatible Chat Completions 客户端。

    配置优先级为“显式构造参数 → 环境变量 → 默认值”。API Key 没有默认值，
    必须通过 ``api_key`` 或 ``DEEPSEEK_API_KEY`` 提供。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = 30.0,
        *,
        _client: Any = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.temperature = _validate_temperature(temperature)
        self.max_tokens = _validate_max_tokens(max_tokens)
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError("DeepSeek API key is missing; set DEEPSEEK_API_KEY or pass api_key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        if _client is not None:
            self._client = _client
        else:
            if OpenAI is None:
                raise ImportError("The 'openai' package is required: pip install openai")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发送聊天消息并返回模型文本。

        ``temperature`` 和 ``max_tokens`` 可以在单次调用中覆盖构造时的默认值。
        """
        _validate_messages(messages)
        request_temperature = self.temperature if temperature is None else _validate_temperature(temperature)
        request_max_tokens = self.max_tokens if max_tokens is None else _validate_max_tokens(max_tokens)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request_temperature,
                max_tokens=request_max_tokens,
                timeout=self.timeout_seconds,
            )
            content = response.choices[0].message.content
            if content is None:
                raise DeepSeekJSONError("DeepSeek returned empty message content")
            return str(content)
        except (APITimeoutError, APIConnectionError, TimeoutError, ConnectionError, OSError) as exc:
            raise DeepSeekNetworkError(f"DeepSeek network request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise DeepSeekJSONError("DeepSeek SDK returned invalid JSON") from exc
        except OpenAIError as exc:
            raise DeepSeekAPIError(f"DeepSeek API request failed: {exc}") from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise DeepSeekJSONError("DeepSeek returned an invalid response structure") from exc

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """发送 system/user 消息并将模型文本解析为 JSON 对象。"""
        content = generate_text(self,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_object(content)


def extract_json_object(content: str) -> Dict[str, Any]:
    """从纯 JSON、Markdown 代码块或夹杂说明文字中提取 JSON 对象。"""
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekJSONError("DeepSeek returned empty JSON content")
    text = content.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise DeepSeekJSONError("No valid JSON object found in DeepSeek response")


def _validate_messages(messages: List[Mapping[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("role") or "content" not in message:
            raise ValueError("each message must contain role and content")


def _validate_temperature(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a number") from exc
    if not 0.0 <= result <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    return result


def _validate_max_tokens(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return value

