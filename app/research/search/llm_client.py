"""供 Search Planner、Evaluator 和 Reflector 使用的 DeepSeek 客户端。"""

import json
from ai_components.chains import generate_text
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
except ImportError:  # 保持不使用 LLM 时 search_agent 仍可导入。
    OpenAI = None  # type: ignore[assignment]

    class OpenAIError(Exception):
        pass

    class APIConnectionError(OpenAIError):
        pass

    class APITimeoutError(OpenAIError):
        pass


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 30.0


class DeepSeekError(RuntimeError):
    """DeepSeek 客户端异常基类。"""


class DeepSeekTimeoutError(DeepSeekError):
    """请求超过设定的 timeout。"""


class DeepSeekNetworkError(DeepSeekError):
    """连接失败、DNS 错误或网络中断。"""


class DeepSeekAPIError(DeepSeekError):
    """认证、限流、请求参数或服务端 API 错误。"""


class DeepSeekJSONError(DeepSeekError):
    """SDK 响应或模型输出不是有效 JSON。"""


# 兼容该模块早期异常名称。
SearchAgentLLMError = DeepSeekError


class DeepSeekClient:
    """同步调用 DeepSeek OpenAI-compatible Chat Completions API。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        _client: Any = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.timeout_seconds = float(timeout_seconds)

        if not self.api_key:
            raise ValueError("DeepSeek API key is missing; set DEEPSEEK_API_KEY or pass api_key")
        if self.timeout_seconds <= 0:
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
        logger.debug("DeepSeekClient initialized: model=%s, base_url=%s", self.model, self.base_url)

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """发送聊天消息并返回 assistant 文本。"""
        _validate_messages(messages)
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        logger.info("调用 DeepSeek：model=%s, message_count=%d", self.model, len(messages))
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=max_tokens,
                timeout=self.timeout_seconds,
            )
            content = response.choices[0].message.content
            if content is None:
                raise DeepSeekJSONError("DeepSeek returned empty message content")
            logger.info("DeepSeek 调用完成，响应字符数=%d", len(str(content)))
            return str(content)
        except (APITimeoutError, TimeoutError) as exc:
            logger.warning("DeepSeek 请求超时：%s", type(exc).__name__)
            raise DeepSeekTimeoutError("DeepSeek request timed out") from exc
        except (APIConnectionError, ConnectionError, OSError) as exc:
            logger.warning("DeepSeek 网络连接失败：%s", type(exc).__name__)
            raise DeepSeekNetworkError(f"DeepSeek network request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            logger.warning("DeepSeek SDK 响应 JSON 解析失败")
            raise DeepSeekJSONError("DeepSeek SDK returned invalid JSON") from exc
        except OpenAIError as exc:
            logger.warning("DeepSeek API 调用失败：%s", type(exc).__name__)
            raise DeepSeekAPIError(f"DeepSeek API request failed: {exc}") from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            logger.warning("DeepSeek 返回结构异常：%s", type(exc).__name__)
            raise DeepSeekJSONError("DeepSeek returned an invalid response structure") from exc

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """供 Planner、Evaluator、Reflector 获取结构化 JSON 响应。"""
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
    """从纯 JSON、Markdown 代码块或说明文字中提取首个 JSON 对象。"""
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
    logger.warning("模型输出中未找到有效 JSON 对象")
    raise DeepSeekJSONError("No valid JSON object found in DeepSeek response")


def _validate_messages(messages: List[Mapping[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("role") or "content" not in message:
            raise ValueError("each message must contain role and content")

