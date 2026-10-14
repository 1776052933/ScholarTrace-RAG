"""基于 OpenAI 兼容 Python SDK 构建的 DeepSeek 客户端。"""

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Protocol

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
except ImportError:  # 保证离线查询理解在未安装 SDK 时仍可导入。
    OpenAI = None  # type: ignore[assignment]

    class OpenAIError(Exception):
        """仅当可选 SDK 不可用时的回退类型。"""

    class APIConnectionError(OpenAIError):
        pass

    class APITimeoutError(OpenAIError):
        pass


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMClient(Protocol):
    """查询理解流水线所使用的协议接口。"""

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


class LLMClientError(RuntimeError):
    """DeepSeek 客户端所有失败的基类异常。"""


class DeepSeekAPIError(LLMClientError):
    """DeepSeek API 请求或响应失败。"""


class DeepSeekTimeoutError(LLMClientError):
    """DeepSeek 请求超时。"""


class DeepSeekJSONError(LLMClientError):
    """模型响应无法解码为 JSON 对象。"""


class DeepSeekClient:
    """DeepSeek OpenAI 兼容聊天 API 的同步客户端。

    显式传入的构造参数优先级高于环境变量 / key_manager 中的值。
    本模块不内置任何 API Key。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        *,
        _client: Any = None,
    ) -> None:
        self.api_key = api_key or _read_setting("DEEPSEEK_API_KEY", "deepseek")
        self.base_url = base_url or _read_setting("DEEPSEEK_BASE_URL", "deepseek_base_url") or DEFAULT_BASE_URL
        self.model = model or _read_setting("DEEPSEEK_MODEL", "deepseek_model") or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is missing; pass api_key or configure "
                "DEEPSEEK_API_KEY/API_key.key_manager"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if _client is not None:
            self._client = _client
        else:
            if OpenAI is None:
                raise ImportError("The 'openai' package is required: pip install openai")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
            )

    def chat(self, messages: List[Mapping[str, Any]]) -> str:
        """发送对话消息并返回助手文本。"""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, Mapping) or not message.get("role") or "content" not in message:
                raise ValueError("each message must contain role and content")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.timeout_seconds,
            )
            content = response.choices[0].message.content
            if content is None:
                raise DeepSeekAPIError("DeepSeek returned an empty message")
            return str(content)
        except (APITimeoutError, TimeoutError) as exc:
            raise DeepSeekTimeoutError("DeepSeek request timed out") from exc
        except json.JSONDecodeError as exc:
            raise DeepSeekJSONError("DeepSeek SDK returned invalid JSON") from exc
        except (APIConnectionError, OpenAIError) as exc:
            raise DeepSeekAPIError(f"DeepSeek API request failed: {exc}") from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise DeepSeekAPIError("DeepSeek returned an invalid response structure") from exc

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """流水线适配方法：请求 DeepSeek 返回 JSON 并解码。"""
        content = self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _parse_json_object(content)


class OpenAICompatibleClient(DeepSeekClient):
    """为旧版查询理解调用方提供的向后兼容别名。"""


def _read_setting(environment_name: str, key_manager_name: str) -> str:
    """从环境变量或现有项目 key_manager 读取配置，不产生副作用。"""
    environment_value = os.environ.get(environment_name, "").strip()
    if environment_value:
        return environment_value
    try:
        from API_key.key_manager import key_get

        for name in (key_manager_name, environment_name):
            value = key_get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    except (ImportError, OSError, ValueError, TypeError):
        return ""


def _parse_json_object(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise DeepSeekJSONError("DeepSeek response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DeepSeekJSONError("DeepSeek JSON response must be an object")
    return value
