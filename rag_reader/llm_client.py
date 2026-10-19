"""RAG Reader 使用的 DeepSeek OpenAI-compatible 客户端。"""

from dataclasses import asdict, dataclass
import logging
import os
from threading import Lock
from typing import Any, Iterator, List, Mapping, Optional

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
except ImportError:  # 允许未启用 LLM 时导入 rag_reader。
    OpenAI = None  # type: ignore[assignment]

    class OpenAIError(Exception):
        pass

    class APIConnectionError(OpenAIError):
        pass

    class APITimeoutError(OpenAIError):
        pass


from .config import RAGSettings
from .errors import RAGGenerationError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 60.0


class DeepSeekError(RAGGenerationError):
    """DeepSeek 客户端异常基类。"""


class DeepSeekTimeoutError(DeepSeekError):
    """DeepSeek 请求超时。"""


class DeepSeekNetworkError(DeepSeekError):
    """DeepSeek 网络连接失败。"""


class DeepSeekAPIError(DeepSeekError):
    """DeepSeek API 返回错误。"""


class DeepSeekResponseError(DeepSeekError):
    """DeepSeek 返回结构无效或内容为空。"""


@dataclass(frozen=True)
class TokenUsage:
    """一次或累计请求的 token 使用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class DeepSeekClient:
    """供论文摘要、知识抽取和 RAG 问答使用的同步客户端。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        _client: Any = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("DEEPSEEK_API_KEY") or _project_key()).strip()
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout)
        if not self.api_key:
            raise ValueError("DeepSeek API key is missing; set DEEPSEEK_API_KEY")
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
        self.last_usage = TokenUsage()
        self.total_usage = TokenUsage()
        self._usage_lock = Lock()
        logger.debug("DeepSeekClient initialized: model=%s, base_url=%s", self.model, self.base_url)

    def chat(
        self,
        messages: List[Mapping[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """发送 Chat Completions 请求并返回 assistant 文本。"""
        _validate_messages(messages)
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        logger.info(
            "调用 DeepSeek：model=%s, messages=%d, max_tokens=%d",
            self.model,
            len(messages),
            max_tokens,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            content = response.choices[0].message.content
            if not content:
                raise DeepSeekResponseError("DeepSeek returned empty content")
            usage = _token_usage(getattr(response, "usage", None))
            self._record_usage(usage)
            logger.info(
                "DeepSeek 调用完成：prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            return str(content)
        except DeepSeekResponseError:
            raise
        except (APITimeoutError, TimeoutError) as exc:
            logger.warning("DeepSeek 请求超时：%s", type(exc).__name__)
            raise DeepSeekTimeoutError("DeepSeek request timed out") from exc
        except (APIConnectionError, ConnectionError, OSError) as exc:
            logger.warning("DeepSeek 网络连接失败：%s", type(exc).__name__)
            raise DeepSeekNetworkError(f"DeepSeek network request failed: {exc}") from exc
        except OpenAIError as exc:
            logger.warning("DeepSeek API 调用失败：%s", type(exc).__name__)
            raise DeepSeekAPIError(f"DeepSeek API request failed: {exc}") from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            logger.warning("DeepSeek 返回结构异常：%s", type(exc).__name__)
            raise DeepSeekResponseError("DeepSeek returned an invalid response structure") from exc

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        """返回最近一次请求和当前客户端累计 token 统计。"""
        with self._usage_lock:
            return {
                "last": self.last_usage.to_dict(),
                "total": self.total_usage.to_dict(),
            }

    def stream_chat(self, messages: List[Mapping[str, Any]], *,
                    temperature: float = 0.0, max_tokens: int = 2048) -> Iterator[str]:
        """Yield provider deltas immediately; never replay a partially delivered call."""
        _validate_messages(messages)
        if not 0 <= float(temperature) <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        stream = None
        usage = TokenUsage()
        finished = None
        has_content = False
        try:
            stream = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=float(temperature),
                max_tokens=max_tokens, timeout=self.timeout, stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = _token_usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finished = choice.finish_reason
                content = choice.delta.content
                if content:
                    has_content = True
                    yield str(content)
            if finished != "stop":
                raise DeepSeekResponseError(f"DeepSeek stream ended without a complete answer ({finished})")
            if not has_content:
                raise DeepSeekResponseError("DeepSeek returned empty content")
        except (APITimeoutError, TimeoutError) as exc:
            raise DeepSeekTimeoutError("DeepSeek stream timed out") from exc
        except (APIConnectionError, ConnectionError, OSError) as exc:
            raise DeepSeekNetworkError("DeepSeek stream connection failed") from exc
        except OpenAIError as exc:
            raise DeepSeekAPIError("DeepSeek streaming API request failed") from exc
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise DeepSeekResponseError("DeepSeek returned an invalid stream structure") from exc
        finally:
            self._record_usage(usage)
            if stream is not None:
                stream.close()

    def reset_token_usage(self) -> None:
        """清零最近一次及累计 token 统计。"""
        with self._usage_lock:
            self.last_usage = TokenUsage()
            self.total_usage = TokenUsage()

    def _record_usage(self, usage: TokenUsage) -> None:
        with self._usage_lock:
            self.last_usage = usage
            self.total_usage = TokenUsage(
                prompt_tokens=self.total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=self.total_usage.completion_tokens + usage.completion_tokens,
                total_tokens=self.total_usage.total_tokens + usage.total_tokens,
            )


def _token_usage(value: Any) -> TokenUsage:
    if value is None:
        return TokenUsage()
    prompt = _usage_value(value, "prompt_tokens")
    completion = _usage_value(value, "completion_tokens")
    total = _usage_value(value, "total_tokens")
    if total == 0 and (prompt or completion):
        total = prompt + completion
    return TokenUsage(prompt, completion, total)


def _usage_value(value: Any, name: str) -> int:
    raw = value.get(name, 0) if isinstance(value, Mapping) else getattr(value, name, 0)
    try:
        result = int(raw or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _validate_messages(messages: List[Mapping[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping) or not message.get("role") or "content" not in message:
            raise ValueError("each message must contain role and content")


def _project_key() -> str:
    try:
        from API_key.key_manager import key_get

        return str(key_get("deepseek") or "")
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""

class LazyDeepSeekClient:
    """Share one configured DeepSeek client without requiring a key at RAG startup."""

    def __init__(self, settings: RAGSettings) -> None:
        self.settings = settings
        self._client: DeepSeekClient | None = None
        self._lock = Lock()

    def _get_client(self) -> DeepSeekClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = DeepSeekClient(
                        base_url=self.settings.llm_base_url,
                        model=self.settings.llm_model,
                        timeout=self.settings.llm_timeout_seconds,
                    )
        return self._client

    def chat(self, messages: List[Mapping[str, Any]], **kwargs: Any) -> str:
        return self._get_client().chat(messages, **kwargs)

    def stream_chat(self, messages: List[Mapping[str, Any]], **kwargs: Any) -> Iterator[str]:
        yield from self._get_client().stream_chat(messages, **kwargs)

    def get_token_usage(self) -> dict[str, dict[str, int]]:
        if self._client is None:
            empty = TokenUsage().to_dict()
            return {"last": empty.copy(), "total": empty.copy()}
        return self._client.get_token_usage()
