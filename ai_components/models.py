"""LangChain chat interface over the existing, tested provider transports.

Transports own credentials, HTTP, timeouts and provider errors. This adapter owns
LangChain messages, invocation, streaming and callbacks. No additional retry is
introduced here: retrying an already emitted stream would duplicate answer text.
"""
from __future__ import annotations

from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ProviderChatModel(BaseChatModel):
    transport: Any = Field(exclude=True, repr=False)
    model_name: str = "provider-chat"

    @property
    def _llm_type(self) -> str:
        return "scholartrace-provider"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # Never serialize the transport (which contains credentials).
        return {"model_name": self.model_name}

    @staticmethod
    def _messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        result = []
        for message in messages:
            if message.type not in roles or not isinstance(message.content, str):
                raise ValueError("This text-generation adapter supports system/human/ai text messages only")
            result.append({"role": roles[message.type], "content": message.content})
        return result

    def _usage(self):
        if not hasattr(self.transport, "get_token_usage"):
            return None
        usage = self.transport.get_token_usage().get("last", {})
        inputs, outputs = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        return {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if stop:
            raise ValueError("Provider transport does not support stop sequences")
        kwargs.pop("cancel_event", None)
        content = self.transport.chat(self._messages(messages), **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=content, usage_metadata=self._usage(),
        ))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        if stop:
            raise ValueError("Provider transport does not support stop sequences")
        cancel = kwargs.pop("cancel_event", None)
        if not hasattr(self.transport, "stream_chat"):
            raise NotImplementedError("This provider transport does not implement streaming")
        stream = self.transport.stream_chat(self._messages(messages), **kwargs)
        try:
            for text in stream:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("answer stream cancelled")
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=self._usage()))
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def as_chat_model(client: Any) -> BaseChatModel:
    """Native LangChain models can be injected without another wrapper."""
    if isinstance(client, BaseChatModel):
        return client
    return ProviderChatModel(transport=client, model_name=str(getattr(client, "model", "provider-chat")))
