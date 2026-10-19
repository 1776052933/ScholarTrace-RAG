"""Shared LCEL generation chains; domain validation stays with each caller."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .models import as_chat_model


def text_chain(client, **model_options):
    """Accept {messages: ...}, preserve history order and stream incrementally."""
    return (
        ChatPromptTemplate.from_messages([MessagesPlaceholder("messages")])
        | as_chat_model(client).bind(**{k: v for k, v in model_options.items() if v is not None})
        | StrOutputParser()
    )


def generate_text(client, messages, *, config=None, **model_options) -> str:
    return text_chain(client, **model_options).invoke({"messages": messages}, config=config)
