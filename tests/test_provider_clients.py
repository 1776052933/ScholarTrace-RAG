"""Five domain clients share provider transport without changing their calls."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from app.research.understanding.llm_client import DeepSeekClient as UnderstandingClient
from app.research.decomposition.llm_client import DeepSeekClient as DecompositionClient
from app.research.search.llm_client import DeepSeekClient as SearchClient
from app.research.ranking.llm_client import DeepSeekClient as RankingClient
from rag_reader.llm_client import DeepSeekClient as ReaderClient


@pytest.mark.parametrize("client_type,expected_tokens", [
    (UnderstandingClient, None),
    (DecompositionClient, 2048),
    (SearchClient, 2048),
    (RankingClient, 1000),
    (ReaderClient, 4096),
])
def test_domain_client_preserves_request_and_response(client_type, expected_tokens):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content="answer"))], usage=None)

    transport = NS(chat=NS(completions=NS(create=create)))
    kwargs = {"api_key": "test-key", "model": "fixture-model", "base_url": "http://fixture/v1/", "_client": transport}
    if client_type in (RankingClient, ReaderClient):
        kwargs["timeout"] = 12
    else:
        kwargs["timeout_seconds"] = 12
    client = client_type(**kwargs)
    messages = [{"role": "user", "content": "question"}]

    assert client.chat(messages) == "answer"
    assert client.base_url == "http://fixture/v1"
    assert calls[0]["model"] == "fixture-model"
    assert calls[0]["messages"] == messages
    assert calls[0]["timeout"] == 12
    assert calls[0].get("max_tokens") == expected_tokens


def test_explicit_provider_config_overrides_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://environment/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "environment-model")
    from ai_components.provider import resolve_provider

    config = resolve_provider("explicit-key", "http://explicit/v1/", "explicit-model")
    assert (config.api_key, config.base_url, config.model) == (
        "explicit-key", "http://explicit/v1", "explicit-model"
    )
