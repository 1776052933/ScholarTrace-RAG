"""Offline checks for the LangSmith application tracing adapter."""
from __future__ import annotations

from contextlib import contextmanager

import langsmith

from observability import tracing


def test_tracing_is_disabled_without_both_settings(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing._client() is None
    with tracing.trace_span("offline") as span:
        span.update(output={"ok": True})

    monkeypatch.setenv("LANGSMITH_API_KEY", "placeholder")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing._client() is None


def test_application_span_maps_to_langsmith_run(monkeypatch):
    observed = []

    class Run:
        def add_outputs(self, outputs):
            observed.append(("outputs", outputs))

        def add_metadata(self, metadata):
            observed.append(("metadata", metadata))

    @contextmanager
    def fake_trace(name, **kwargs):
        observed.append(("start", name, kwargs))
        yield Run()

    monkeypatch.setattr(tracing, "_client", lambda: object())
    monkeypatch.setattr(langsmith, "trace", fake_trace)
    with tracing.trace_span("rag.answer_generation", as_type="generation",
                            model="test-model", input={"chunk_ids": ["c1"]}) as span:
        span.update(output={"answer_length": 10}, usage_details={"input_tokens": 5})

    assert observed[0][1] == "rag.answer_generation"
    assert observed[0][2]["run_type"] == "chain"
    assert observed[0][2]["inputs"] == {"chunk_ids": ["c1"]}
    assert observed[0][2]["metadata"] == {"model": "test-model"}
    assert ("outputs", {"answer_length": 10}) in observed
    assert ("metadata", {"usage_details": {"input_tokens": 5}}) in observed


def test_real_sdk_preserves_parent_child_runs_locally(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "local-only-test-key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setattr(tracing, "_client_instance", None)

    # Local mode creates real SDK run trees but does not submit them.
    with langsmith.tracing_context(enabled="local"):
        with tracing.trace_span("research.job", input={"job_id": "test"}) as parent:
            with tracing.trace_span("search.plan", input={"round": 1}) as child:
                child.update(output={"paper_count": 0})
            assert child.run.parent_run_id == parent.run.id
            assert child.run.outputs == {"paper_count": 0}
