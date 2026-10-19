"""Conversation persistence, grounded follow-ups and real incremental stream contracts."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.main import create_app
from app.services.conversations import ConversationBusyError, ConversationSettings
from app.storage.models import ConversationModel, ConversationTurnModel, utc_now
from literature_database.models import PaperModel
from rag_reader.llm_client import DeepSeekClient, DeepSeekResponseError
from rag_reader.qa import PaperQA, RAGAnswerParseError, _fit_evidence
from rag_reader.schema import DocumentChunk
from rag_reader.streaming import AnswerTextDecoder


class Retriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, paper_id, query, *, top_k):
        self.queries.append((paper_id, query))
        return [DocumentChunk(chunk_id=f"{paper_id}-{i}", paper_id=paper_id,
                              text=f"{paper_id} uses six attention layers. Original evidence {i}.",
                              page=i, metadata={"title": paper_id}) for i in (1, 2)]

    def retrieve_multiple(self, ids, query, *, top_k):
        return [c for paper_id in ids for c in self.retrieve(paper_id, query, top_k=top_k)][:top_k]


class Model:
    model = "local-test-model"

    def __init__(self):
        self.inputs = []
        self.rewrites = []
        self.closed = 0
        self.fail = False

    def chat(self, messages, **kwargs):
        self.rewrites.append(messages)
        return json.dumps({"question": "What datasets does the attention paper use?"})

    def stream_chat(self, messages, **kwargs):
        self.inputs.append(messages)
        payload = json.loads(messages[-1]["content"])
        data = json.dumps({"answer": "它使用六层。\nEvidence stays grounded 🧪.",
                           "sources": [payload["chunks"][0]["chunk_id"], "fabricated-id"],
                           "confidence": 0.9})
        try:
            for index in range(0, len(data), 5):
                if self.fail and index > 35:
                    raise RuntimeError("simulated provider disconnect")
                yield data[index:index + 5]
        finally:
            self.closed += 1


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "chroma")
    monkeypatch.setenv("CHAT_HISTORY_MAX_TURNS", "1")
    url = f"sqlite:///{(tmp_path / 'chat.sqlite3').as_posix()}"
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    app = create_app(database_url=url, data_root=tmp_path)
    with app.state.database.get_session() as session:
        session.add_all([PaperModel(paper_id=p, title=p, authors=[], embedding_status="COMPLETED")
                         for p in ("attention", "second")])
        session.commit()
    model, retriever = Model(), Retriever()
    app.state.questions.reader_for = lambda ids: NS(qa=PaperQA(retriever, model))
    return app, model, retriever, url, tmp_path


def parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads("\n".join(line[6:] for line in lines if line.startswith("data: ")))
        events.append({"event": event, "data": data})
    return events


def ask(client, conversation_id, question):
    response = client.post(f"/api/v1/conversations/{conversation_id}/messages/stream",
                           json={"question": question})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_history_followup_retrieval_citations_and_restart(setup):
    app, model, retriever, url, root = setup
    with TestClient(app) as client:
        created = client.post("/api/v1/conversations", json={"paper_ids": ["attention", "second"]})
        assert created.status_code == 201
        cid = created.json()["conversation_id"]
        first = ask(client, cid, "What is the architecture?")
        assert first[-1]["event"] == "done"
        assert len([e for e in first if e["event"] == "delta"]) > 2
        assert "".join(e["data"]["text"] for e in first if e["event"] == "delta") == first[-1]["data"]["answer"]["answer"]
        assert first[-1]["data"]["answer"]["sources"] == ["attention-1"]
        aid = first[-1]["data"]["answer_id"]
        assert client.get(f"/api/v1/answers/{aid}/evidence").json()[0]["page"] == 1
        assert "Original evidence" in client.get(f"/api/v1/answers/{aid}/export").text
        second = ask(client, cid, "它用了哪些数据集？")
        assert second[-1]["data"]["context"]["history_turns"] == 1
        assert any(query == "What datasets does the attention paper use?" for _, query in retriever.queries)
        assert model.inputs[-1][1]["role"] == "user"
        assert model.inputs[-1][2]["role"] == "assistant"
        assert model.rewrites[-1][1]["content"] == "What is the architecture?"
        ask(client, cid, "还有哪些限制？")
        assert model.rewrites[-1][1]["content"] == "它用了哪些数据集？"
        isolated = client.post("/api/v1/conversations", json={"paper_ids": ["attention"]}).json()["conversation_id"]
        events = ask(client, isolated, "Independent question")
        assert events[-1]["data"]["context"]["history_turns"] == 0
        assert len(model.inputs[-1]) == 2
        listed = client.get("/api/v1/conversations", params=[("paper_id", "second"), ("paper_id", "attention")]).json()
        assert [item["conversation_id"] for item in listed] == [cid]
    restarted = create_app(database_url=url, data_root=root)
    with TestClient(restarted) as client:
        snapshot = client.get(f"/api/v1/conversations/{cid}").json()
        assert [turn["status"] for turn in snapshot["turns"]] == ["completed"] * 3
        page = client.get(f"/api/v1/conversations/{cid}?limit=1").json()
        assert page["has_older"] and page["turns"][0]["turn_number"] == 3
        assert client.get(f"/api/v1/conversations/{cid}?before=3").json()["turns"][-1]["turn_number"] == 2


def test_failed_and_cancelled_turns_never_become_history(setup):
    app, model, _, _, _ = setup
    service = app.state.conversations
    cid = service.create(["attention"])["conversation_id"]
    model.fail = True
    with TestClient(app) as client:
        events = ask(client, cid, "Will fail")
        assert events[-1]["event"] == "error"
        assert not any(event["event"] == "done" for event in events)
        snapshot = client.get(f"/api/v1/conversations/{cid}").json()
        assert snapshot["active_turn_id"] is None
        assert snapshot["turns"][0]["status"] == "failed"
        assert snapshot["turns"][0]["answer"] is None
    model.fail = False
    turn = service.begin(cid, "Close during answer")
    iterator = service.stream(turn)
    while next(iterator)["event"] != "delta":
        pass
    iterator.close()
    assert model.closed == 2
    assert service.get(cid)["turns"][-1]["status"] == "cancelled"
    following = service.begin(cid, "Try again")
    assert following.history == []
    list(service.stream(following))
    assert service.get(cid)["turns"][-1]["status"] == "completed"


def test_concurrent_turn_and_expired_lease_are_safe(setup):
    app, _, _, _, _ = setup
    service = app.state.conversations
    cid = service.create(["attention"])["conversation_id"]
    first = service.begin(cid, "First")
    with pytest.raises(ConversationBusyError):
        service.begin(cid, "Concurrent")
    with TestClient(app) as client:
        assert client.post(f"/api/v1/conversations/{cid}/messages/stream", json={"question": "Concurrent"}).status_code == 409
    with app.state.database.get_session() as session:
        session.get(ConversationModel, cid).active_since = utc_now() - timedelta(hours=1)
        session.commit()
    second = service.begin(cid, "Recovered")
    assert service.get(cid)["turns"][0]["status"] == "interrupted"
    with pytest.raises(ConversationBusyError):
        service._complete(first, "old", {"answer": "old"})
    service.fail(first, "cancelled", "old disconnected")
    assert service.get(cid)["active_turn_id"] == second.turn_id
    list(service.stream(second))


def test_api_validates_scope_and_question_before_streaming(setup):
    app, *_ = setup
    with TestClient(app) as client:
        assert client.post("/api/v1/conversations", json={"paper_ids": ["attention", "attention"]}).status_code == 422
        assert client.post("/api/v1/conversations", json={"paper_ids": ["missing"]}).status_code == 404
        assert client.get("/api/v1/conversations/missing").status_code == 404
        cid = client.post("/api/v1/conversations", json={"paper_ids": ["attention"]}).json()["conversation_id"]
        assert client.post(f"/api/v1/conversations/{cid}/messages/stream", json={"question": "  "}).status_code == 422
        assert client.post(f"/api/v1/conversations/{cid}/messages/stream", json={"question": "x", "history": []}).status_code == 422


def test_history_and_entire_prompt_have_explicit_budgets(setup):
    app, model, *_ = setup
    service = app.state.conversations
    service.settings = ConversationSettings(history_max_chars=100, prompt_max_chars=4000)
    cid = service.create(["attention"])["conversation_id"]
    first = service.begin(cid, "x" * 200)
    list(service.stream(first))
    second = service.begin(cid, "follow up")
    assert second.history == []
    list(service.stream(second))
    assert len(json.dumps(model.inputs[-1], ensure_ascii=False)) <= 4000
    chunk = DocumentChunk(chunk_id="a", paper_id="a", text="word\n" * 4000)
    messages, used = _fit_evidence("system", {"chunks": []}, [], [chunk], 4000)
    assert len(json.dumps(messages, ensure_ascii=False)) <= 4000
    assert 0 < len(used[0].text) < len(chunk.text)


@pytest.mark.parametrize("text", [
    '中文 "quoted"\\path\nnext 🧪',
    "a closing brace } and an answer key: \"answer\": \"fake\"",
])
def test_json_answer_decoder_handles_every_escape_boundary(text):
    content = json.dumps({"question": 'question with "answer": "fake"', "answer": text,
                          "sources": [], "confidence": 0.5}, ensure_ascii=True)
    decoder = AnswerTextDecoder()
    assert "".join(decoder.feed(char) for char in content) == text


def test_provider_stream_yields_before_completion_and_closes():
    steps = []

    class Stream:
        def __iter__(self):
            steps.append("first")
            yield NS(choices=[NS(delta=NS(content="hello"), finish_reason=None)], usage=None)
            steps.append("second")
            yield NS(choices=[NS(delta=NS(content=" world"), finish_reason="stop")], usage=None)
            yield NS(choices=[], usage={"prompt_tokens": 7, "completion_tokens": 2})

        def close(self):
            steps.append("closed")

    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return Stream()
    client = DeepSeekClient(api_key="local-fixture", _client=NS(chat=NS(completions=NS(create=create))))
    iterator = client.stream_chat([{"role": "user", "content": "hello"}])
    assert next(iterator) == "hello" and steps == ["first"]
    assert "".join(iterator) == " world"
    assert calls[0]["stream"] is True
    assert client.get_token_usage()["last"]["total_tokens"] == 9
    assert steps[-1] == "closed"


def test_truncated_provider_output_is_not_a_successful_stream():
    class Stream:
        def __iter__(self):
            yield NS(choices=[NS(delta=NS(content="partial"), finish_reason="length")], usage=None)
        def close(self):
            pass
    client = DeepSeekClient(api_key="local-fixture", _client=NS(chat=NS(completions=NS(create=lambda **kw: Stream()))))
    with pytest.raises(DeepSeekResponseError, match="length"):
        list(client.stream_chat([{"role": "user", "content": "hello"}]))


def test_sse_reaches_http_client_before_generation_finishes(setup, monkeypatch):
    """A real socket avoids TestClient's buffering and proves incremental API → UI delivery."""
    import socket
    import threading
    import time
    import uvicorn
    from Auto_Paper_UI.api_client import stream_conversation

    app, model, *_ = setup
    release = threading.Event()
    reached_tail = threading.Event()
    original = model.stream_chat

    def gated(messages, **kwargs):
        stream = original(messages, **kwargs)
        count = 0
        try:
            for delta in stream:
                yield delta
                count += 1
                if count == 6:
                    reached_tail.set()
                    if not release.wait(10):
                        raise RuntimeError("test did not receive initial deltas")
        finally:
            stream.close()

    model.stream_chat = gated
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        monkeypatch.setenv("AUTO_PAPER_API_URL", f"http://127.0.0.1:{port}")
        cid = app.state.conversations.create(["attention"])["conversation_id"]
        iterator = stream_conversation(cid, "A streaming question")
        initial = []
        while not initial or initial[-1]["event"] != "delta":
            initial.append(next(iterator))
        assert not release.is_set()
        assert app.state.conversations.get(cid)["turns"][0]["status"] == "generating"
        release.set()
        remaining = list(iterator)
        assert remaining[-1]["event"] == "done"
        assert app.state.conversations.get(cid)["turns"][0]["status"] == "completed"
        release.clear()
        cid2 = app.state.conversations.create(["attention"])["conversation_id"]
        interrupted = stream_conversation(cid2, "Disconnect before completion")
        while next(interrupted)["event"] != "delta":
            pass
        interrupted.close()
        release.set()
        deadline = time.monotonic() + 5
        snapshot = app.state.conversations.get(cid2)
        while snapshot["active_turn_id"] and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = app.state.conversations.get(cid2)
        assert snapshot["active_turn_id"] is None
        assert snapshot["turns"][0]["status"] == "cancelled"
        assert snapshot["turns"][0]["answer"] is None
    finally:
        release.set()
        server.should_exit = True
        thread.join(10)
        sock.close()
    assert not thread.is_alive()
