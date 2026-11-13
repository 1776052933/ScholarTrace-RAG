"""Fail-closed answer delivery: never show an unverified streamed draft."""
import json

from rag_reader.qa import PaperQA
from rag_reader.schema import DocumentChunk


class Retriever:
    def retrieve(self, paper_id, question, *, top_k):
        text = ("Paper A reports a score of 28.4." if paper_id == "a"
                else "Paper B reports a score of 41.8.")
        return [DocumentChunk(chunk_id=f"{paper_id}-{index}", paper_id=paper_id,
                              text=text, page=1) for index in (1, 2)]

    def retrieve_multiple(self, ids, question, *, top_k):
        return [chunk for paper_id in ids for chunk in self.retrieve(paper_id, question, top_k=top_k)]


class Model:
    model = "local-strict-fixture"

    def __init__(self, answer, sources, *, verdict="supported", fail_verifier=False):
        self.answer = answer
        self.sources = sources
        self.verdict = verdict
        self.fail_verifier = fail_verifier
        self.verify_calls = 0

    def _generated(self):
        return json.dumps({"answer": self.answer, "sources": self.sources, "confidence": 0.99})

    def chat(self, messages, **kwargs):
        if "核验器" not in messages[0]["content"]:
            return self._generated()
        self.verify_calls += 1
        if self.fail_verifier:
            raise RuntimeError("verifier unavailable")
        cited = json.loads(messages[-1]["content"])["cited_chunks"]
        return json.dumps({"decision": self.verdict,
                           "quotes": [{"chunk_id": item["chunk_id"], "quote": item["text"]}
                                      for item in cited] if self.verdict == "supported" else []})

    def stream_chat(self, messages, **kwargs):
        raw = self._generated()
        for start in range(0, len(raw), 7):
            yield raw[start:start + 7]


def _ask(model, ids=("a", "b")):
    return PaperQA(Retriever(), model, verify_support=True).workflow.invoke(list(ids), "Compare scores")


def test_reader_defaults_to_original_streaming_without_support_verifier(monkeypatch):
    from rag_reader.config import RAGSettings
    from rag_reader.pipeline import RAGReader

    monkeypatch.delenv("RAG_STRICT_SUPPORT_ENABLED", raising=False)
    model = Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "b-1"])
    reader = RAGReader(retriever=Retriever(), embedding_model=object(), vector_store=object(),
                       llm_client=model)
    assert RAGSettings.from_env().strict_support_enabled is False
    assert reader.qa.verify_support is False
    events = list(reader.qa.stream(["a", "b"], "Compare scores"))
    deltas = [event["data"]["text"] for event in events if event["event"] == "delta"]
    assert len(deltas) > 1
    assert "".join(deltas) == model.answer
    assert model.verify_calls == 0
    assert events[-1]["data"]["sources"] == ["a-1", "b-1"]


def test_supported_multi_paper_answer_is_released_after_verification():
    model = Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "b-1"])
    answer = _ask(model)
    assert answer.sources == ["a-1", "b-1"]
    assert model.verify_calls == 1

    events = list(PaperQA(Retriever(), model, verify_support=True).stream(
        ["a", "b"], "Compare scores"))
    assert events[-1]["event"] == "answer"
    assert events[-1]["data"]["answer"] == answer.answer
    assert "".join(event["data"]["text"] for event in events if event["event"] == "delta") == answer.answer
    assert next(i for i, event in enumerate(events) if event["event"] == "delta") > next(
        i for i, event in enumerate(events) if event["event"] == "status"
        and event["data"]["stage"] == "verifying")


def test_missing_paper_citation_refuses_without_verifier():
    model = Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1"])
    answer = _ask(model)
    assert answer.sources == [] and answer.confidence == 0
    assert model.verify_calls == 0


def test_unsupported_uncertain_or_wrong_number_never_reaches_stream():
    for model in (
        Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "b-1"], verdict="uncertain"),
        Model("Paper A scores 28.4 and Paper B scores 42.8.", ["a-1", "b-1"]),
        Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "fabricated"]),
        Model("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "b-1"], fail_verifier=True),
    ):
        events = list(PaperQA(Retriever(), model, verify_support=True).stream(
            ["a", "b"], "Compare scores"))
        answer = events[-1]["data"]
        deltas = "".join(event["data"]["text"] for event in events if event["event"] == "delta")
        assert answer["sources"] == [] and answer["confidence"] == 0
        assert deltas == answer["answer"]
        assert "28.4" not in deltas and "41.8" not in deltas and "42.8" not in deltas


def test_model_refusal_with_citations_becomes_uncited_refusal():
    model = Model("根据提供的论文片段无法确定。", ["a-1", "b-1"])
    answer = _ask(model)
    assert answer.sources == [] and answer.confidence == 0
    assert model.verify_calls == 0


def test_malformed_draft_or_verifier_result_fails_closed():
    class BadDraft(Model):
        def _generated(self):
            return '{"answer":"Wrong draft", "sources":'

    draft = BadDraft("ignored", [])
    events = list(PaperQA(Retriever(), draft, verify_support=True).stream(
        ["a", "b"], "Compare scores"))
    assert events[-1]["data"]["sources"] == []
    assert all("Wrong draft" not in event["data"].get("text", "") for event in events)
    assert draft.verify_calls == 0

    class BadVerdict(Model):
        def chat(self, messages, **kwargs):
            if "核验器" in messages[0]["content"]:
                self.verify_calls += 1
                return '{"decision":'
            return super().chat(messages, **kwargs)

    verifier = BadVerdict("Paper A scores 28.4 and Paper B scores 41.8.", ["a-1", "b-1"])
    answer = _ask(verifier)
    assert answer.sources == [] and answer.confidence == 0
    assert verifier.verify_calls == 1
