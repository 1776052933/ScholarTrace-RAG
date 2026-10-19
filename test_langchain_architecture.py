"""Behavioral contracts for LangChain adapters and the complete AI workflows."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ai_components.chains import generate_text, text_chain
from ai_components.models import as_chat_model
from app.services.workflow import AutoPaperWorkflow, WorkflowCancelled
from app.workflows.discovery import DiscoveryGraph
from rag_reader.langchain_retriever import ScopedPaperRetriever, to_chunk, to_document
from rag_reader.qa import PaperQA, RAGAnswerParseError
from rag_reader.schema import DocumentChunk


class Calls(BaseCallbackHandler):
    def __init__(self):
        self.models = 0
        self.retrievals = 0
        self.tokens = []
        self.nodes = []

    def on_chat_model_start(self, *args, **kwargs):
        self.models += 1

    def on_retriever_start(self, *args, **kwargs):
        self.retrievals += 1

    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.nodes.append(kwargs.get("name"))


def chunk(paper_id="a", number=1):
    return DocumentChunk(chunk_id=f"{paper_id}-{number}", paper_id=paper_id,
                         text="The method uses six layers.", page=number, section="Method",
                         metadata={"title": "A paper", "retrieval_score": 0.8,
                                   "parser": "docling", "nested": {"table": True}})


class Retrieval:
    def retrieve(self, paper_id, query, *, top_k):
        return [chunk(paper_id, i) for i in (1, 2)]

    def retrieve_multiple(self, ids, query, *, top_k):
        return [chunk(p, i) for p in ids for i in (1, 2)][:top_k]


class Transport:
    model = "local-fixture"

    def __init__(self):
        self.inputs = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.inputs.append((messages, kwargs))
        return json.dumps({"answer": "Six layers.", "sources": ["a-1", "foreign-9"], "confidence": 0.9})

    def stream_chat(self, messages, **kwargs):
        try:
            yield from self.chat(messages, **kwargs)
        finally:
            self.closed = True

    def get_token_usage(self):
        return {"last": {"prompt_tokens": 7, "completion_tokens": 4}}


def test_model_chain_preserves_history_options_callbacks_and_usage():
    transport, calls = Transport(), Calls()
    messages = [("system", "Return {JSON}"), ("human", "first"),
                ("ai", "earlier answer"), ("human", "followup")]
    output = generate_text(transport, messages, temperature=0.2, max_tokens=99,
                           config={"callbacks": [calls]})
    assert json.loads(output)["answer"] == "Six layers."
    sent, options = transport.inputs[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[0]["content"] == "Return {JSON}"
    assert options == {"temperature": 0.2, "max_tokens": 99}
    assert calls.models == 1
    message = as_chat_model(transport).invoke(messages)
    assert message.usage_metadata == {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11}
    assert "transport" not in as_chat_model(transport).model_dump()


def test_native_langchain_model_is_injectable_and_chain_streams():
    model = FakeListChatModel(responses=["abcdef"])
    assert as_chat_model(model) is model
    calls = Calls()
    output = list(text_chain(model).stream({"messages": [("human", "question")]},
                                           config={"callbacks": [calls]}))
    assert "".join(output) == "abcdef" and len(output) > 1
    assert "".join(calls.tokens) == "abcdef"


def test_provider_stream_closes_through_lcel_on_consumer_cancel():
    transport = Transport()
    stream = text_chain(transport).stream({"messages": [("human", "question")]})
    assert next(stream) == "{"
    stream.close()
    assert transport.closed


def test_retriever_document_roundtrip_batch_and_scope_are_strict():
    original = chunk()
    document = to_document(original)
    assert to_chunk(document) == original
    assert document.metadata["page"] == 1
    assert document.metadata["retrieval_score"] == 0.8
    calls = Calls()
    retriever = ScopedPaperRetriever(backend=Retrieval(), paper_ids=("a",), top_k=2)
    results = retriever.batch(["method", "layers"], config={"callbacks": [calls]})
    assert calls.retrievals == 2
    assert all({d.metadata["paper_id"] for d in docs} == {"a"} for docs in results)
    retriever.backend = NS(retrieve=lambda *args, **kwargs: [chunk("foreign")])
    with pytest.raises(ValueError, match="outside"):
        retriever.invoke("question")
    with pytest.raises(ValueError, match="non-empty"):
        ScopedPaperRetriever(backend=Retrieval(), paper_ids=())


def test_answer_graph_invoke_stream_share_validation_and_propagate_callbacks():
    qa, calls = PaperQA(Retrieval(), Transport()), Calls()
    answer = qa.ask("a", "How many layers?", config={"callbacks": [calls]})
    events = list(qa.stream(["a"], "How many layers?", config={"callbacks": [calls]}))
    assert answer.model_dump(mode="json") == events[-1]["data"]
    assert answer.sources == ["a-1"] and answer.citations[0].page == 1
    assert "".join(e["data"]["text"] for e in events if e["event"] == "delta") == answer.answer
    assert calls.models == 2 and calls.retrievals >= 2
    assert {"rewrite", "retrieve", "prepare", "generate", "validate"} <= set(calls.nodes)


def test_answer_graph_accepts_native_model_and_never_calls_model_without_evidence():
    model = FakeListChatModel(responses=['{"answer":"Six layers.","sources":["a-1"],"confidence":0.9}'])
    qa = PaperQA(Retrieval(), model)
    assert qa.ask("a", "question").sources == ["a-1"]
    assert list(qa.stream(["a"], "question"))[-1]["data"]["sources"] == ["a-1"]
    calls = Calls()
    empty = NS(retrieve=lambda *args, **kwargs: [])
    answer = PaperQA(empty, model).ask("a", "question", config={"callbacks": [calls]})
    assert answer.confidence == 0 and answer.sources == [] and calls.models == 0
    assert "abstain" in calls.nodes and "generate" not in calls.nodes


def test_invalid_stream_never_emits_validated_answer():
    class Invalid(Transport):
        def chat(self, messages, **kwargs):
            return '{"answer":"Partial answer", "sources":'

    transport = Invalid()
    events = []
    with pytest.raises(RAGAnswerParseError):
        for event in PaperQA(Retrieval(), transport).stream(["a"], "question"):
            events.append(event)
    assert any(e["event"] == "delta" for e in events)
    assert not any(e["event"] == "answer" for e in events)
    assert transport.closed


@pytest.mark.parametrize("citations", [False, True])
def test_discovery_graph_branches_preserve_time_filter_and_stage_outputs(citations):
    calls, order = Calls(), []

    class Stage:
        def __init__(self, name, result):
            self.name, self.result = name, result

        def run(self, *args):
            order.append(self.name)
            if self.name == "search":
                assert args[0]["publication_date_to"] == "2025-01-01"
            if self.name == "rank":
                assert len(args[1]["papers"]) == (2 if citations else 1)
            return self.result

    def expand(query, papers, **kwargs):
        order.append("citations")
        assert kwargs["publication_date_to"] == "2025-01-01"
        return NS(papers=papers + [{"paper_id": "b", "title": "B"}], summary={"added": 1})

    workflow = NS(
        query_understanding=Stage("understand", {"query": "layers"}),
        query_decomposition=Stage("decompose", {"sub_queries": []}),
        search_agent=Stage("search", {"papers": [{"paper_id": "a", "title": "A"}]}),
        citation_tracker=NS(expand=expand) if citations else None,
        ranking_agent=Stage("rank", {"papers": [{"paper_id": "a", "title": "A"}]}),
    )
    result = DiscoveryGraph(workflow, lambda: None).invoke("layers", "2025-01-01",
                                                          config={"callbacks": [calls]})
    assert order == ["understand", "decompose", "search"] + (["citations"] if citations else []) + ["rank"]
    assert ("citation_tracking" in result["stages"]) is citations
    assert {"understand", "decompose", "search", "rank"} <= set(calls.nodes)


def test_cancelled_discovery_does_not_register_or_ingest():
    from test_m1_service import Stage, Literature

    stopped = False

    class Search(Stage):
        def run(self, *args):
            nonlocal stopped
            stopped = True
            return self.result

    class NoWrites(Literature):
        def register_ranked_papers(self, *args, **kwargs):
            pytest.fail("cancelled discovery reached persistence")

    workflow = AutoPaperWorkflow(
        query_understanding=Stage({"query": "layers"}), query_decomposition=Stage({"sub_queries": []}),
        search_agent=Search({"papers": []}), ranking_agent=Stage({"papers": []}),
        literature_service=NoWrites(), enable_rag=False, enable_citation_tracking=False,
    )
    with pytest.raises(WorkflowCancelled):
        workflow.run("layers", download_papers=False, ingest_rag=False, should_cancel=lambda: stopped)


def test_discovery_runs_real_search_subgraph_with_checkpoint(tmp_path):
    from test_m1_service import Stage, Literature
    from test_m5_workflow import _Planner, _Searcher, _Evaluator, _Reflector
    from search_agent.config import SearchAgentConfig
    from search_agent.pipeline import SearchAgent

    agent = SearchAgent(config=SearchAgentConfig(max_rounds=2, use_llm=False),
                        planner=_Planner(), searcher=_Searcher(),
                        evaluator=_Evaluator(refine=False), reflector=_Reflector())
    workflow = AutoPaperWorkflow(
        query_understanding=Stage({"query": "graph rag"}),
        query_decomposition=Stage({"original_query": "graph rag", "sub_queries": [{"query": "graph rag"}]}),
        search_agent=agent, ranking_agent=Stage({"papers": []}), literature_service=Literature(),
        enable_rag=False, enable_citation_tracking=False, chroma_directory=str(tmp_path / "chroma"),
    )
    result = workflow.run("graph rag", download_papers=False, ingest_rag=False)
    search = result["stages"]["search_agent"]["result"]
    assert search["rounds_completed"] == 1 and search["graph_thread_id"]
    assert (tmp_path / "search_graph.sqlite3").is_file()


def test_embeddings_use_same_vectors_for_documents_and_queries(monkeypatch):
    from langchain_core.embeddings import Embeddings
    from rag_reader.embedding import EmbeddingModel

    def encode(texts, **kwargs):
        return [1.0, 0.0] if isinstance(texts, str) else [[1.0, 0.0] for text in texts]

    monkeypatch.setattr(EmbeddingModel, "_get_model", lambda self: NS(encode=encode))
    model = EmbeddingModel()
    assert isinstance(model, Embeddings)
    assert model.embed_query("layers") == model.embed_documents(["layers"])[0]
    assert model.embed_chunks([chunk()]) == model.embed_documents([chunk().text])
