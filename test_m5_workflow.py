"""M5 graph routing, bounded evidence expansion, and MCP API contracts."""
from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager

from rag_reader.evidence_graph import EvidenceGraph
from app.workflows.search import GraphSearchAgent
from rag_reader.schema import DocumentChunk
from search_agent.config import SearchAgentConfig
from search_agent.pipeline import SearchAgent
from search_agent.schema import (
    PaperMetadata, SearchEvaluation, SearchInput, SearchResult, SearchStrategy, SearchTask,
)


def _plan(round_number: int) -> SearchStrategy:
    return SearchStrategy(
        original_query="graph rag", search_round=round_number, strategy="search",
        queries=[SearchTask(id=f"q{round_number}", sub_query_index=0, query="graph rag",
                            source="openalex", purpose="test")],
    )


class _Planner:
    def plan(self, _search_input, round_number):
        return _plan(round_number)


class _Searcher:
    def __init__(self):
        self.calls = 0

    def search(self, _plan, **_kwargs):
        self.calls += 1
        return SearchResult(papers=[PaperMetadata(
            title=f"Paper {self.calls}", source="openalex",
            openalex_id=f"W{self.calls}",
        )], total=1, query_used="graph rag")


class _Evaluator:
    def __init__(self, refine=True):
        self.refine = refine

    def evaluate_with_rules(self, _result, _plan):
        return SearchEvaluation(relevance_score=0.5, quality_score=0.5,
                                coverage_score=0.5, need_refine=self.refine)


class _Reflector:
    def reflect(self, plan, _result, _evaluation):
        return _plan(plan.search_round + 1)


def test_search_graph_caps_rounds_and_persists_state(tmp_path):
    searcher = _Searcher()
    agent = SearchAgent(config=SearchAgentConfig(max_rounds=2, use_llm=False),
                        planner=_Planner(), searcher=searcher,
                        evaluator=_Evaluator(), reflector=_Reflector())
    path = tmp_path / "graph.sqlite3"
    result = GraphSearchAgent(agent, checkpoint_path=path).run(SearchInput(
        original_query="graph rag", sub_queries=[{"query": "graph rag"}],
    ))
    assert result.rounds_completed == searcher.calls == 2
    assert result.stopped_reason == "达到最大搜索轮次"
    assert not result.iterations[-1].reflection.should_continue
    assert result.graph_thread_id and len(result.papers) == 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] >= 2


def test_search_graph_stops_when_quality_gate_passes(tmp_path):
    searcher = _Searcher()
    agent = SearchAgent(config=SearchAgentConfig(max_rounds=3, use_llm=False),
                        planner=_Planner(), searcher=searcher,
                        evaluator=_Evaluator(refine=False), reflector=_Reflector())
    result = GraphSearchAgent(agent, checkpoint_path=tmp_path / "graph.sqlite3").run({
        "original_query": "graph rag", "sub_queries": [{"query": "graph rag"}],
    })
    assert result.rounds_completed == searcher.calls == 1
    assert result.stopped_reason == "搜索结果质量达到阈值"
    assert not result.iterations[0].reflection.should_continue


def test_search_graph_stops_after_round_when_time_budget_expires(tmp_path, monkeypatch):
    from app.workflows import search as module

    ticks = iter([0.0, 1000.0])
    monkeypatch.setattr(module, "monotonic", lambda: next(ticks))
    searcher = _Searcher()
    agent = SearchAgent(config=SearchAgentConfig(max_rounds=3, use_llm=False),
                        planner=_Planner(), searcher=searcher,
                        evaluator=_Evaluator(), reflector=_Reflector())
    result = GraphSearchAgent(agent, checkpoint_path=tmp_path / "graph.sqlite3").run({
        "original_query": "graph rag", "sub_queries": [{"query": "graph rag"}],
    })
    assert result.rounds_completed == searcher.calls == 1
    assert result.stopped_reason == "搜索超时，已返回当前候选论文"
    assert not result.iterations[0].reflection.should_continue


def test_application_workflow_uses_search_graph(tmp_path):
    from app.services.workflow import AutoPaperWorkflow

    agent = SearchAgent(config=SearchAgentConfig(max_rounds=2, use_llm=False),
                        planner=_Planner(), searcher=_Searcher(),
                        evaluator=_Evaluator(), reflector=_Reflector())
    workflow = AutoPaperWorkflow(
        query_understanding=object(), query_decomposition=object(),
        search_agent=agent, ranking_agent=object(), literature_service=object(),
        enable_rag=False, enable_citation_tracking=False,
        chroma_directory=str(tmp_path / "chroma"),
    )
    assert isinstance(workflow.search_agent, GraphSearchAgent)
    assert workflow.search_agent.checkpoint_path == tmp_path / "search_graph.sqlite3"


def test_search_graph_falls_back_to_rule_evaluation(tmp_path):
    class Evaluator(_Evaluator):
        def evaluate(self, _result, _plan):
            raise RuntimeError("model unavailable")

    agent = SearchAgent(config=SearchAgentConfig(max_rounds=2, use_llm=True),
                        planner=_Planner(), searcher=_Searcher(),
                        evaluator=Evaluator(refine=False), reflector=_Reflector())
    result = GraphSearchAgent(agent, checkpoint_path=tmp_path / "graph.sqlite3").run({
        "original_query": "graph rag", "sub_queries": [{"query": "graph rag"}],
    })
    assert result.rounds_completed == 1
    assert result.stopped_reason == "搜索结果质量达到阈值"


def _chunk(paper_id: str, number: int) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"{paper_id}-{number}", paper_id=paper_id,
                         text=f"Evidence {number}", page=number)


class _Retriever:
    def __init__(self):
        self.calls = []

    def retrieve_multiple(self, ids, question, *, top_k):
        self.calls.append(("global", ids, top_k))
        return [_chunk("a", 1), _chunk("a", 2)]

    def retrieve(self, paper_id, question, *, top_k):
        self.calls.append((paper_id, top_k))
        return [_chunk(paper_id, number) for number in range(1, min(top_k, 5) + 1)]


def test_evidence_graph_expands_only_undercovered_paper(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_MAX_EXPANSIONS", "1")
    retriever = _Retriever()
    chunks, report = EvidenceGraph(retriever, top_k=2).run(["a", "b"], "compare")
    assert {chunk.paper_id for chunk in chunks} == {"a", "b"}
    assert report == {"sufficient": True, "stop_reason": "covered", "expansions": 1,
                      "missing_paper_ids": [], "absent_paper_ids": [], "chunk_count": 4}
    assert ("a", 4) not in retriever.calls
    assert ("b", 4) in retriever.calls


def test_evidence_graph_reports_limit_when_paper_has_no_passages(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_MAX_EXPANSIONS", "1")
    retriever = _Retriever()
    retriever.retrieve = lambda paper_id, question, *, top_k: []
    _, report = EvidenceGraph(retriever, top_k=2).run(["a", "b"], "compare")
    assert report["stop_reason"] == "expansion_limit"
    assert report["missing_paper_ids"] == ["b"]
    assert report["absent_paper_ids"] == ["b"]
    assert report["expansions"] == 1


def test_balanced_evidence_reserves_slots_for_each_paper():
    retriever = _Retriever()
    chunks, report = EvidenceGraph(retriever, top_k=5,
                                   strategy="balanced", max_chunks=10).run(["a", "b"], "compare")
    assert len(chunks) == 6
    assert [chunk.paper_id for chunk in chunks] == ["a"] * 3 + ["b"] * 3
    assert report["expansions"] == 0
    assert retriever.calls == [("a", 3), ("b", 3)]


def test_balanced_evidence_caps_five_papers_at_ten_chunks():
    retriever = _Retriever()
    ids = ["a", "b", "c", "d", "e"]
    chunks, _ = EvidenceGraph(retriever, top_k=5,
                              strategy="balanced", max_chunks=10).run(ids, "compare")
    assert len(chunks) == 10
    assert {chunk.paper_id for chunk in chunks} == set(ids)
    assert all(top_k == 2 for _, top_k in retriever.calls)


def test_mcp_tools_reuse_api(monkeypatch):
    from app.mcp import server

    calls = []

    def fake_api(method, path, payload=None):
        calls.append((method, path, payload))
        if path == "/api/v1/jobs":
            return {"job_id": "job-1", "status": "queued", "stage": None}
        if path == "/api/v1/questions/multi":
            return {"answer_id": "answer-1", "answer": {"sources": ["chunk-1"]}}
        if path.endswith("/evidence"):
            return [{"chunk_id": "chunk-1", "page": 2}]
        raise AssertionError(path)

    monkeypatch.setattr(server, "_api", fake_api)
    assert server.search_papers("test")["job_id"] == "job-1"
    assert calls[0][2]["download_papers"] is False
    assert server.ask_papers(["a", "b"], "compare")["answer_id"] == "answer-1"
    assert server.get_evidence("answer-1")[0]["page"] == 2


def test_mcp_client_can_discover_and_call_tools(monkeypatch):
    import anyio
    from mcp import Client
    from app.mcp import server

    monkeypatch.setattr(server, "_api", lambda method, path, payload=None: {
        "job_id": "job-1", "status": "queued", "stage": None,
    })

    async def run():
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            assert {item.name for item in tools.tools} == {
                "search_papers", "get_job_status", "start_ingestion", "ask_papers", "get_evidence",
            }
            result = await client.call_tool("search_papers", {"query": "graph rag"})
            assert not result.is_error
            assert result.structured_content["job_id"] == "job-1"

    anyio.run(run)


def test_rag_trace_records_nodes_and_token_usage_without_paper_text(monkeypatch):
    from observability import tracing
    from rag_reader.qa import PaperQA

    observations = []

    class Span:
        def __init__(self, entry):
            self.entry = entry

        def update(self, **kwargs):
            self.entry["updates"].append(kwargs)

    class Client:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            entry = {"start": kwargs, "updates": []}
            observations.append(entry)
            yield Span(entry)

    monkeypatch.setattr(tracing, "_client", lambda: Client())

    class Retriever:
        def retrieve(self, paper_id, question, *, top_k):
            return [_chunk(paper_id, 1), _chunk(paper_id, 2)]

    class LLM:
        model = "test-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps({"answer": "Supported", "sources": ["a-1"], "confidence": 0.8})

        def get_token_usage(self):
            return {"last": {"prompt_tokens": 11, "completion_tokens": 7}}

    answer = PaperQA(Retriever(), llm_client=LLM()).ask("a", "question")
    assert answer.sources == ["a-1"]
    names = [entry["start"]["name"] for entry in observations]
    assert "rag.evidence_graph" in names and "rag.evidence.assess" in names
    generation = next(entry for entry in observations
                      if entry["start"]["name"] == "rag.answer_generation")
    assert generation["updates"][0]["usage_details"] == {
        "input_tokens": 11, "output_tokens": 7,
    }
    assert "Evidence 1" not in str(observations)
