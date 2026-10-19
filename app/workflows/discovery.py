"""Paper discovery orchestration, before application-owned ingestion/transactions."""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class DiscoveryState(TypedDict, total=False):
    query: str
    publication_date_to: str | None
    analysis: Any
    decomposition: Any
    search_result: Any
    search_elapsed: float
    ranking_result: Any
    stages: dict[str, Any]


class DiscoveryGraph:
    def __init__(self, workflow, check_cancel):
        self.workflow = workflow
        self.check_cancel = check_cancel
        graph = StateGraph(DiscoveryState)
        for name in ("understand", "decompose", "search", "citations", "rank"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "understand")
        graph.add_edge("understand", "decompose")
        graph.add_edge("decompose", "search")
        graph.add_conditional_edges("search", self._after_search,
                                    {"citations": "citations", "rank": "rank"})
        graph.add_edge("citations", "rank")
        graph.add_edge("rank", END)
        # Search has its own bounded loop and checkpoint. This parent graph does
        # not replay external searches, registrations or ingestion side effects.
        self.graph = graph.compile()

    def _understand(self, state):
        from .contracts import _stage, _timed

        self.check_cancel()
        result, elapsed = _timed("query_understanding", self.workflow.query_understanding.run, state["query"])
        return {"analysis": result, "stages": {"query_understanding": _stage(result, elapsed)}}

    def _decompose(self, state):
        from .contracts import _merge_upstream_subqueries, _stage, _timed, _understanding_for_decomposition

        self.check_cancel()
        result, elapsed = _timed("query_decomposition", self.workflow.query_decomposition.run,
                                 _understanding_for_decomposition(state["analysis"]))
        result = _merge_upstream_subqueries(result, state["analysis"])
        return {"decomposition": result, "stages": {
            **state["stages"], "query_decomposition": _stage(result, elapsed)}}

    def _search(self, state):
        from .contracts import _model_data, _stage, _timed

        self.check_cancel()
        payload = _model_data(state["decomposition"])
        if state["publication_date_to"]:
            payload["publication_date_to"] = state["publication_date_to"]
        result, elapsed = _timed("search_agent", self.workflow.search_agent.run, payload)
        return {"search_result": result, "search_elapsed": elapsed, "stages": {
            **state["stages"], "search_agent": _stage(result, elapsed)}}

    def _after_search(self, state):
        self.check_cancel()
        return "citations" if self.workflow.citation_tracker is not None else "rank"

    def _citations(self, state):
        from .contracts import _replace_papers, _stage, _timed, _value

        self.check_cancel()
        expansion, elapsed = _timed("citation_tracking", self.workflow.citation_tracker.expand,
                                    state["query"], list(_value(state["search_result"], "papers") or []),
                                    publication_date_to=state["publication_date_to"])
        result = _replace_papers(state["search_result"], expansion.papers)
        return {"search_result": result, "stages": {
            **state["stages"], "citation_tracking": _stage(expansion.summary, elapsed),
            "search_agent": _stage(result, state["search_elapsed"])}}

    def _rank(self, state):
        from .contracts import _model_data, _stage, _timed

        self.check_cancel()
        result, elapsed = _timed("ranking_agent", self.workflow.ranking_agent.run,
                                 _model_data(state["analysis"]), state["search_result"])
        self.check_cancel()
        return {"ranking_result": result, "stages": {
            **state["stages"], "ranking_agent": _stage(result, elapsed)}}

    def invoke(self, query, publication_date_to=None, *, config=None):
        return self.graph.invoke({"query": query, "publication_date_to": publication_date_to}, config=config)
