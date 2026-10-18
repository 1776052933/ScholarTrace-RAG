"""Durable LangGraph orchestration of the existing paper search components."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, TypedDict
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from observability.tracing import trace_span
from search_agent.pipeline import SearchAgent
from search_agent.schema import (
    Paper, Reflection, SearchAgentResult, SearchEvaluation, SearchInput,
    SearchIteration, SearchPlan, SearchResult, SearchStrategy,
)
from search_agent.searcher import deduplicate_papers, to_internal_paper


logger = logging.getLogger(__name__)


class SearchState(TypedDict, total=False):
    thread_id: str
    deadline: float
    search_input: dict[str, Any]
    round_number: int
    papers: list[dict[str, Any]]
    plan: dict[str, Any]
    pending_plan: dict[str, Any] | None
    search_result: dict[str, Any]
    cumulative_result: dict[str, Any]
    new_count: int
    evaluation: dict[str, Any]
    iterations: list[dict[str, Any]]
    stopped_reason: str
    final_result: dict[str, Any]


def _plan(value: Mapping[str, Any]) -> SearchPlan:
    queries = value.get("queries") or []
    cls = SearchStrategy if queries and isinstance(queries[0], Mapping) and "id" in queries[0] else SearchPlan
    return cls.model_validate(value)


class GraphSearchAgent:
    """Keep planner/searcher/evaluator/reflector as domain components; graph owns the loop."""

    def __init__(self, agent: SearchAgent, *, checkpoint_path: str | Path) -> None:
        self.agent = agent
        self.checkpoint_path = Path(checkpoint_path)
        self.max_rounds = agent.config.max_rounds
        self.max_seconds = max(1.0, float(os.getenv("SEARCH_GRAPH_MAX_SECONDS", "300")))

    def _graph(self):
        graph = StateGraph(SearchState)
        for name, action in (
            ("plan", self._plan), ("search", self._search),
            ("evaluate", self._evaluate), ("reflect", self._reflect),
            ("finish", self._finish),
        ):
            def traced(state: SearchState, *, _name=name, _action=action):
                with trace_span(f"search.{_name}", input={"round": state["round_number"]}) as span:
                    result = _action(state)
                    span.update(output={"round": state["round_number"], "paper_count": len(state.get("papers", []))})
                    return result
            graph.add_node(name, traced)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "evaluate")
        graph.add_edge("evaluate", "reflect")
        graph.add_conditional_edges("reflect", self._route, {"again": "plan", "done": "finish"})
        graph.add_edge("finish", END)
        return graph

    def _plan(self, state: SearchState) -> dict[str, Any]:
        round_number = state["round_number"]
        logger.info("开始第 %d 轮论文搜索", round_number)
        pending = state.get("pending_plan")
        plan = _plan(pending) if pending else self.agent.planner.plan(
            SearchInput.model_validate(state["search_input"]), round_number
        )
        return {"plan": plan.model_dump(mode="json"), "pending_plan": None}

    def _search(self, state: SearchState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        search_input = SearchInput.model_validate(state["search_input"])
        found = self.agent.searcher.search(
            plan, publication_date_to=getattr(search_input, "publication_date_to", None)
        )
        previous = [Paper.model_validate(item) for item in state.get("papers", [])]
        merged = deduplicate_papers([*previous, *(to_internal_paper(item) for item in found.papers)])
        cumulative = SearchResult(
            papers=merged, total=len(merged), query_used=found.query_used,
            query_sources=found.query_sources,
        )
        return {
            "papers": [paper.model_dump(mode="json") for paper in merged],
            "new_count": len(merged) - len(previous),
            "search_result": found.model_dump(mode="json"),
            "cumulative_result": cumulative.model_dump(mode="json"),
        }

    def _evaluate(self, state: SearchState) -> dict[str, Any]:
        result = SearchResult.model_validate(state["cumulative_result"])
        plan = _plan(state["plan"])
        try:
            evaluation = (
                self.agent.evaluator.evaluate(result, plan)
                if self.agent.config.use_llm
                else self.agent.evaluator.evaluate_with_rules(result, plan)
            )
        except Exception:
            logger.warning("DeepSeek 搜索评估失败，使用规则回退", exc_info=True)
            evaluation = self.agent.evaluator.evaluate_with_rules(result, plan)
        return {"evaluation": evaluation.model_dump(mode="json")}

    def _reflect(self, state: SearchState) -> dict[str, Any]:
        plan = _plan(state["plan"])
        result = SearchResult.model_validate(state["cumulative_result"])
        evaluation = SearchEvaluation.model_validate(state["evaluation"])
        next_plan = self.agent.reflector.reflect(plan, result, evaluation)
        timed_out = monotonic() >= state["deadline"]
        continue_search = (evaluation.need_refine and next_plan is not None and state["round_number"] < self.max_rounds
                           and not timed_out)
        reason = (
            "搜索结果质量达到阈值" if not evaluation.need_refine
            else "搜索超时，已返回当前候选论文" if timed_out
            else "达到最大搜索轮次" if state["round_number"] >= self.max_rounds
            else "搜索覆盖度已达到停止阈值" if next_plan is None
            else next_plan.strategy
        )
        reflection = Reflection(
            should_continue=continue_search,
            reason=reason,
            add_keywords=[item.query for item in next_plan.queries] if next_plan else [],
            source_adjustments=(list(dict.fromkeys(item.source for item in next_plan.queries)) if next_plan else []),
        )
        iteration = SearchIteration(
            round_number=state["round_number"], plan=plan,
            new_paper_count=state["new_count"], total_paper_count=len(state["papers"]),
            evaluation=evaluation, reflection=reflection,
        )
        iterations = [*state.get("iterations", []), iteration.model_dump(mode="json")]
        return {
            "iterations": iterations,
            "pending_plan": next_plan.model_dump(mode="json") if continue_search else None,
            "stopped_reason": reason,
            "round_number": state["round_number"] + 1 if continue_search and evaluation.need_refine else state["round_number"],
        }

    @staticmethod
    def _route(state: SearchState) -> str:
        evaluation = state["evaluation"]
        return "again" if evaluation["need_refine"] and state.get("pending_plan") else "done"

    @staticmethod
    def _finish(state: SearchState) -> dict[str, Any]:
        papers = sorted(
            [Paper.model_validate(item) for item in state.get("papers", [])],
            key=lambda paper: (paper.relevance_score, paper.citation_count or 0), reverse=True,
        )
        result = SearchAgentResult(
            original_query=state["search_input"]["original_query"],
            rounds_completed=len(state["iterations"]),
            stopped_reason=state["stopped_reason"], papers=papers,
            graph_thread_id=state["thread_id"],
            iterations=[SearchIteration.model_validate(item) for item in state["iterations"]],
            final_evaluation=SearchEvaluation.model_validate(state["evaluation"]),
        )
        return {"final_result": result.model_dump(mode="json")}

    def run(self, decomposition_result: SearchInput | Mapping[str, Any]) -> SearchAgentResult:
        search_input = (
            decomposition_result if isinstance(decomposition_result, SearchInput)
            else SearchInput.model_validate(dict(decomposition_result))
        )
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        thread_id = uuid4().hex
        initial: SearchState = {
            "thread_id": thread_id,
            "deadline": monotonic() + self.max_seconds,
            "search_input": search_input.model_dump(mode="json"),
            "round_number": 1, "papers": [], "iterations": [],
        }
        with trace_span("research.search_graph", input={"thread_id": thread_id}) as span:
            with SqliteSaver.from_conn_string(str(self.checkpoint_path)) as saver:
                graph = self._graph().compile(checkpointer=saver)
                state = graph.invoke(
                    initial, config={"configurable": {"thread_id": thread_id},
                                     "recursion_limit": 4 * self.max_rounds + 5},
                )
            span.update(output={"thread_id": thread_id, "rounds": len(state["iterations"]),
                                "paper_count": len(state["papers"]), "stop_reason": state["stopped_reason"]})
        return SearchAgentResult.model_validate(state["final_result"])
