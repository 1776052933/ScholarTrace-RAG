"""Bounded LangGraph retrieval for evidence across selected papers."""
from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from observability.tracing import trace_span
from rag_reader.schema import DocumentChunk


class EvidenceState(TypedDict, total=False):
    paper_ids: list[str]
    question: str
    top_k: int
    chunks: list[dict[str, Any]]
    missing_ids: list[str]
    absent_ids: list[str]
    rounds: int
    sufficient: bool
    stop_reason: str


class EvidenceGraph:
    """Expand retrieval when the global Top-K underrepresents selected papers.

    Coverage is a deterministic gate, not a claim that passages support an answer.
    The answer model must still decide whether the retrieved text is sufficient.
    """

    def __init__(self, retriever: Any, *, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k
        self.max_expansions = min(2, max(0, int(os.getenv("RAG_EVIDENCE_MAX_EXPANSIONS", "1"))))
        graph = StateGraph(EvidenceState)
        for name, action in (
            ("retrieve", self._retrieve), ("assess", self._assess),
            ("expand", self._expand), ("finish", self._finish),
        ):
            def traced(state: EvidenceState, *, _name=name, _action=action):
                with trace_span(f"rag.evidence.{_name}",
                                input={"paper_ids": state["paper_ids"], "round": state["rounds"]}) as span:
                    result = _action(state)
                    span.update(output={"chunk_count": len(result.get("chunks", state.get("chunks", [])))})
                    return result
            graph.add_node(name, traced)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "assess")
        graph.add_conditional_edges("assess", self._route, {"expand": "expand", "finish": "finish"})
        graph.add_edge("expand", "assess")
        graph.add_edge("finish", END)
        self.graph = graph.compile()

    def run(self, paper_ids: list[str], question: str) -> tuple[list[DocumentChunk], dict[str, Any]]:
        with trace_span("rag.evidence_graph", input={"paper_ids": paper_ids}) as span:
            state = self.graph.invoke({
                "paper_ids": paper_ids, "question": question, "top_k": self.top_k,
                "chunks": [], "missing_ids": [], "rounds": 0,
            })
            span.update(output={"chunk_count": len(state["chunks"]),
                                "expansions": state["rounds"],
                                "stop_reason": state["stop_reason"]})
        chunks = [DocumentChunk.model_validate(item) for item in state["chunks"]]
        return chunks, {
            "sufficient": state["sufficient"], "stop_reason": state["stop_reason"],
            "expansions": state["rounds"], "missing_paper_ids": state["missing_ids"],
            "absent_paper_ids": state["absent_ids"],
            "chunk_count": len(chunks),
        }

    def _retrieve(self, state: EvidenceState) -> dict[str, Any]:
        ids = state["paper_ids"]
        found = (
            self.retriever.retrieve(ids[0], state["question"], top_k=state["top_k"])
            if len(ids) == 1 else
            self.retriever.retrieve_multiple(ids, state["question"], top_k=state["top_k"])
        )
        return {"chunks": [chunk.model_dump(mode="json") for chunk in found]}

    @staticmethod
    def _assess(state: EvidenceState) -> dict[str, Any]:
        counts = {paper_id: 0 for paper_id in state["paper_ids"]}
        for chunk in state["chunks"]:
            if chunk["paper_id"] in counts:
                counts[chunk["paper_id"]] += 1
        # Two passages per paper reduce the chance of comparing one isolated line.
        missing = [paper_id for paper_id, count in counts.items() if count < 2]
        absent = [paper_id for paper_id, count in counts.items() if count == 0]
        return {"missing_ids": missing, "absent_ids": absent, "sufficient": not missing}

    def _route(self, state: EvidenceState) -> str:
        return "expand" if state["missing_ids"] and state["rounds"] < self.max_expansions else "finish"

    def _expand(self, state: EvidenceState) -> dict[str, Any]:
        chunks = [DocumentChunk.model_validate(item) for item in state["chunks"]]
        seen = {chunk.chunk_id for chunk in chunks}
        limit = min(8, max(2, state["top_k"] + 2 * (state["rounds"] + 1)))
        for paper_id in state["missing_ids"]:
            needed = 2 - sum(chunk.paper_id == paper_id for chunk in chunks)
            for chunk in self.retriever.retrieve(paper_id, state["question"], top_k=limit):
                if chunk.chunk_id not in seen:
                    chunks.append(chunk)
                    seen.add(chunk.chunk_id)
                    needed -= 1
                    if needed <= 0:
                        break
        return {"chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                "rounds": state["rounds"] + 1}

    @staticmethod
    def _finish(state: EvidenceState) -> dict[str, Any]:
        reason = "covered" if state["sufficient"] else (
            "no_evidence" if not state["chunks"] else "expansion_limit"
        )
        return {"stop_reason": reason}
