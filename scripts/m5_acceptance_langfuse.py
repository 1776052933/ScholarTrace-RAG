"""Verify the real Langfuse SDK exports nested OTLP spans to a local receiver."""
from __future__ import annotations

import gzip
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def main() -> None:
    requests_seen = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            requests_seen.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    receiver = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=receiver.serve_forever, daemon=True)
    thread.start()
    try:
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-local-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-local-test"
        os.environ["LANGFUSE_BASE_URL"] = f"http://127.0.0.1:{receiver.server_port}"
        from observability.tracing import flush_traces, trace_span
        from app.workflows.search import GraphSearchAgent
        from search_agent.schema import SearchEvaluation, SearchPlan, SearchQuery, SearchResult

        class Planner:
            def plan(self, search_input, round_number):
                return SearchPlan(
                    original_query=search_input.original_query, search_round=round_number,
                    strategy="local-test", queries=[SearchQuery(
                        query="local-test", source="openalex", purpose="trace acceptance",
                    )],
                )

        class Searcher:
            def search(self, _plan, **_kwargs):
                return SearchResult(papers=[], total=0, query_used="local-test")

        class Evaluator:
            def evaluate_with_rules(self, _result, _plan):
                return SearchEvaluation(relevance_score=0, quality_score=0,
                                        coverage_score=0, need_refine=False)

        class Reflector:
            def reflect(self, _plan, _result, _evaluation):
                return None

        agent = SimpleNamespace(
            config=SimpleNamespace(max_rounds=1, use_llm=False), planner=Planner(),
            searcher=Searcher(), evaluator=Evaluator(), reflector=Reflector(),
        )

        with trace_span("research.job", input={"job_id": "local-test"}) as parent:
            with TemporaryDirectory() as folder:
                result = GraphSearchAgent(
                    agent, checkpoint_path=Path(folder) / "graph.sqlite3",
                ).run({"original_query": "local-test",
                       "sub_queries": [{"query": "local-test"}]})
                assert result.rounds_completed == 1
            with trace_span("rag.answer_generation", as_type="generation", model="test-model") as generation:
                generation.update(output={"answer_length": 9}, usage_details={
                    "input_tokens": 11, "output_tokens": 7,
                })
            parent.update(output={"stage_count": 2})
        flush_traces()
        spans = []
        for path, body in requests_seen:
            assert path == "/api/public/otel/v1/traces", path
            batch = ExportTraceServiceRequest.FromString(body)
            for resource in batch.resource_spans:
                for scope in resource.scope_spans:
                    spans.extend(scope.spans)
        by_name = {span.name: span for span in spans}
        assert {"research.job", "research.search_graph", "search.plan", "search.search",
                "search.evaluate", "search.reflect", "search.finish",
                "rag.answer_generation"} <= set(by_name)
        root = by_name["research.job"]
        assert by_name["research.search_graph"].parent_span_id == root.span_id
        assert by_name["search.plan"].parent_span_id == by_name["research.search_graph"].span_id
        assert by_name["rag.answer_generation"].parent_span_id == root.span_id
        print(json.dumps({"otlp_requests": len(requests_seen),
                          "span_names": sorted(by_name), "parent_links": 3}))
    finally:
        receiver.shutdown()
        receiver.server_close()


if __name__ == "__main__":
    main()
