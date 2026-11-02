"""Send one harmless LangSmith trace to check credentials and connectivity.

Run only after explicitly enabling tracing and configuring a LangSmith API key.
No model provider is called by this script.
"""
from __future__ import annotations

import asyncio
import os
from time import sleep

from langsmith import Client

from observability.tracing import flush_traces, trace_span


def main() -> None:
    if os.getenv("LANGSMITH_TRACING", "").lower() not in {"1", "true", "yes"}:
        raise SystemExit("Set LANGSMITH_TRACING=true before running this check")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before running this check")
    with trace_span("scholartrace.smoke_test", input={"test": "connectivity"}) as run:
        run.update(output={"status": "ok"})
        run_id = run.run.id
    flush_traces()
    client = Client()
    project = client.read_project(project_name=os.getenv("LANGSMITH_PROJECT", "default"))
    for attempt in range(5):
        try:
            stored = asyncio.run(client.runs.retrieve(
                run_id, project_id=project.id, selects=["ID", "NAME"],
            ))
            break
        except Exception:
            if attempt == 4:
                raise
            sleep(1)
    if stored.name != "scholartrace.smoke_test":
        raise RuntimeError("LangSmith returned an unexpected trace")
    print(f"LangSmith trace confirmed: {run_id}")


if __name__ == "__main__":
    main()
