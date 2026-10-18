"""Read-only acceptance of the live MCP transport and persisted project data."""
from __future__ import annotations

import argparse
import asyncio
import json

from mcp import Client
import requests


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", default="http://mcp:8001/mcp")
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--answer-id")
    args = parser.parse_args()

    health = requests.get(f"{args.api_url}/api/v1/health", timeout=10)
    health.raise_for_status()
    jobs_response = requests.get(f"{args.api_url}/api/v1/jobs", timeout=10)
    jobs_response.raise_for_status()
    jobs = jobs_response.json()
    async with Client(args.mcp_url) as client:
        listed = await client.list_tools()
        names = sorted(tool.name for tool in listed.tools)
        expected = ["ask_papers", "get_evidence", "get_job_status", "search_papers", "start_ingestion"]
        assert names == expected, names
        report = {"api_health": health.status_code, "mcp_tools": names}
        if jobs:
            job_id = jobs[0]["job_id"]
            result = await client.call_tool("get_job_status", {"job_id": job_id})
            assert not result.is_error, result
            assert result.structured_content["job_id"] == job_id
            report["job"] = {"job_id": job_id, "status": result.structured_content["status"]}
        if args.answer_id:
            result = await client.call_tool("get_evidence", {"answer_id": args.answer_id})
            assert not result.is_error, result
            report["evidence_blocks"] = len(result.structured_content.get("result", []))
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
