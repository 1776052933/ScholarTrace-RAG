"""Exercise M5 evidence graph against indexed Milvus papers without generation."""
from __future__ import annotations

import argparse
import json

import requests

from rag_reader.config import RAGSettings
from rag_reader.evidence_graph import EvidenceGraph
from rag_reader.pipeline import RAGReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-id", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="Compare the methods and evidence in these papers")
    args = parser.parse_args()
    response = requests.get(f"{args.api_url}/api/v1/answers/{args.answer_id}/evidence", timeout=10)
    response.raise_for_status()
    paper_ids = list(dict.fromkeys(
        item["paper_id"] for item in response.json() if item.get("paper_id")
    ))
    if len(paper_ids) < 2:
        raise SystemExit("the saved answer does not cite two distinct papers")
    settings = RAGSettings.from_env()
    reader = RAGReader(settings=settings)
    chunks, report = EvidenceGraph(reader.retriever, top_k=settings.retrieval_top_k).run(
        paper_ids, args.question,
    )
    print(json.dumps({
        "collection": settings.collection_name,
        "paper_ids": paper_ids,
        "report": report,
        "per_paper": {paper_id: sum(chunk.paper_id == paper_id for chunk in chunks)
                      for paper_id in paper_ids},
        "pages": {paper_id: sorted({chunk.page for chunk in chunks
                                    if chunk.paper_id == paper_id and chunk.page})
                  for paper_id in paper_ids},
    }, ensure_ascii=False))
    if report["absent_paper_ids"]:
        raise SystemExit("one or more selected papers are absent from retrieval")


if __name__ == "__main__":
    main()
