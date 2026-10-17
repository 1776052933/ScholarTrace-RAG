"""Check live multi-paper evidence retrieval without calling a generation model."""
from __future__ import annotations

import argparse
import json

from rag_reader.config import RAGSettings
from rag_reader.pipeline import RAGReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-id", action="append", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    settings = RAGSettings.from_env()
    reader = RAGReader(settings=settings)
    chunks = reader.retriever.retrieve_multiple(
        args.paper_id, args.question, top_k=settings.retrieval_top_k
    )
    present = {chunk.paper_id for chunk in chunks}
    for paper_id in args.paper_id:
        if paper_id not in present:
            chunks.extend(reader.retriever.retrieve(paper_id, args.question, top_k=1))
    result = {
        "collection": settings.collection_name,
        "requested_papers": args.paper_id,
        "evidence": [
            {"paper_id": chunk.paper_id, "chunk_id": chunk.chunk_id,
             "page": chunk.page, "excerpt": chunk.text[:140]}
            for chunk in chunks
        ],
    }
    print(json.dumps(result, ensure_ascii=False))
    if set(args.paper_id) - {chunk.paper_id for chunk in chunks}:
        raise SystemExit("one or more selected papers have no retrievable evidence")


if __name__ == "__main__":
    main()
