"""M0 fixed QASPER single-paper retrieval and optional answer baseline.

Run with: python -m evaluation.run_m0_qasper --limit 5 --retrieval-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import traceback
from typing import Any

from literature_database.schema import PaperMetadata, PaperResource
from rag_reader.embedding import EmbeddingModel
from rag_reader.llm_client import DeepSeekClient
from rag_reader.pipeline import RAGReader
from rag_reader.qa import PaperQA
from rag_reader.retriever import PaperRetriever
from rag_reader.splitter import DocumentSplitter
from rag_reader.vector_store import VectorStore

from .m0_common import DATA, code_hash, embedding_revision, git_head, new_run_dir, require_real_embedding, runtime_info, sha256, summarize, utc_now, write_json
from .m0_metrics import answer_f1, evidence_recall, reference_answer


class RecordingRetriever(PaperRetriever):
    def retrieve(self, paper_id: str | None, question: str, *, top_k: int | None = None):
        start = time.perf_counter()
        self.last_chunks = super().retrieve(paper_id, question, top_k=top_k)
        self.last_seconds = time.perf_counter() - start
        return self.last_chunks


def paper_text(paper: dict[str, Any]) -> str:
    """Only source text is materialized; gold answers/evidence never enter the index."""
    blocks = [str(paper.get("title") or ""), str(paper.get("abstract") or "")]
    for section in paper.get("full_text") or []:
        blocks.append(str(section.get("section_name") or ""))
        blocks.extend(str(paragraph) for paragraph in section.get("paragraphs") or [])
    return "\n\n".join(block for block in blocks if block.strip())


def select_questions(data: dict[str, Any], limit: int, seed: int) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        (paper_id, question)
        for paper_id, paper in sorted(data.items())
        for question in paper.get("qas") or []
        if question.get("question_id") and question.get("question")
    ]
    random.Random(seed).shuffle(rows)
    chosen = rows[:limit] if limit else rows
    return sorted(chosen, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=5, help="0 means the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.top_k < 1:
        parser.error("--limit must be >= 0 and --top-k must be >= 1")

    dataset = DATA / "qasper" / f"qasper-{args.split}-v0.3.json"
    if not dataset.is_file():
        parser.error(f"Missing QASPER file: {dataset}")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    selected = select_questions(data, args.limit, args.seed)
    if not selected:
        parser.error("No questions selected")
    if args.resume:
        if not args.run_dir:
            parser.error("--resume requires --run-dir")
        run_dir = Path(args.run_dir).resolve()
        if not (run_dir / "manifest.json").is_file():
            parser.error("No manifest.json in resume directory")
    else:
        run_dir = new_run_dir("qasper", args.run_dir)
    started = utc_now()
    model = EmbeddingModel()
    backend = require_real_embedding(model)
    store = VectorStore(persist_directory=str(run_dir / "chroma"))
    retriever = RecordingRetriever(embedding_model=model, vector_store=store)
    client = None if args.retrieval_only else DeepSeekClient()
    reader = RAGReader(
        embedding_model=model,
        vector_store=store,
        retriever=retriever,
        splitter=DocumentSplitter(),
        qa=PaperQA(retriever, llm_client=client, top_k=args.top_k),
    )
    manifest = {
        "task": "qasper_single_paper_qa",
        "started_at_utc": started,
        "git_head": git_head(),
        "code_sha256": code_hash(), "runtime": runtime_info(),
        "dataset": str(dataset.relative_to(DATA)),
        "dataset_sha256": sha256(dataset),
        "split": args.split,
        "selection": {"seed": args.seed, "limit": args.limit, "question_ids": [row["question_id"] for _, row in selected]},
        "mode": "retrieval_only" if args.retrieval_only else "retrieval_and_answer",
        "index": {"store": "Chroma", "model": model.model_name, "backend": backend, "model_revision": embedding_revision(model),
                  "chunk_size": reader.splitter.chunk_size, "chunk_overlap": reader.splitter.chunk_overlap,
                  "top_k": args.top_k},
        "llm": None if client is None else {"base_url": client.base_url, "model": client.model, "temperature": 0.0},
        "scope": "full_split" if args.limit == 0 or args.limit >= sum(len(p.get("qas") or []) for p in data.values()) else "sample",
        "metric_note": "Evidence recall is a gold-token coverage >= 0.8 proxy using best human annotator; answer token-F1 is not official QASPER score.",
    }
    if args.resume:
        previous = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        fields = ("code_sha256", "dataset_sha256", "selection", "mode", "index", "llm", "runtime")
        if any(previous.get(field) != manifest.get(field) for field in fields):
            parser.error("Resume code, data, selected questions, or model config differs from original run")
        manifest = previous
    else:
        write_json(run_dir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    current_paper = None
    ingest_seconds = 0.0
    for index, (paper_id, question) in enumerate(selected, 1):
        record_path = run_dir / "records" / f"{index:04d}_{question['question_id']}.json"
        if args.resume and record_path.is_file():
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        start = time.perf_counter()
        record: dict[str, Any] = {
            "status": "failed", "paper_id": paper_id, "question_id": question["question_id"],
            "question": question["question"], "gold_answers": [reference_answer(a) for a in question.get("answers") or []],
            "gold_evidence": [((a.get("answer") or {}).get("highlighted_evidence") or (a.get("answer") or {}).get("evidence") or []) for a in question.get("answers") or []],
            "metrics": {},
        }
        try:
            if paper_id != current_paper:
                document_path = run_dir / "documents" / f"{paper_id}.txt"
                document_path.parent.mkdir(parents=True, exist_ok=True)
                document_path.write_text(paper_text(data[paper_id]), encoding="utf-8")
                resource = PaperResource(paper=PaperMetadata(paper_id=paper_id, title=data[paper_id]["title"]), text_path=str(document_path))
                ingest_start = time.perf_counter()
                chunks = reader.ingest_paper(resource)
                record["ingest_seconds"] = time.perf_counter() - ingest_start
                current_paper = paper_id
                record["paper_chunk_count"] = len(chunks)
            if args.retrieval_only:
                retriever.retrieve(paper_id, question["question"], top_k=args.top_k)
                response = None
            else:
                response = reader.ask(paper_id, question["question"])
            retrieved = retriever.last_chunks
            record["retrieval_seconds"] = retriever.last_seconds
            record["retrieved"] = [
                {"chunk_id": c.chunk_id, "score": c.metadata.get("retrieval_score"),
                 "section": c.section, "page": c.page, "text": c.text}
                for c in retrieved
            ]
            record["metrics"]["evidence_recall_proxy"] = evidence_recall(
                question.get("answers") or [], [c.text for c in retrieved]
            )
            if response is not None:
                record["answer"] = response.to_dict()
                record["metrics"]["answer_token_f1_proxy"] = answer_f1(response.answer, question.get("answers") or [])
                record["metrics"]["has_citation"] = float(bool(response.sources))
                record["usage"] = client.get_token_usage()["last"] if client else None
            record["status"] = "completed"
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=3)}
        record["total_seconds"] = time.perf_counter() - start
        records.append(record)
        write_json(record_path, record)
        print(f"{index}/{len(selected)} {record['status']} {question['question_id']} {record['total_seconds']:.1f}s", flush=True)

    names = ["evidence_recall_proxy", "answer_token_f1_proxy", "has_citation"]
    summary = summarize(records, names)
    summary.update({"task": manifest["task"], "scope": manifest["scope"], "mode": manifest["mode"],
                    "ingest_seconds": sum(r.get("ingest_seconds", 0.0) for r in records), "finished_at_utc": utc_now(),
                    "token_usage_total": ({name: sum((r.get("usage") or {}).get(name, 0) for r in records) for name in ("prompt_tokens", "completion_tokens", "total_tokens")} if client else None)})
    write_json(run_dir / "summary.json", summary)
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
