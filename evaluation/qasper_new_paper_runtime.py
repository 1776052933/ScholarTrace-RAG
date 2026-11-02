"""Isolated source-only index and retrieval for frozen, unseen QASPER papers.

Use inside the project's existing Python evaluation image. This module never
loads gold answers during index or retrieval and never calls a generation API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from . import qasper_followup as base
from .current_datasets import Source
from .current_runtime import CurrentRuntime
from .qasper_paper_holdout import MANIFEST, SOURCE as QASPER
from .qasper_phase3_selection import project, normalize, read, write


ROOT = MANIFEST.parent
COLLECTION = "eval_current_qasper_new_papers_20261022"


def frozen() -> dict:
    manifest = read(MANIFEST)
    if manifest["source_sha256"] != base.sha(QASPER):
        raise ValueError("Frozen QASPER source changed")
    if len(manifest["paper_ids"]) != 30 or len(set(manifest["paper_ids"])) != 30:
        raise ValueError("Invalid frozen paper set")
    return manifest


def runtime() -> CurrentRuntime:
    from rag_reader.config import RAGSettings

    settings = RAGSettings(
        embedding_model=str(base.ROOT / "data/m3/bge_m3_model"),
        embedding_device="cpu", embedding_batch_size=8,
        chunk_size=1000, chunk_overlap=200, retrieval_top_k=5,
        vector_backend="milvus", collection_name=COLLECTION,
        milvus_uri=os.getenv("QASPER_MILVUS_URI", "http://milvus:19530"),
        retrieval_mode="dense", retrieval_candidate_k=20,
        reranker_enabled=True,
        reranker_model=str(base.ROOT / "data/m3/reranker_windows_model"),
        pdf_parser_backend="pymupdf")
    return CurrentRuntime(settings, ROOT, generation=False)


def index() -> dict:
    from .run_m0_qasper import paper_text

    manifest = frozen()
    source = read(QASPER)
    rt = runtime()
    folder = ROOT / "index"
    existing = list(folder.glob("*.json")) if folder.exists() else []
    if not existing:
        rt.ensure_new_index()
    statuses = Counter()
    for number, pid in enumerate(manifest["paper_ids"], 1):
        path = folder / f"{pid}.json"
        if path.exists() and read(path)["status"] == "completed":
            old = read(path)
            if not rt.verify_ids(old["chunk_ids"]):
                raise ValueError(f"Stored chunk IDs missing from Milvus: {pid}")
            statuses["completed"] += 1
            continue
        paper = source[pid]
        item = Source(pid, paper["title"], paper_text(paper))
        record = {"paper_id": pid, "status": "started", "source_text_sha256":
                  hashlib.sha256(item.text.encode()).hexdigest(),
                  "collection": COLLECTION}
        write(path, record)
        started = time.perf_counter()
        try:
            record["chunk_ids"] = rt.index_sources([item])[pid]
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)[:300]
        record["seconds"] = time.perf_counter() - started
        write(path, record)
        statuses[record["status"]] += 1
        print(f"new-paper index {number}/30 {record['status']} {record['seconds']:.1f}s", flush=True)
        if record["status"] != "completed":
            raise RuntimeError(f"Index failed for {pid}: {record.get('error_type')}")
    result = {"collection": COLLECTION, "papers": len(manifest["paper_ids"]),
              "chunks": sum(len(read(folder / f"{pid}.json")["chunk_ids"])
                            for pid in manifest["paper_ids"]),
              "statuses": dict(statuses)}
    write(ROOT / "index_complete.json", result)
    return result


def retrieve() -> dict:
    manifest = frozen()
    if not (ROOT / "index_complete.json").exists():
        raise ValueError("Complete new-paper index before retrieval")
    source = read(QASPER)
    rt = runtime()
    reranker = rt.reader.retriever.reranker
    store = rt.reader.vector_store
    store.prepare_bm25()
    statuses = Counter()
    for number, case in enumerate(manifest["cases"], 1):
        qid, pid = case["question_id"], case["paper_id"]
        path = ROOT / "retrieval" / f"{qid}.json"
        if path.exists():
            statuses[read(path)["status"]] += 1
            continue
        question = next(q["question"] for q in source[pid]["qas"] if q["question_id"] == qid)
        record = {"question_id": qid, "paper_id": pid, "question": question,
                  "collection": COLLECTION, "status": "started",
                  "candidate_k": 20, "mode": "dense", "reranker": str(rt.settings.reranker_model)}
        write(path, record)
        started = time.perf_counter()
        try:
            vector = rt.reader.embedding_model.embed_query(question)
            candidates = store.search(vector, top_k=20, paper_id=pid,
                                      query_text=question, mode="dense")
            ranked = reranker.rerank(question, candidates, top_k=10)
            record["candidate_ids"] = [chunk["chunk_id"] for chunk in candidates]
            record["ranked"] = [{"chunk_id": chunk["chunk_id"], "paper_id": pid,
                                "section": chunk["metadata"].get("section", ""),
                                "page": chunk["metadata"].get("page"),
                                "text": chunk["text"],
                                "rerank_score": chunk["metadata"].get("raw_rerank_score")}
                               for chunk in ranked]
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)[:300]
        record["seconds"] = time.perf_counter() - started
        write(path, record)
        statuses[record["status"]] += 1
        if number % 5 == 0 or record["status"] != "completed":
            print(f"new-paper retrieve {number}/{len(manifest['cases'])} {record['status']} {record['seconds']:.1f}s", flush=True)
        if record["status"] != "completed":
            raise RuntimeError(f"Retrieval failed for {qid}: {record.get('error_type')}")
    result = {"papers": len(manifest["paper_ids"]), "questions": len(manifest["cases"]),
              "statuses": dict(statuses), "collection": COLLECTION}
    write(ROOT / "retrieval_complete.json", result)
    return result


def score_retrieval() -> dict:
    manifest = frozen()
    source = read(QASPER)
    gold = base.official().get_answers_and_evidence(source, False)
    rows = []
    for case in manifest["cases"]:
        qid, pid = case["question_id"], case["paper_id"]
        retrieved = read(ROOT / "retrieval" / f"{qid}.json")
        if retrieved["status"] != "completed":
            raise ValueError(f"Incomplete retrieval: {qid}")
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        keys = {normalize(p) for p in paragraphs}
        groups = [{normalize(p) for p in ref["evidence"] if normalize(p) in keys}
                  for ref in gold[qid]]
        groups = [group for group in groups if group]
        ranks = {}
        for top_k in (5, 10):
            predicted = project([chunk["text"] for chunk in retrieved["ranked"][:top_k]], paragraphs)
            ranks[f"hit_at_{top_k}"] = int(any(predicted & group for group in groups)) if groups else None
        rows.append({"question_id": qid, "paper_id": pid, "has_text_gold": bool(groups),
                     **ranks, "seconds": retrieved["seconds"]})
    eligible = [row for row in rows if row["has_text_gold"]]
    result = {"questions": len(rows), "papers": len(manifest["paper_ids"]),
              "text_gold_questions": len(eligible),
              "text_hit_at_5": sum(row["hit_at_5"] for row in eligible),
              "text_hit_at_10": sum(row["hit_at_10"] for row in eligible),
              "mean_retrieval_seconds": statistics.mean(row["seconds"] for row in rows),
              "note": "Frozen new-paper IDs. Paragraph text projection is not semantic citation support."}
    write(ROOT / "retrieval_scores.json", result)
    write(ROOT / "retrieval_cases.json", rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "retrieve", "score-retrieve"])
    args = parser.parse_args()
    result = index() if args.action == "index" else retrieve() if args.action == "retrieve" else score_retrieval()
    print(json.dumps(result, ensure_ascii=False, indent=2))
