"""Retrieve the locked QASPER question holdout from the existing 151-paper index.

This phase never loads gold answers or sends anything to a model provider.
"""
from __future__ import annotations

from collections import Counter
import json
import statistics
import time

from rag_reader.embedding import EmbeddingModel
from rag_reader.milvus_store import MilvusVectorStore
from rag_reader.reranker import CrossEncoderReranker

from . import qasper_followup as base
from .qasper_holdout import PATH as HOLDOUT
from .qasper_retrieval_ablation import COLLECTION, MODEL, RERANKER_MODEL

FOLDER = base.NEW / "holdout_retrieval"


def run(uri="http://127.0.0.1:19532", device="cuda"):
    frozen = base.read(HOLDOUT)
    if frozen["collection"] != COLLECTION or frozen["source_sha256"] != base.sha(base.SOURCE):
        raise ValueError("Holdout source or collection changed")
    store = MilvusVectorStore(COLLECTION, uri=uri,
                              pipeline_signature="page-section-1000-200-pymupdf-v1")
    store.prepare_bm25()
    embedding = EmbeddingModel(str(MODEL), device="cpu", batch_size=8)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoderReranker(str(RERANKER_MODEL),
                                   model=CrossEncoder(str(RERANKER_MODEL),
                                                      max_length=512, device=device))
    statuses = Counter()
    for index, case in enumerate(frozen["cases"], 1):
        path = FOLDER / f"{case['question_id']}.json"
        if path.exists():
            statuses[base.read(path)["status"]] += 1
            continue
        record = {"question_id": case["question_id"], "paper_id": case["paper_id"],
                  "question": case["question"], "status": "started", "mode": "dense",
                  "candidate_k": 20, "top_k": 5, "reranker_device": device}
        base.write(path, record)
        started = time.perf_counter()
        try:
            vector = embedding.embed_query(case["question"])
            candidates = store.search(vector, paper_id=case["paper_id"],
                                      query_text=case["question"], mode="dense", top_k=20)
            ranked = reranker.rerank(case["question"], candidates, top_k=5)
            record["candidate_ids"] = [row["chunk_id"] for row in candidates]
            record["retrieved"] = [{"chunk_id": row["chunk_id"], "paper_id": case["paper_id"],
                                    "section": row["metadata"].get("section", ""), "text": row["text"]}
                                   for row in ranked]
            record["status"] = "completed"
        except Exception as error:
            record["status"], record["error_type"] = "failed", type(error).__name__
            record["error_detail"] = str(error)[:300]
        record["seconds"] = time.perf_counter() - started
        base.write(path, record)
        statuses[record["status"]] += 1
        print(f"holdout retrieval {index}/60 {record['status']} {record['seconds']:.1f}s", flush=True)
    summary = {"questions": len(frozen["cases"]), "statuses": dict(statuses),
               "collection": COLLECTION, "holdout_sha256": base.sha(HOLDOUT),
               "mode": "dense", "candidate_k": 20, "reranker": str(RERANKER_MODEL)}
    base.write(FOLDER / "summary.json", summary)
    return summary


def score():
    frozen = base.read(HOLDOUT)
    source = base.read(base.SOURCE)
    evaluator = base.official()
    gold = evaluator.get_answers_and_evidence(source, False)
    predictions, rows = {}, []
    for case in frozen["cases"]:
        qid, pid = case["question_id"], case["paper_id"]
        record = base.read(FOLDER / f"{qid}.json")
        if record["status"] != "completed":
            raise ValueError(f"Incomplete retrieval: {qid}")
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        evidence = base.project([c["text"] for c in record["retrieved"]], paragraphs)
        predictions[qid] = {"answer": "", "evidence": evidence}
        paragraph_keys = {base.normalize(p) for p in paragraphs}
        groups = [set(base.normalize(p) for p in ref["evidence"] if base.normalize(p) in paragraph_keys)
                  for ref in gold[qid]]
        groups = [g for g in groups if g]
        actual = {base.normalize(p) for p in evidence}
        rows.append({"question_id": qid, "stratum": case["stratum"],
                     "has_text_gold": bool(groups),
                     "hit_at_5": int(any(actual & group for group in groups)) if groups else None,
                     "seconds": record["seconds"]})
    official = evaluator.evaluate({c["question_id"]: gold[c["question_id"]]
                                   for c in frozen["cases"]}, predictions)
    eligible = [r for r in rows if r["has_text_gold"]]
    result = {"questions": len(rows), "text_gold_questions": len(eligible),
              "text_hit_at_5": statistics.mean(r["hit_at_5"] for r in eligible) if eligible else None,
              "retrieved_paragraph_evidence_f1": official["Evidence F1"],
              "mean_seconds": statistics.mean(r["seconds"] for r in rows),
              "note": "Text paragraph projection only; excludes table/caption-only gold from hit-rate denominator."}
    base.write(FOLDER / "scores.json", result)
    base.write(FOLDER / "scored_cases.json", rows)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default="http://127.0.0.1:19532")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--score", action="store_true", help="Score saved retrieval without loading models")
    args = p.parse_args()
    print(json.dumps(score() if args.score else run(args.uri, args.device), ensure_ascii=False, indent=2))
