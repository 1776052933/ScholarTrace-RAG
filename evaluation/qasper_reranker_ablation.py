"""Full-200 retrieval ablation: dense Top-5 before versus after reranking.

The original reranked Top-5 is read from the frozen baseline. This script only
queries the same Milvus index for raw dense candidates and never changes the
application reranker setting or calls a generation model.
"""
from __future__ import annotations

from collections import Counter
import json
import statistics
import time

from rag_reader.embedding import EmbeddingModel
from rag_reader.milvus_store import MilvusVectorStore

from . import qasper_followup as base
from .qasper_retrieval_ablation import COLLECTION, MODEL

FOLDER = base.NEW / "reranker_ablation"


def run(uri="http://127.0.0.1:19532"):
    manifest, ids, _, old_retrieval, source, question_map = base.original()
    assert manifest["settings"]["collection_name"] == COLLECTION
    store = MilvusVectorStore(COLLECTION, uri=uri,
                              pipeline_signature="page-section-1000-200-pymupdf-v1")
    store.prepare_bm25()
    embedding = EmbeddingModel(str(MODEL), device="cpu", batch_size=8)
    gold = base.official().get_answers_and_evidence(source, False)
    cases = []
    for index, qid in enumerate(ids, 1):
        path = FOLDER / f"{qid}.json"
        if path.exists():
            cases.append(base.read(path))
            continue
        pid, paper, question = question_map[qid]
        paragraphs = [p for section in paper["full_text"] for p in section["paragraphs"]]
        paragraph_keys = {base.normalize(p) for p in paragraphs}
        groups = [set(base.normalize(p) for p in ref["evidence"] if base.normalize(p) in paragraph_keys)
                  for ref in gold[qid]]
        groups = [g for g in groups if g]
        started = time.perf_counter()
        vector = embedding.embed_query(question["question"])
        candidates = store.search(vector, top_k=20, paper_id=pid, mode="dense")
        def match(excerpts):
            if not groups:
                return None
            projected = {base.normalize(p) for p in base.project(excerpts, paragraphs)}
            return int(any(projected & group for group in groups))
        row = {"question_id": qid, "paper_id": pid, "has_text_gold": bool(groups),
               "raw_dense_hit_at_5": match([c["text"] for c in candidates[:5]]),
               "dense_candidate_hit_at_20": match([c["text"] for c in candidates]),
               "reranked_hit_at_5": match([c["text"] for c in old_retrieval[qid]["retrieved"]]),
               "raw_dense_top5_ids": [c["chunk_id"] for c in candidates[:5]],
               "raw_dense_candidate20_ids": [c["chunk_id"] for c in candidates],
               "seconds": time.perf_counter() - started}
        base.write(path, row)
        cases.append(row)
        if index % 20 == 0:
            print(f"raw dense {index}/{len(ids)}", flush=True)
    eligible = [r for r in cases if r["has_text_gold"]]
    pair = Counter((r["raw_dense_hit_at_5"], r["reranked_hit_at_5"]) for r in eligible)
    report = {"questions": len(cases), "text_gold_questions": len(eligible),
              "collection": COLLECTION,
              "raw_dense_hit_at_5": statistics.mean(r["raw_dense_hit_at_5"] for r in eligible),
              "reranked_hit_at_5": statistics.mean(r["reranked_hit_at_5"] for r in eligible),
              "candidate_hit_at_20": statistics.mean(r["dense_candidate_hit_at_20"] for r in eligible),
              "paired": {"both_hit": pair[(1,1)], "raw_only": pair[(1,0)],
                         "reranked_only": pair[(0,1)], "both_miss": pair[(0,0)]},
              "mean_dense_query_seconds": statistics.mean(r["seconds"] for r in cases),
              "note": "Same source index and paper scope; old reranked outputs versus newly queried raw dense order. Repeated Milvus query may differ if index state changed. Hit is paragraph projection, not semantic answer support."}
    base.write(FOLDER / "summary.json", report)
    base.write(FOLDER / "cases.json", cases)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default="http://127.0.0.1:19532")
    args = p.parse_args()
    print(json.dumps(run(args.uri), ensure_ascii=False, indent=2))
