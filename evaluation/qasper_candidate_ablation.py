"""Compare first-stage dense versus hybrid candidate recall on missed evidence.

Only the query and already indexed paper ID are sent to Milvus. Annotated
paragraphs are consulted after search to score candidates. This experiment
does not change the application default or claim final reranked Top-5 gains.
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

FOLDER = base.NEW / "candidate_ablation"
VARIANTS = (("dense", 20), ("hybrid", 20), ("hybrid", 40))


def run(uri="http://127.0.0.1:19532"):
    selected = [r for r in base.read(base.NEW / "baseline_cases.json")
                if r["review_priority"] == "text_evidence_missed"]
    if len(selected) != 41:
        raise ValueError("Expected the frozen 41-case missed-evidence cohort")
    store = MilvusVectorStore(COLLECTION, uri=uri,
                              pipeline_signature="page-section-1000-200-pymupdf-v1")
    store.prepare_bm25()
    embedding = EmbeddingModel(str(MODEL), device="cpu", batch_size=8)
    source = base.read(base.SOURCE)
    gold = base.official().get_answers_and_evidence(source, False)
    rows = []
    for index, case in enumerate(selected, 1):
        qid = case["question_id"]
        path = FOLDER / f"{qid}.json"
        if path.exists():
            rows.append(base.read(path))
            continue
        pid = case["paper_id"]
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        paragraph_keys = {base.normalize(p) for p in paragraphs}
        groups = [set(base.normalize(p) for p in ref["evidence"] if base.normalize(p) in paragraph_keys)
                  for ref in gold[qid]]
        groups = [g for g in groups if g]
        assert groups
        query = case["question"]
        vector = embedding.embed_query(query)
        results = {}
        for mode, k in VARIANTS:
            began = time.perf_counter()
            candidates = store.search(vector, paper_id=pid, query_text=query, mode=mode,
                                      top_k=k, candidate_k=k)
            projected = {base.normalize(p) for p in base.project(
                [item["text"] for item in candidates], paragraphs)}
            results[f"{mode}_{k}"] = {
                "candidate_hit": int(any(projected & group for group in groups)),
                "candidate_ids": [item["chunk_id"] for item in candidates],
                "seconds": time.perf_counter() - began,
            }
        row = {"question_id": qid, "paper_id": pid, "variants": results}
        base.write(path, row)
        rows.append(row)
        print(f"candidate {index}/{len(selected)} " + " ".join(
            f"{name}:{result['candidate_hit']}" for name,result in results.items()), flush=True)
    report = {"questions": len(rows), "cohort": "41 text-evidence misses in baseline reranked Top-5",
              "collection": COLLECTION, "embedding": str(MODEL),
              "variants": {f"{mode}_{k}": {
                  "candidate_hit_count": sum(r["variants"][f"{mode}_{k}"]["candidate_hit"] for r in rows),
                  "candidate_hit_rate": statistics.mean(r["variants"][f"{mode}_{k}"]["candidate_hit"] for r in rows),
                  "mean_search_seconds": statistics.mean(r["variants"][f"{mode}_{k}"]["seconds"] for r in rows),
              } for mode,k in VARIANTS},
              "note": "Candidate recall only; reranker can still discard the evidence. This error-selected cohort is not representative of all 200 questions."}
    base.write(FOLDER / "summary.json", report)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default="http://127.0.0.1:19532")
    args = p.parse_args()
    print(json.dumps(run(args.uri), ensure_ascii=False, indent=2))
