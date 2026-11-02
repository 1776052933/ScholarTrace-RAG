"""Read-only retrieval ablation on the QASPER 200-question Milvus collection.

This compares modes only for the 41 baseline questions whose annotated text
evidence was missed by Top-5. The same paper scope, embedding and reranker are
used throughout. Gold is loaded only after retrieval for scoring.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import time

from rag_reader.embedding import EmbeddingModel
from rag_reader.milvus_store import MilvusVectorStore
from rag_reader.reranker import CrossEncoderReranker

from . import qasper_followup as base

COLLECTION = "eval_current_f17d4b078e13aaab6ac9e586"
MODEL = base.ROOT / "data/m3/bge_m3_model"
RERANKER_MODEL = base.ROOT / "data/m3/reranker_windows_model"
RESULTS = base.NEW / "retrieval_ablation"
VARIANTS = (("dense", 20), ("hybrid", 20), ("hybrid", 40))


def run(limit=41, uri="http://127.0.0.1:19532", device="cpu"):
    if not MODEL.is_dir():
        raise FileNotFoundError(MODEL)
    if not (RERANKER_MODEL / "model.safetensors").is_file():
        raise FileNotFoundError(RERANKER_MODEL)
    baseline = base.read(base.NEW / "baseline_cases.json")
    selected = [row for row in baseline if row["review_priority"] == "text_evidence_missed"][:limit]
    manifest = base.read(base.OLD / "manifest.json")
    if manifest["settings"]["collection_name"] != COLLECTION:
        raise RuntimeError("Evaluation collection differs from frozen baseline")
    store = MilvusVectorStore(COLLECTION, uri=uri,
                              pipeline_signature="page-section-1000-200-pymupdf-v1")
    store.prepare_bm25()
    description = json.loads(store.client.describe_collection(COLLECTION)["description"])
    if description["dimension"] != 1024 or description["schema"] != "scholartrace-m3-v1":
        raise RuntimeError("Unexpected Milvus embedding signature")
    embedding = EmbeddingModel(str(MODEL), device="cpu", batch_size=8)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoderReranker(str(RERANKER_MODEL),
                                   model=CrossEncoder(str(RERANKER_MODEL),
                                                      max_length=512, device=device))
    source = base.read(base.SOURCE)
    gold = base.official().get_answers_and_evidence(source, False)
    rows = []
    for index, row in enumerate(selected, 1):
        qid = row["question_id"]
        path = RESULTS / f"{qid}.json"
        if path.exists():
            rows.append(base.read(path))
            continue
        pid = row["paper_id"]
        paper = source[pid]
        paragraphs = [p for section in paper["full_text"] for p in section["paragraphs"]]
        paragraph_keys = {base.normalize(p) for p in paragraphs}
        groups = [set(base.normalize(p) for p in ref["evidence"] if base.normalize(p) in paragraph_keys)
                  for ref in gold[qid]]
        groups = [g for g in groups if g]
        assert groups
        query = row["question"]
        started = time.perf_counter()
        vector = embedding.embed_query(query)
        results = {}
        for mode, candidate_k in VARIANTS:
            variant = f"{mode}_{candidate_k}"
            stage = time.perf_counter()
            candidates = store.search(vector, top_k=candidate_k, paper_id=pid,
                                      query_text=query, mode=mode, candidate_k=candidate_k)
            ranked = reranker.rerank(query, candidates, top_k=5)
            def score_hits(items):
                matched = {base.normalize(p) for p in base.project(
                    [item["text"] for item in items], paragraphs)}
                return int(any(matched & group for group in groups))
            results[variant] = {
                "candidate_hit": score_hits(candidates),
                "hit_at_5": score_hits(ranked),
                "candidate_ids": [c["chunk_id"] for c in candidates],
                "top5_ids": [c["chunk_id"] for c in ranked],
                "top5_chunks": [{"chunk_id": c["chunk_id"], "text": c["text"],
                                 "paper_id": pid, "section": c["metadata"].get("section", "")}
                                for c in ranked],
                "seconds": time.perf_counter() - stage,
            }
        output = {"question_id": qid, "paper_id": pid, "question": query,
                  "reranker_device": device,
                  "variants": results, "total_seconds": time.perf_counter()-started}
        base.write(path, output)
        rows.append(output)
        print(f"{index}/{len(selected)} " + " ".join(
            f"{k}:{v['hit_at_5']}" for k,v in results.items()), flush=True)
    summary = {"questions": len(rows), "cohort": "baseline text evidence missed in Top-5",
               "source_sha256": base.sha(base.SOURCE),
               "collection": COLLECTION, "milvus_signature": description,
               "reranker_devices": dict(Counter(
                   r.get("reranker_device", "cpu") for r in rows)),
               "variants": {f"{mode}_{k}": {
                   "candidate_hit_rate": statistics.mean(r["variants"][f"{mode}_{k}"]["candidate_hit"] for r in rows),
                   "hit_at_5": statistics.mean(r["variants"][f"{mode}_{k}"]["hit_at_5"] for r in rows),
                   "mean_seconds_cuda": (statistics.mean(r["variants"][f"{mode}_{k}"]["seconds"]
                       for r in rows if r.get("reranker_device") == "cuda")
                       if any(r.get("reranker_device") == "cuda" for r in rows) else None),
               } for mode,k in VARIANTS},
               "note": "Diagnostic subset selected for failure; gains cannot be generalized to full 200 without full paired evaluation."}
    base.write(RESULTS / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=41)
    p.add_argument("--uri", default="http://127.0.0.1:19532")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()
    print(json.dumps(run(args.limit, args.uri, args.device), ensure_ascii=False, indent=2))
