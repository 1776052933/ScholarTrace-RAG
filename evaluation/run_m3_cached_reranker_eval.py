"""Run the default BGE reranker on the frozen 30-question QASPER dev index.

Reuses the prior Chroma index, verifies question IDs and baseline Top-5 chunk
IDs, then compares dense Top-5 with reranked Top-5 from the same Top-20 pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from rag_reader.embedding import EmbeddingModel
from rag_reader.reranker import CrossEncoderReranker
from rag_reader.vector_store import VectorStore
from .m0_metrics import evidence_recall
from .run_m0_qasper import select_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-run", type=Path, default=Path("evaluation/runs/m3_accept_20261017_30b"))
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/qasper/qasper-dev-v0.3.json"))
    parser.add_argument("--embedding-model", default="/app/data/m3/bge_m3_model")
    parser.add_argument("--reranker-model", default="/app/data/m3/reranker_hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs/m3_bge_reranker_30dev_20261017/report.json"))
    args = parser.parse_args()
    prior_manifest = json.loads((args.prior_run / "manifest.json").read_text(encoding="utf-8"))
    if prior_manifest["split"] != "dev" or prior_manifest["limit"] != 30:
        parser.error("Expected the frozen 30-question QASPER dev run")
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if dataset_hash != prior_manifest["dataset_sha256"]:
        parser.error("QASPER dataset SHA-256 differs from the prior run")
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected = select_questions(data, 30, 42)
    if [item["question_id"] for _, item in selected] != prior_manifest["question_ids"]:
        parser.error("Fixed QASPER question IDs differ from the prior run")
    old_records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((args.prior_run / "records").glob("*.json"))]
    if len(old_records) != 30:
        parser.error("Prior run does not have all 30 question records")
    model = EmbeddingModel(model_name=args.embedding_model, device="cpu", batch_size=16)
    store = VectorStore(collection_name="m3_qasper_dense", persist_directory=str(args.prior_run / "chroma"))
    reranker = CrossEncoderReranker(model_name=args.reranker_model)
    started = perf_counter()
    reranker._get_model()
    load_seconds = perf_counter() - started
    records = []
    for index, ((paper_id, question), old) in enumerate(zip(selected, old_records), 1):
        if old["paper_id"] != paper_id or old["question_id"] != question["question_id"]:
            parser.error(f"Prior question mismatch at item {index}")
        started = perf_counter()
        vector = model.embed(question["question"])
        candidates = store.search(vector, top_k=20, paper_id=paper_id)
        search_seconds = perf_counter() - started
        if len(candidates) < 5:
            raise RuntimeError(f"Too few candidates for {paper_id}")
        baseline_ids = [item["chunk_id"] for item in candidates[:5]]
        old_ids = [item["chunk_id"] for item in old["variants"]["chroma_dense"]["retrieved"]]
        baseline_matches_prior = baseline_ids == old_ids
        started = perf_counter()
        ranked = reranker.rerank(question["question"], candidates, top_k=5)
        rerank_seconds = perf_counter() - started
        baseline_score = evidence_recall(question.get("answers") or [], [item["text"] for item in candidates[:5]])
        reranked_score = evidence_recall(question.get("answers") or [], [item["text"] for item in ranked])
        records.append({"paper_id": paper_id, "question_id": question["question_id"],
                        "question": question["question"], "baseline_matches_prior": baseline_matches_prior,
                        "search_seconds_including_embedding": round(search_seconds, 4),
                        "rerank_seconds": round(rerank_seconds, 4),
                        "dense_top5": {"evidence_recall_proxy": baseline_score, "chunk_ids": baseline_ids},
                        "reranked_top5": {"evidence_recall_proxy": reranked_score,
                                          "chunk_ids": [item["chunk_id"] for item in ranked],
                                          "rerank_scores": [item["metadata"]["rerank_score"] for item in ranked]}})
        print(f"{index}/30 {question['question_id']} dense={baseline_score} reranked={reranked_score}", flush=True)
    valid = [item for item in records if item["dense_top5"]["evidence_recall_proxy"] is not None
             and item["reranked_top5"]["evidence_recall_proxy"] is not None]
    differences = [item["reranked_top5"]["evidence_recall_proxy"] - item["dense_top5"]["evidence_recall_proxy"]
                   for item in valid]
    timing = sorted(item["rerank_seconds"] for item in records)
    result = {
        "task": "m3_bge_reranker_fixed_qasper_dev30",
        "prior_manifest_sha256": hashlib.sha256((args.prior_run / "manifest.json").read_bytes()).hexdigest(),
        "dataset_sha256": dataset_hash,
        "embedding_weights_sha256": hashlib.sha256((Path(args.embedding_model) / "model.safetensors").read_bytes()).hexdigest(),
        "reranker_weights_sha256": hashlib.sha256((Path(args.reranker_model) / "model.safetensors").read_bytes()).hexdigest(),
        "model_load_seconds": round(load_seconds, 4),
        "summary": {"questions": len(records), "scorable_questions": len(valid),
                    "baseline_matches_prior": sum(item["baseline_matches_prior"] for item in records),
                    "dense_evidence_recall_proxy": sum(item["dense_top5"]["evidence_recall_proxy"] for item in valid) / len(valid),
                    "reranked_evidence_recall_proxy": sum(item["reranked_top5"]["evidence_recall_proxy"] for item in valid) / len(valid),
                    "rerank_wins": sum(diff > 0 for diff in differences),
                    "rerank_losses": sum(diff < 0 for diff in differences),
                    "rerank_ties": sum(diff == 0 for diff in differences),
                    "rerank_p50_seconds": median(timing),
                    "rerank_p95_seconds": timing[max(0, int(0.95 * (len(timing) - 1)))]},
        "records": records,
        "limits": ["Evidence recall is the existing M0 token-coverage proxy, not official QASPER evidence F1.",
                   "This run tests the default reranker on Chroma dense Top-20, not Milvus hybrid or generated answers.",
                   "Model load time is reported separately from per-query rerank inference."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result["summary"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
