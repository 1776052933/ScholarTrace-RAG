"""M3 fixed QASPER retrieval ablation; no gold evidence enters the index.

Example:
  python -m evaluation.run_m3_ablation --limit 10 --seed 42 --top-k 5
  python -m evaluation.run_m3_ablation --limit 10 --skip-milvus --skip-reranker
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from rag_reader.cleaner import TextCleaner
from rag_reader.embedding import EmbeddingModel, embedding_fingerprint
from rag_reader.milvus_store import MilvusVectorStore
from rag_reader.reranker import CrossEncoderReranker
from rag_reader.retriever import PaperRetriever
from rag_reader.schema import PaperDocument
from rag_reader.splitter import DocumentSplitter
from rag_reader.vector_store import VectorStore

from .m0_common import DATA, RUNS, code_hash, embedding_revision, git_head, require_real_embedding, runtime_info, sha256, utc_now, write_json
from .m0_metrics import evidence_recall
from .run_m0_qasper import paper_text, select_questions


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({name for record in records for name in record["variants"]})
    output = {}
    for name in variants:
        successful = [record["variants"][name] for record in records
                      if record["variants"].get(name, {}).get("status") == "completed"]
        evidence = [item["evidence_recall_proxy"] for item in successful
                    if item["evidence_recall_proxy"] is not None]
        times = sorted(item["seconds"] for item in successful)
        output[name] = {
            "question_count": len(records),
            "completed": len(successful),
            "failed": len(records) - len(successful),
            "evidence_recall_proxy": sum(evidence) / len(evidence) if evidence else None,
            "latency_p50_seconds": median(times) if times else None,
            "latency_p95_seconds": times[max(0, int((len(times) - 1) * 0.95))] if times else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=10, help="0 = all questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--collection", default="m3_qasper_dev")
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")))
    parser.add_argument("--reranker-model", default=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.top_k < 1 or args.candidate_k < args.top_k or args.embedding_batch_size < 1:
        parser.error("limit >= 0, candidate-k >= top-k >= 1 and embedding-batch-size >= 1 are required")
    dataset = DATA / "qasper" / f"qasper-{args.split}-v0.3.json"
    if not dataset.is_file():
        parser.error(f"Missing QASPER dataset: {dataset}")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    selected = select_questions(data, args.limit, args.seed)
    if not selected:
        parser.error("No questions selected")
    run_dir = args.run_dir or RUNS / f"m3_ablation_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = EmbeddingModel(model_name=args.embedding_model, batch_size=args.embedding_batch_size)
    require_real_embedding(model)
    fingerprint = embedding_fingerprint(model)
    splitter = DocumentSplitter()
    cleaner = TextCleaner()
    chroma = VectorStore(collection_name="m3_qasper_dense", persist_directory=str(run_dir / "chroma"))
    milvus = None if args.skip_milvus else MilvusVectorStore(
        collection_name=args.collection, uri=args.milvus_uri,
        token=os.getenv("MILVUS_TOKEN"),
        pipeline_signature=f"qasper-structured-{splitter.chunk_size}-{splitter.chunk_overlap}-v1",
    )
    reranker = None if args.skip_reranker else CrossEncoderReranker(model_name=args.reranker_model)
    variants: dict[str, PaperRetriever] = {
        "chroma_dense": PaperRetriever(model, chroma, default_top_k=args.top_k),
    }
    if reranker is not None:
        variants["chroma_dense_rerank"] = PaperRetriever(
            model, chroma, default_top_k=args.top_k,
            candidate_k=args.candidate_k, reranker=reranker,
        )
    if milvus is not None:
        for mode in ("dense", "bm25", "hybrid"):
            variants[f"milvus_{mode}"] = PaperRetriever(
                model, milvus, default_top_k=args.top_k,
                retrieval_mode=mode, candidate_k=args.candidate_k,
            )
        if reranker is not None:
            variants["milvus_hybrid_rerank"] = PaperRetriever(
                model, milvus, default_top_k=args.top_k,
                retrieval_mode="hybrid", candidate_k=args.candidate_k,
                reranker=reranker,
            )
    manifest = {
        "task": "m3_qasper_fixed_retrieval_ablation",
        "started_at_utc": utc_now(),
        "git_head": git_head(),
        "code_sha256": code_hash(),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "runtime": runtime_info(),
        "dataset_sha256": sha256(dataset),
        "split": args.split, "seed": args.seed, "limit": args.limit,
        "question_ids": [row["question_id"] for _, row in selected],
        "embedding_model": model.model_name,
        "embedding_batch_size": model.batch_size,
        "embedding_revision": embedding_revision(model),
        "embedding_fingerprint": fingerprint,
        "reranker_model": args.reranker_model if reranker is not None else None,
        "chunk_size": splitter.chunk_size, "chunk_overlap": splitter.chunk_overlap,
        "top_k": args.top_k, "candidate_k": args.candidate_k,
        "variants": list(variants), "milvus_collection": args.collection if milvus else None,
        "metric_note": "Evidence recall is the M0 token-coverage proxy, not official QASPER evidence F1. QASPER has structured text, not PDF page ground truth.",
    }
    write_json(run_dir / "manifest.json", manifest)

    # Index each selected paper once using identical cleaned chunks and embeddings.
    indexed = set()
    ingest_seconds = 0.0
    for paper_id, _question in selected:
        if paper_id in indexed:
            continue
        started = perf_counter()
        content = paper_text(data[paper_id])
        pages = cleaner.clean_page_records([{"page": None, "text": content}])
        document = PaperDocument(paper_id=paper_id, title=data[paper_id]["title"], content=content)
        chunks = splitter.split(document, pages=pages)
        if not chunks:
            raise RuntimeError(f"no chunks for paper {paper_id}")
        vectors = model.embed_chunks(chunks)
        for store in (chroma, milvus):
            if store is None:
                continue
            store.ensure_embedding_compatible(fingerprint, len(vectors[0]))
            store.delete(paper_id=paper_id)
            if store.add_documents(chunks, vectors) != len(chunks):
                raise RuntimeError(f"indexing failed: {paper_id}")
            if hasattr(store, "verify_documents") and not store.verify_documents(chunks):
                raise RuntimeError(f"Milvus readback failed: {paper_id}")
        indexed.add(paper_id)
        ingest_seconds += perf_counter() - started
        print(f"Indexed {len(indexed)} papers", flush=True)

    records = []
    for index, (paper_id, question) in enumerate(selected, 1):
        record = {
            "paper_id": paper_id,
            "question_id": question["question_id"],
            "question": question["question"],
            "variants": {},
        }
        for name, retriever in variants.items():
            started = perf_counter()
            try:
                chunks = retriever.retrieve(paper_id, question["question"], top_k=args.top_k)
                record["variants"][name] = {
                    "status": "completed",
                    "seconds": perf_counter() - started,
                    "retrieved": [{
                        "chunk_id": item.chunk_id, "page": item.page,
                        "section": item.section, "text": item.text,
                        "score": item.metadata.get("retrieval_score"),
                    } for item in chunks],
                    "evidence_recall_proxy": evidence_recall(
                        question.get("answers") or [], [item.text for item in chunks]
                    ),
                }
            except Exception as exc:
                record["variants"][name] = {
                    "status": "failed", "seconds": perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        records.append(record)
        write_json(run_dir / "records" / f"{index:04d}_{question['question_id']}.json", record)
        print(f"{index}/{len(selected)} {question['question_id']}", flush=True)
    write_json(run_dir / "summary.json", {
        "finished_at_utc": utc_now(), "paper_count": len(indexed),
        "ingest_seconds": ingest_seconds, "variants": _summary(records),
    })
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
