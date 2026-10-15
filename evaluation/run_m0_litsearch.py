"""M0 LitSearch fixed-corpus title/abstract dense-retrieval baseline.

Run with: python -m evaluation.run_m0_litsearch --corpus-limit 100 --query-limit 5
Use --corpus-limit 0 --query-limit 0 for the full fixed corpus and query split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterator

import pyarrow.parquet as pq

from rag_reader.embedding import EmbeddingModel, embedding_fingerprint
from rag_reader.retriever import PaperRetriever
from rag_reader.schema import DocumentChunk
from rag_reader.vector_store import VectorStore

from .litsearch import load_queries
from .metrics import evaluate_query
from .m0_common import DATA, code_hash, embedding_revision, git_head, new_run_dir, require_real_embedding, runtime_info, sha256, summarize, utc_now, write_json


def corpus_rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=128, columns=["corpusid", "title", "abstract"]):
            yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-limit", type=int, default=100, help="0 = entire fixed corpus")
    parser.add_argument("--query-limit", type=int, default=5, help="0 = all 597 queries")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.corpus_limit < 0 or args.query_limit < 0 or args.top_k < 1:
        parser.error("limits must be >= 0 and top-k must be >= 1")
    files = sorted((DATA / "litsearch" / "corpus_clean").glob("*.parquet"))
    query_file = DATA / "litsearch" / "query" / "full-00000-of-00001.parquet"
    if len(files) != 6 or not query_file.is_file():
        parser.error("Missing fixed LitSearch corpus or query files under data/benchmarks/litsearch")
    full_corpus_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if args.resume:
        if not args.run_dir:
            parser.error("--resume requires --run-dir")
        run_dir = Path(args.run_dir).resolve()
        if not (run_dir / "manifest.json").is_file():
            parser.error("No manifest.json in resume directory")
    else:
        run_dir = new_run_dir("litsearch", args.run_dir)

    queries = load_queries(query_file)
    selected = queries[:args.query_limit] if args.query_limit else queries
    model = EmbeddingModel()
    backend = require_real_embedding(model)
    corpus_target = min(args.corpus_limit or full_corpus_rows, full_corpus_rows)
    config = {"corpus_limit": args.corpus_limit, "query_limit": args.query_limit, "top_k": args.top_k}
    manifest = {
        "task": "litsearch_fixed_title_abstract_dense", "started_at_utc": utc_now(),
        "git_head": git_head(), "code_sha256": code_hash(), "runtime": runtime_info(),
        "config": config,
        "corpus": {"path": "litsearch/corpus_clean", "files": [p.name for p in files],
                   "available_rows": full_corpus_rows, "target_rows": corpus_target,
                   "revision": "9573fb284a1026c998df47024b888a163f0f0e25"},
        "queries": {"path": "litsearch/query/full-00000-of-00001.parquet",
                    "sha256": sha256(query_file), "total": len(queries),
                    "selected_ids": [q["query_id"] for q in selected]},
        "index": {"store": "Chroma", "model": model.model_name, "backend": backend, "model_revision": embedding_revision(model),
                  "text_fields": ["title", "abstract"], "top_k": args.top_k},
        "scope": "full_fixed_corpus" if corpus_target == full_corpus_rows and len(selected) == len(queries) else "sample",
        "metric_note": "This is a fixed-corpus BGE-M3 dense baseline, not the live SearchAgent and not an official LitSearch leaderboard run.",
    }
    if args.resume:
        previous = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous["config"] != config or previous["code_sha256"] != manifest["code_sha256"] or previous["queries"]["sha256"] != manifest["queries"]["sha256"] or previous["runtime"] != manifest["runtime"] or previous["index"] != manifest["index"] or previous["corpus"] != manifest["corpus"]:
            parser.error("Resume config, code, or query data differs from original run")
        manifest = previous
    else:
        write_json(run_dir / "manifest.json", manifest)

    store = VectorStore(persist_directory=str(run_dir / "chroma"), collection_name="litsearch_m0")
    retriever = PaperRetriever(embedding_model=model, vector_store=store)
    state_file = run_dir / "index_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {"processed_rows": 0, "indexed_documents": 0}
    indexed_ids: set[int] = set()
    pending: list[DocumentChunk] = []
    processed = 0
    indexed = state["indexed_documents"]
    index_start = time.perf_counter()

    def flush() -> None:
        nonlocal indexed
        if not pending:
            return
        vectors = model.embed_chunks(pending)
        store.ensure_embedding_compatible(embedding_fingerprint(model), len(vectors[0]))
        store.add_documents(pending, vectors)
        indexed += len(pending)
        pending.clear()
        write_json(state_file, {"processed_rows": processed, "indexed_documents": indexed})
        print(f"Indexed {processed}/{corpus_target} rows", flush=True)

    for row in corpus_rows(files):
        if processed >= corpus_target:
            break
        processed += 1
        corpus_id = row.get("corpusid")
        if corpus_id is not None:
            indexed_ids.add(int(corpus_id))
        if processed <= state["processed_rows"]:
            continue
        title, abstract = str(row.get("title") or "").strip(), str(row.get("abstract") or "").strip()
        if corpus_id is not None and (title or abstract):
            paper_id = f"litsearch-{int(corpus_id)}"
            pending.append(DocumentChunk(chunk_id=f"{paper_id}-chunk-00001", paper_id=paper_id,
                                         text="\n\n".join(part for part in [title, abstract] if part),
                                         metadata={"title": title, "corpusid": int(corpus_id)}))
        if len(pending) >= 32 or processed % 32 == 0:
            flush()
        elif processed > state["processed_rows"] and not pending:
            write_json(state_file, {"processed_rows": processed, "indexed_documents": indexed})
    flush()
    index_seconds = time.perf_counter() - index_start
    if processed != corpus_target:
        raise RuntimeError(f"Only saw {processed} of {corpus_target} requested corpus rows")

    records: list[dict[str, Any]] = []
    for number, query in enumerate(selected, 1):
        path = run_dir / "records" / f"{query['query_id']}.json"
        if args.resume and path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        start = time.perf_counter()
        record: dict[str, Any] = {"status": "failed", "query_id": query["query_id"],
                                  "query": query["query"], "gold_corpus_ids": query["gold_corpus_ids"],
                                  "gold_present_in_index": [id_ for id_ in query["gold_corpus_ids"] if id_ in indexed_ids],
                                  "metrics": {}}
        try:
            chunks = retriever.retrieve_all(query["query"], top_k=args.top_k)
            predicted = [int(chunk.paper_id.removeprefix("litsearch-")) for chunk in chunks]
            record["predicted_corpus_ids"] = predicted
            record["predicted"] = [{"corpusid": int(chunk.paper_id.removeprefix("litsearch-")),
                                    "score": chunk.metadata.get("retrieval_score"),
                                    "title": chunk.metadata.get("title")} for chunk in chunks]
            record["metrics"] = evaluate_query(query["gold_corpus_ids"], predicted,
                                                [k for k in (5, 10, 20) if k <= args.top_k])
            record["status"] = "completed"
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=3)}
        record["total_seconds"] = time.perf_counter() - start
        records.append(record)
        write_json(path, record)
        print(f"{number}/{len(selected)} {record['status']} {query['query_id']} {record['total_seconds']:.1f}s", flush=True)
    names = sorted({name for record in records for name in record.get("metrics", {})})
    summary = summarize(records, names)
    summary.update({"task": manifest["task"], "scope": manifest["scope"],
                    "corpus_available_rows": full_corpus_rows, "corpus_processed_rows": processed,
                    "indexed_documents": indexed, "index_seconds_this_run": index_seconds,
                    "gold_coverage_in_index": sum(len(r["gold_present_in_index"]) for r in records) /
                        max(1, sum(len(r["gold_corpus_ids"]) for r in records)),
                    "finished_at_utc": utc_now()})
    write_json(run_dir / "summary.json", summary)
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
