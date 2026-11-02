"""Current-core benchmark entry point. Default phase is a read-only plan."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit, urlunsplit

from rag_reader.config import RAGSettings
from .current_datasets import load_dataset
from .m0_common import DATA, ROOT, git_head, require_real_embedding, sha256, utc_now, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=["qasper", "litsearch", "scholarqa-cs"])
    p.add_argument("--phase", choices=["plan", "index", "retrieve", "answer"], default="plan")
    p.add_argument("--data-root", type=Path, default=DATA)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--limit", type=int, default=0, help="0 selects all questions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--corpus-limit", type=int, default=0)
    p.add_argument("--corpus-jsonl", type=Path)
    p.add_argument("--corpus-provenance", default="")
    p.add_argument("--backend", choices=["milvus", "chroma"])
    p.add_argument("--retrieval-mode", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--allow-model-calls", action="store_true")
    p.add_argument("--max-answers", type=int, default=0,
                   help="Maximum new question attempts this invocation, not an HTTP retry limit")
    return p


def settings_for(args):
    values = RAGSettings.from_env().model_dump()
    values["vector_backend"] = args.backend or os.getenv("RAG_VECTOR_BACKEND") or "milvus"
    values["collection_name"] = "eval_current_" + hashlib.sha256(
        str(args.run_dir.resolve()).encode()).hexdigest()[:24]
    if args.retrieval_mode:
        values["retrieval_mode"] = args.retrieval_mode
    if args.top_k is not None:
        values["retrieval_top_k"] = args.top_k
    settings = RAGSettings.model_validate(values)
    if settings.vector_backend == "chroma" and settings.retrieval_mode != "dense":
        raise ValueError("BM25/hybrid requires Milvus")
    return settings


def public_settings(settings):
    values = settings.model_dump(exclude={"milvus_token"})
    for key in ("milvus_uri", "llm_base_url"):
        parsed = urlsplit(values[key])
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f"Use credentials outside {key}; credential URLs cannot be recorded")
        values[key] = urlunsplit(parsed)
    return values


def code_digest():
    paths = set()
    for folder in ("rag_reader", "ai_components", "evaluation"):
        paths.update((ROOT / folder).glob("*.py"))
    paths.update([ROOT / "app/config.py", ROOT / "app/papers/schema.py"])
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def execute(args, *, runtime_factory=None):
    # Evaluation payloads stay local unless answer generation was explicitly enabled.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_API_KEY"] = ""
    if args.phase == "answer":
        if args.dataset == "litsearch":
            raise ValueError("LitSearch supports retrieval, not answer generation")
        if not args.allow_model_calls or args.max_answers < 1:
            raise ValueError("answer requires --allow-model-calls and positive --max-answers")
    dataset = load_dataset(args.dataset, args.data_root, split=args.split, limit=args.limit,
                           seed=args.seed, corpus_limit=args.corpus_limit,
                           corpus=args.corpus_jsonl, provenance=args.corpus_provenance)
    if not dataset.cases or len({c.case_id for c in dataset.cases}) != len(dataset.cases):
        raise ValueError("Questions must be non-empty and have unique IDs")
    settings = settings_for(args)
    plan = {"dataset": dataset.name, "ready": dataset.ready,
            "questions": len(dataset.cases), "protocol": dataset.protocol,
            "settings": public_settings(settings), "external_tracing": False}
    if args.phase == "plan":
        return plan
    if not dataset.ready:
        raise ValueError("Dataset needs an independent source corpus; inspect --phase plan")
    manifest = {**plan, "case_ids": [c.case_id for c in dataset.cases],
                "files": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in dataset.files],
                "code_sha256": code_digest(),
                "evidence_environment": {k: os.getenv(k, default) for k, default in (
                    ("RAG_EVIDENCE_STRATEGY", "coverage"),
                    ("RAG_EVIDENCE_MAX_CHUNKS", "10"),
                    ("RAG_EVIDENCE_MAX_EXPANSIONS", "1"))}}
    directory = args.run_dir.resolve()
    with run_lock(directory):
        path = directory / "manifest.json"
        fresh = not path.exists()
        if fresh:
            if args.phase != "index":
                raise ValueError("Run index before retrieval/answers")
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("Unowned non-empty run directory")
        elif json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Dataset/code/settings changed; use a new run directory")
        if runtime_factory is None:
            from .current_runtime import CurrentRuntime
            runtime_factory = CurrentRuntime
        runtime = runtime_factory(settings, directory, generation=args.phase == "answer")
        if fresh:
            runtime.ensure_new_index()
            write_json(path, manifest)
            write_json(directory / "references.json", dataset.references)
            write_json(directory / "origin.json", {"git_head": git_head(), "created_at": utc_now()})
        if hasattr(runtime, "reader"):
            from .m0_common import embedding_revision, runtime_info
            backend = require_real_embedding(runtime.reader.embedding_model)
            write_json(directory / f"runtime_{args.phase}.json", {
                **runtime_info(), "embedding_backend": backend,
                "embedding_revision": embedding_revision(runtime.reader.embedding_model)})
        with sqlite3.connect(directory / "progress.sqlite3") as db:
            db.execute("CREATE TABLE IF NOT EXISTS papers (id TEXT PRIMARY KEY, digest TEXT, chunks TEXT)")
            indexed = {row[0]: (row[1], json.loads(row[2]))
                       for row in db.execute("SELECT id,digest,chunks FROM papers")}
            for _, ids in indexed.values():
                if not runtime.verify_ids(ids):
                    raise ValueError("Checkpoint vectors are missing; use a new run directory")
            if args.phase == "index":
                seen, batch = set(), []

                def flush():
                    result = runtime.index_sources([s for s, _ in batch])
                    for source, digest in batch:
                        ids = result[source.paper_id]
                        if not ids:
                            raise ValueError("Paper produced no chunks")
                        db.execute("INSERT INTO papers VALUES (?,?,?)",
                                   (source.paper_id, digest, json.dumps(ids)))
                    db.commit()
                    write_json(directory / "index_progress.json", {
                        "papers": db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
                        "updated_at": utc_now(),
                    })
                    batch.clear()

                for source in dataset.sources():
                    if source.paper_id in seen:
                        raise ValueError("Duplicate source paper_id")
                    seen.add(source.paper_id)
                    digest = hashlib.sha256(json.dumps(
                        source.__dict__, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if source.paper_id in indexed:
                        if indexed[source.paper_id][0] != digest:
                            raise ValueError("Source content changed")
                        continue
                    batch.append((source, digest))
                    if len(batch) >= (64 if dataset.name == "litsearch" else 1):
                        flush()
                if batch:
                    flush()
                if not seen or set(indexed) - seen:
                    raise ValueError("Empty or inconsistent corpus")
                report = {"status": "indexed", "papers": len(seen)}
                write_json(directory / "index_complete.json", report)
                return report
            complete = directory / "index_complete.json"
            if not complete.exists() or json.loads(complete.read_text())["papers"] != len(indexed):
                raise ValueError("Index is incomplete")
            results = directory / args.phase
            results.mkdir(exist_ok=True)
            attempts, statuses = 0, Counter()
            for case in dataset.cases:
                output = results / (hashlib.sha256(case.case_id.encode()).hexdigest() + ".json")
                if output.exists():
                    statuses[json.loads(output.read_text(encoding="utf-8"))["status"]] += 1
                    continue  # Includes failed/started: never silently repeat a paid request.
                if args.phase == "answer" and attempts >= args.max_answers:
                    statuses["pending"] += 1
                    continue
                record = {"case_id": case.case_id, "question": case.question,
                          "status": "started", "started_at": utc_now()}
                write_json(output, record)
                attempts += 1
                started = time.monotonic()
                try:
                    result = runtime.run_case(case, dataset.name, answer=args.phase == "answer")
                    record.update(result)
                    record["status"] = result.get("status", "completed")
                except Exception as exc:
                    record.update(status="failed", error_type=type(exc).__name__)
                record["seconds"] = time.monotonic() - started
                write_json(output, record)
                statuses[record["status"]] += 1
            summary = {"questions": len(dataset.cases), "statuses": dict(statuses),
                       "new_attempts": attempts, "quality_scores": None}
            write_json(directory / f"summary_{args.phase}.json", summary)
            return summary


def main():
    args = parser().parse_args()
    print(json.dumps(execute(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
