"""Paired QASPER evidence-budget experiment on frozen dense-20 candidates.

Only the scoring functions read gold annotations. Retrieval is frozen by chunk
IDs before reranking, and all generation requests are persisted before sending.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from openai import OpenAI

from . import qasper_followup as base
from .qasper_holdout import PATH as HOLDOUT
from .qasper_prompt_ablation import PROMPT_V3
from .qasper_retrieval_ablation import COLLECTION, RERANKER_MODEL

ROOT = base.NEW / "evidence_budget"
VARIANTS = ("top5", "top8", "top10", "merge8")
INDEX = re.compile(r"^(.*)-chunk-(\d+)$")


def frozen_cases(cohort: str) -> list[dict]:
    if cohort == "200":
        source = base.read(base.NEW / "reranker_ablation" / "cases.json")
        return [{"question_id": row["question_id"], "paper_id": row["paper_id"],
                 "question": None, "candidate_ids": row["raw_dense_candidate20_ids"]}
                for row in source]
    if cohort == "holdout60":
        frozen = base.read(HOLDOUT)
        return [{"question_id": row["question_id"], "paper_id": row["paper_id"],
                 "question": row["question"],
                 "candidate_ids": base.read(base.NEW / "holdout_retrieval" /
                                             f"{row['question_id']}.json")["candidate_ids"]}
                for row in frozen["cases"]]
    raise ValueError(cohort)


def merge_adjacent(chunks: list[dict], *, max_chars: int = 2200) -> list[dict]:
    """Merge only selected neighboring chunks on the same page and section."""
    indexed = []
    for rank, chunk in enumerate(chunks):
        match = INDEX.fullmatch(chunk["chunk_id"])
        indexed.append((rank, int(match.group(2)) if match else None, chunk))
    pending = sorted(indexed, key=lambda item: (item[2]["paper_id"], item[1] or -1))
    groups: list[list[tuple[int, int | None, dict]]] = []
    for item in pending:
        if groups:
            previous = groups[-1][-1]
            group_text = "\n".join(x[2]["text"] for x in groups[-1])
            if (item[1] is not None and previous[1] is not None and
                item[1] == previous[1] + 1 and
                item[2]["paper_id"] == previous[2]["paper_id"] and
                item[2].get("section") == previous[2].get("section") and
                item[2].get("page") == previous[2].get("page") and
                len(group_text) + len(item[2]["text"]) + 1 <= max_chars):
                groups[-1].append(item)
                continue
        groups.append([item])
    groups.sort(key=lambda group: min(item[0] for item in group))
    output = []
    for group in groups:
        members = [item[2] for item in group]
        merged = dict(members[0])
        merged["source_chunk_ids"] = [item["chunk_id"] for item in members]
        if len(members) > 1:
            merged["chunk_id"] = "merge:" + "+".join(merged["source_chunk_ids"])
            merged["text"] = "\n".join(item["text"] for item in members)
        output.append(merged)
    return output


def variant_chunks(ranked: list[dict], variant: str) -> list[dict]:
    if variant == "merge8":
        return merge_adjacent(ranked[:8])
    if variant in {"top5", "top8", "top10"}:
        return ranked[:int(variant[3:])]
    raise ValueError(variant)


def constituent_texts(chunks: list[dict], ranked: list[dict]) -> list[str]:
    """Score a merged span through the original chunks it actually contains."""
    raw_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in ranked}
    return [raw_by_id[chunk_id] for chunk in chunks
            for chunk_id in chunk.get("source_chunk_ids", [chunk["chunk_id"]])]


def run_retrieval(cohort: str, uri: str = "http://127.0.0.1:19532",
                  device: str = "cpu", limit: int | None = None) -> dict:
    from rag_reader.milvus_store import MilvusVectorStore
    from rag_reader.reranker import CrossEncoderReranker
    from sentence_transformers import CrossEncoder

    cases = frozen_cases(cohort)
    if limit is not None:
        cases = cases[:limit]
    _, _, _, _, _, question_map = base.original()
    store = MilvusVectorStore(COLLECTION, uri=uri,
                              pipeline_signature="page-section-1000-200-pymupdf-v1")
    store.prepare_bm25()
    reranker = CrossEncoderReranker(str(RERANKER_MODEL),
                  model=CrossEncoder(str(RERANKER_MODEL), max_length=512, device=device))
    statuses = Counter()
    for i, case in enumerate(cases, 1):
        qid = case["question_id"]
        path = ROOT / cohort / "retrieval" / f"{qid}.json"
        if path.exists():
            previous = base.read(path)
            if previous["status"] == "completed":
                statuses["completed"] += 1
                continue
        question = case["question"] or question_map[qid][2]["question"]
        record = {"question_id": qid, "paper_id": case["paper_id"],
                  "question": question, "candidate_ids": case["candidate_ids"],
                  "candidate_ids_sha256": hashlib.sha256(json.dumps(case["candidate_ids"]).encode()).hexdigest(),
                  "status": "started", "device": device, "collection": COLLECTION}
        base.write(path, record)
        start = time.perf_counter()
        try:
            rows = store.client.get(collection_name=COLLECTION, ids=case["candidate_ids"],
                                    output_fields=["chunk_id", "paper_id", "text", "page", "section", "metadata"])
            found = {row["chunk_id"]: row for row in rows}
            if set(found) != set(case["candidate_ids"]):
                raise ValueError("Frozen candidate IDs differ from collection")
            candidates = []
            for chunk_id in case["candidate_ids"]:
                row = found[chunk_id]
                if row["paper_id"] != case["paper_id"]:
                    raise ValueError("Candidate outside paper scope")
                candidates.append({"chunk_id": chunk_id, "text": row["text"],
                                   "score": 0.0, "metadata": row.get("metadata") or {}})
            ranked = reranker.rerank(question, candidates, top_k=10)
            record["ranked"] = [{"chunk_id": row["chunk_id"], "paper_id": case["paper_id"],
                                 "section": found[row["chunk_id"]]["section"],
                                 "page": found[row["chunk_id"]]["page"],
                                 "text": row["text"],
                                 "rerank_score": row["metadata"]["raw_rerank_score"]}
                                for row in ranked]
            record["status"] = "completed"
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)[:300]
        record["seconds"] = time.perf_counter() - start
        base.write(path, record)
        statuses[record["status"]] += 1
        if i % 10 == 0 or record["status"] != "completed":
            print(f"{cohort} rerank {i}/{len(cases)} {record['status']} {record['seconds']:.1f}s", flush=True)
    return {"cohort": cohort, "cases": len(cases), "statuses": dict(statuses)}


def score_retrieval(cohort: str) -> dict:
    cases = frozen_cases(cohort)
    source = base.read(base.SOURCE)
    gold = base.official().get_answers_and_evidence(source, False)
    results = {v: [] for v in VARIANTS}
    for case in cases:
        qid, pid = case["question_id"], case["paper_id"]
        record = base.read(ROOT / cohort / "retrieval" / f"{qid}.json")
        if record["status"] != "completed":
            raise ValueError(f"Incomplete retrieval: {qid}")
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        keys = {base.normalize(p) for p in paragraphs}
        groups = [set(base.normalize(p) for p in ref["evidence"] if base.normalize(p) in keys)
                  for ref in gold[qid]]
        groups = [g for g in groups if g]
        for variant in VARIANTS:
            chunks = variant_chunks(record["ranked"], variant)
            projected = {base.normalize(p) for p in base.project(
                constituent_texts(chunks, record["ranked"]), paragraphs)}
            prompt_chunks = [{k: c[k] for k in ("chunk_id", "paper_id", "section", "text")}
                             for c in chunks]
            results[variant].append({"question_id": qid, "has_text_gold": bool(groups),
                "hit": int(any(projected & g for g in groups)) if groups else None,
                "chunk_count": len(chunks),
                "input_chars": len(json.dumps({"paper_id": pid, "question": record["question"],
                                               "chunks": prompt_chunks}, ensure_ascii=False)),
                "text_chars": sum(len(c["text"]) for c in chunks),
                "rerank_seconds": record["seconds"]})
    summary = {"cohort": cohort, "questions": len(cases), "text_gold_questions":
               sum(row["has_text_gold"] for row in results["top5"]), "variants": {}}
    for variant, rows in results.items():
        eligible = [r for r in rows if r["has_text_gold"]]
        summary["variants"][variant] = {
            "text_hit": sum(r["hit"] for r in eligible),
            "text_hit_rate": statistics.mean(r["hit"] for r in eligible),
            "mean_chunks": statistics.mean(r["chunk_count"] for r in rows),
            "mean_input_chars": statistics.mean(r["input_chars"] for r in rows),
            "mean_text_chars": statistics.mean(r["text_chars"] for r in rows),
            "mean_rerank_seconds": statistics.mean(r["rerank_seconds"] for r in rows)}
        base.write(ROOT / cohort / f"retrieval_cases_{variant}.json", rows)
    base.write(ROOT / cohort / "retrieval_scores.json", summary)
    return summary


def run_answers(cohort: str, variant: str, limit: int | None = None) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    cases = frozen_cases(cohort)
    if limit is not None:
        cases = cases[:limit]
    _, _, _, _, _, question_map = base.original()
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.getenv("NO_PROXY", ""), "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    key = Path("F:/qwen/local_api_key.txt").read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=key, base_url="http://127.0.0.1:18510/v1", timeout=180, max_retries=0)
    statuses = Counter()
    for i, case in enumerate(cases, 1):
        qid = case["question_id"]
        path = ROOT / cohort / "answers" / variant / f"{qid}.json"
        if path.exists():
            statuses[base.read(path)["status"]] += 1
            continue
        retrieval = base.read(ROOT / cohort / "retrieval" / f"{qid}.json")
        if retrieval["status"] != "completed":
            raise ValueError(f"Incomplete retrieval: {qid}")
        chunks = variant_chunks(retrieval["ranked"], variant)
        question = case["question"] or question_map[qid][2]["question"]
        payload = {"paper_id": case["paper_id"], "question": question,
                   "chunks": [{k: c[k] for k in ("chunk_id", "paper_id", "section", "text")}
                              for c in chunks]}
        messages = [{"role": "system", "content": PROMPT_V3},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        record = {"question_id": qid, "variant": variant, "status": "started",
                  "model": "qwen3-4b-local", "requested_at": datetime.now(timezone.utc).isoformat(),
                  "prompt_sha256": hashlib.sha256(PROMPT_V3.encode()).hexdigest(),
                  "messages_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                  "chunks": chunks, "question": question}
        base.write(path, record)
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(model="qwen3-4b-local", messages=messages,
                                                        max_tokens=512, temperature=0)
            record["raw_output"] = response.choices[0].message.content or ""
            record["finish_reason"] = response.choices[0].finish_reason
            record["usage"] = response.usage.model_dump() if response.usage else None
            try:
                record["parsed"] = base.parse_model_json(record["raw_output"],
                    {c["chunk_id"] for c in chunks}, question)
                record["status"] = "completed" if record["finish_reason"] == "stop" else "truncated"
            except ValueError as exc:
                record["status"], record["parse_error"] = "invalid_format", str(exc)
        except Exception as exc:
            record["status"], record["error_type"] = "failed", type(exc).__name__
            record["error"] = str(exc)[:300]
        record["seconds"] = time.perf_counter() - start
        base.write(path, record)
        statuses[record["status"]] += 1
        if i % 10 == 0 or record["status"] != "completed":
            print(f"{cohort} {variant} answer {i}/{len(cases)} {record['status']} {record['seconds']:.1f}s", flush=True)
    return {"cohort": cohort, "variant": variant, "cases": len(cases), "statuses": dict(statuses)}


def score_answers(cohort: str, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    source = base.read(base.SOURCE)
    evaluator = base.official()
    gold = evaluator.get_answers_and_evidence(source, False)
    rows, predictions = [], {}
    for case in frozen_cases(cohort):
        qid, pid = case["question_id"], case["paper_id"]
        if variant == "top5" and cohort == "holdout60":
            record = base.read(base.NEW / "prompt_v3_holdout60" / "qwen" / f"{qid}.json")
            old_inputs = base.read(base.NEW / "prompt_v3_holdout60" / "inputs.json")
            chunks = next(c["chunks"] for c in old_inputs["cases"] if c["question_id"] == qid)
            new_ids = [c["chunk_id"] for c in base.read(ROOT / cohort / "retrieval" / f"{qid}.json")["ranked"][:5]]
            if [c["chunk_id"] for c in chunks] != new_ids:
                raise ValueError("Frozen Top-5 differs; cannot reuse original model answer")
            question = case["question"]
        else:
            record = base.read(ROOT / cohort / "answers" / variant / f"{qid}.json")
            chunks = record["chunks"]
            question = record["question"]
        parsed = record.get("parsed") if record["status"] == "completed" else None
        if parsed:
            try:
                parsed = base.parse_model_json(record["raw_output"],
                    {c["chunk_id"] for c in chunks}, question)
            except ValueError:
                parsed = None
        answer = parsed["short_answer"] if parsed else ""
        retrieval = base.read(ROOT / cohort / "retrieval" / f"{qid}.json")
        source_chunks = {c["chunk_id"]: c for c in chunks}
        selected = [source_chunks[source] for source in (parsed["sources"] if parsed else [])]
        source_text = constituent_texts(selected, retrieval["ranked"])
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        evidence = base.project(source_text, paragraphs) if parsed else []
        predictions[qid] = {"answer": answer, "evidence": evidence}
        types = {r["type"] for r in gold[qid]}
        rows.append({"question_id": qid, "type": next(iter(types)) if len(types) == 1 else "mixed",
            "answer": answer, "gold_answers": list(dict.fromkeys(r["answer"] for r in gold[qid])),
            "status": record["status"], "sources": parsed["sources"] if parsed else [],
            "f1": max(evaluator.token_f1_score(answer, r["answer"]) for r in gold[qid]),
            "input_tokens": (record.get("usage") or {}).get("prompt_tokens", 0),
            "output_tokens": (record.get("usage") or {}).get("completion_tokens", 0),
            "seconds": record.get("seconds")})
    official = evaluator.evaluate({c["question_id"]: gold[c["question_id"]] for c in frozen_cases(cohort)}, predictions)
    result = {"cohort": cohort, "variant": variant, "questions": len(rows),
              "statuses": dict(Counter(row["status"] for row in rows)), "official_scores": official,
              "mean_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
              "mean_answer_seconds": statistics.mean(r["seconds"] for r in rows),
              "note": "Exploratory paired answer comparison on existing QASPER dev papers; not unseen-paper validation."}
    base.write(ROOT / cohort / f"answer_scores_{variant}.json", result)
    base.write(ROOT / cohort / f"answer_cases_{variant}.json", rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["retrieve", "score-retrieve", "answer", "score-answer"])
    parser.add_argument("--cohort", choices=["200", "holdout60"], required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--uri", default="http://127.0.0.1:19532")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.action in {"answer", "score-answer"} and not args.variant:
        parser.error("--variant is required for answers")
    outcome = (run_retrieval(args.cohort, args.uri, args.device, args.limit)
               if args.action == "retrieve" else
               score_retrieval(args.cohort) if args.action == "score-retrieve" else
               run_answers(args.cohort, args.variant, args.limit) if args.action == "answer" else
               score_answers(args.cohort, args.variant))
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
