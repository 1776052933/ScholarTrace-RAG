"""Paired local QASPER answers on fixed five-chunk evidence inputs.

Requests contain only questions and saved chunks. Gold answers/evidence enter
only the separate score command. No automatic retries of started requests.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

from . import qasper_followup as base
from .qasper_phase3_selection import PROJECT, RUNS, read, write
from .qasper_prompt_ablation import PROMPT_V3


NEW = PROJECT / "evaluation/runs/qasper_new_papers_20261022"
OUT = RUNS / "phase34_answers"
RELATED_GUARD = (PROMPT_V3 + "\nAdditional subject check: before answering, identify whether each supporting "
                 "statement is attributed to this paper's authors/system or to a cited earlier study. "
                 "If only an earlier study states the requested result, use Unanswerable. "
                 "Do not discard an otherwise direct fact solely because its section is named Related Work.")
NUMERIC_GUARD = (PROMPT_V3 + "\nAdditional number check: for any count, percentage, score or comparison, "
                 "match the named entity, dataset, metric and unit, and copy the exact stated figure. "
                 "Compute a difference only when both comparable figures are explicitly supplied; "
                 "otherwise use Unanswerable. Do not substitute a different metric or a relative change.")
VARIANTS = ("baseline5", "model_selected5", "related_guard5", "numeric_guard5")


def answer_prompt(variant: str) -> str:
    return {"baseline5": PROMPT_V3, "model_selected5": PROMPT_V3,
            "related_guard5": RELATED_GUARD, "numeric_guard5": NUMERIC_GUARD}[variant]


def cases(cohort: str) -> list[dict]:
    if cohort == "holdout60":
        return read(RUNS / "holdout_60.json")["cases"]
    if cohort == "new_papers":
        return read(NEW / "frozen_manifest.json")["cases"]
    raise ValueError(cohort)


def chunks_for(cohort: str, qid: str, variant: str) -> list[dict]:
    if cohort == "holdout60" and variant in {"baseline5", "related_guard5", "numeric_guard5"}:
        inputs = read(RUNS / "prompt_v3_holdout60/inputs.json")
        return next(case["chunks"] for case in inputs["cases"] if case["question_id"] == qid)
    folder = RUNS / "evidence_budget/holdout60/retrieval" if cohort == "holdout60" else NEW / "retrieval"
    record = read(folder / f"{qid}.json")
    if record["status"] != "completed":
        raise ValueError(f"Retrieval incomplete: {qid}")
    ranked = record["ranked"]
    if variant in {"baseline5", "related_guard5", "numeric_guard5"}:
        return ranked[:5]
    if variant == "model_selected5":
        selected = read(RUNS / "phase3_model_selection" / cohort / f"{qid}.json")
        if selected["status"] == "completed":
            by_id = {chunk["chunk_id"]: chunk for chunk in ranked}
            return [by_id[chunk_id] for chunk_id in selected["selected_ids"]]
        return ranked[:5]
    raise ValueError(variant)


def input_for(case: dict, cohort: str, variant: str, source: dict) -> tuple[str, list[dict], list[dict]]:
    pid, qid = case["paper_id"], case["question_id"]
    question = next(q["question"] for q in source[pid]["qas"] if q["question_id"] == qid)
    chunks = chunks_for(cohort, qid, variant)
    if len(chunks) != 5 or any(chunk["paper_id"] != pid for chunk in chunks):
        raise ValueError(f"Five in-scope chunks required: {qid}")
    prompt_chunks = [{key: chunk[key] for key in ("chunk_id", "paper_id", "section", "text")}
                     for chunk in chunks]
    messages = [{"role": "system", "content": answer_prompt(variant)},
                {"role": "user", "content": json.dumps({"paper_id": pid, "question": question,
                                                         "chunks": prompt_chunks}, ensure_ascii=False)}]
    return question, chunks, messages


def response_record(cohort: str, variant: str, qid: str) -> dict:
    if cohort == "holdout60" and variant == "baseline5":
        return read(RUNS / "prompt_v3_holdout60/qwen" / f"{qid}.json")
    return read(OUT / cohort / variant / f"{qid}.json")


def run(cohort: str, variant: str, limit: int | None = None) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    if cohort == "holdout60" and variant == "baseline5":
        raise ValueError("Holdout60 baseline response already exists; use score")
    key = Path("F:/qwen/local_api_key.txt").read_text(encoding="utf-8").strip()
    source = read(base.SOURCE)
    selected = cases(cohort)
    if limit is not None:
        selected = selected[:limit]
    statuses = Counter()
    for index, case in enumerate(selected, 1):
        qid = case["question_id"]
        path = OUT / cohort / variant / f"{qid}.json"
        if path.exists():
            statuses[read(path)["status"]] += 1
            continue
        question, chunks, messages = input_for(case, cohort, variant, source)
        payload = {"model": "qwen3-4b-local", "messages": messages,
                   "temperature": 0, "max_tokens": 512}
        raw_request = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        record = {"question_id": qid, "paper_id": case["paper_id"],
                  "cohort": cohort, "variant": variant, "status": "started",
                  "requested_at": datetime.now(timezone.utc).isoformat(),
                  "prompt_sha256": hashlib.sha256(answer_prompt(variant).encode()).hexdigest(),
                  "request_sha256": hashlib.sha256(raw_request).hexdigest(),
                  "chunk_ids": [chunk["chunk_id"] for chunk in chunks]}
        write(path, record)
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                "http://127.0.0.1:18510/v1/chat/completions", data=raw_request,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
                reply = json.load(response)
            record["raw_output"] = reply["choices"][0]["message"].get("content") or ""
            record["usage"] = reply.get("usage")
            record["finish_reason"] = reply["choices"][0].get("finish_reason")
            try:
                record["parsed"] = base.parse_model_json(
                    record["raw_output"], set(record["chunk_ids"]), question)
                record["status"] = "completed" if record["finish_reason"] == "stop" else "truncated"
            except ValueError as error:
                record["status"] = "invalid_format"
                record["parse_error"] = str(error)
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)[:300]
        record["seconds"] = time.perf_counter() - started
        write(path, record)
        statuses[record["status"]] += 1
        if index % 10 == 0 or record["status"] != "completed":
            print(f"{cohort} {variant} answer {index}/{len(selected)} {record['status']} {record['seconds']:.1f}s", flush=True)
    return {"cohort": cohort, "variant": variant, "questions": len(selected),
            "statuses": dict(statuses)}


def score(cohort: str, variant: str) -> dict:
    source = read(base.SOURCE)
    evaluator = base.official()
    gold = evaluator.get_answers_and_evidence(source, False)
    rows, predictions = [], {}
    for case in cases(cohort):
        qid, pid = case["question_id"], case["paper_id"]
        question, chunks, _ = input_for(case, cohort, variant, source)
        record = response_record(cohort, variant, qid)
        parsed = None
        if record["status"] == "completed":
            try:
                parsed = base.parse_model_json(record["raw_output"],
                                               {c["chunk_id"] for c in chunks}, question)
            except ValueError:
                pass
        answer = parsed["short_answer"] if parsed else ""
        by_id = {c["chunk_id"]: c for c in chunks}
        source_texts = [by_id[cid]["text"] for cid in parsed["sources"]] if parsed else []
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        evidence = base.project(source_texts, paragraphs)
        predictions[qid] = {"answer": answer, "evidence": evidence}
        types = {ref["type"] for ref in gold[qid]}
        rows.append({"question_id": qid, "paper_id": pid,
                     "type": next(iter(types)) if len(types) == 1 else "mixed",
                     "answer": answer, "status": record["status"],
                     "sources": parsed["sources"] if parsed else [],
                     "answer_f1": max(evaluator.token_f1_score(answer, ref["answer"])
                                      for ref in gold[qid]),
                     "input_tokens": (record.get("usage") or {}).get("prompt_tokens", 0),
                     "answer_seconds": record.get("seconds", 0)})
    official = evaluator.evaluate({c["question_id"]: gold[c["question_id"]]
                                   for c in cases(cohort)}, predictions)
    none = [r for r in rows if r["type"] == "none"]
    answerable = [r for r in rows if r["type"] in {"extractive", "abstractive", "boolean"}]
    summary = {"cohort": cohort, "variant": variant, "questions": len(rows),
               "papers": len({r["paper_id"] for r in rows}),
               "statuses": dict(Counter(r["status"] for r in rows)),
               "official_scores": official,
               "none_nonstandard_nonempty": sum(bool(r["answer"]) and r["answer"] != "Unanswerable"
                                                for r in none),
               "none_denominator": len(none),
               "answerable_abstentions": sum(r["answer"] == "Unanswerable" for r in answerable),
               "answerable_denominator": len(answerable),
               "mean_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
               "mean_answer_seconds": statistics.mean(r["answer_seconds"] for r in rows),
               "note": "Official lexical metrics. Nonstandard nonempty none-label output is not necessarily a semantic false answer."}
    write(OUT / cohort / f"scores_{variant}.json", summary)
    write(OUT / cohort / f"cases_{variant}.json", rows)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "score"])
    parser.add_argument("--cohort", choices=["holdout60", "new_papers"], required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    outcome = run(args.cohort, args.variant, args.limit) if args.action == "run" else score(args.cohort, args.variant)
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
