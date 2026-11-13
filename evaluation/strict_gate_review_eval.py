"""Paired, resumable audit of the production support gate on reviewed PDF answers.

The 44 cached answers stay fixed. This isolates the *gate* from retrieval and
generation changes; it is not an end-to-end benchmark of the current product.
Only public-paper questions, answers and cited excerpts are sent to the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from urllib import error, request

from rag_reader.prompts import RAG_SUPPORT_VERIFIER_PROMPT
from rag_reader.support_gate import strict_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWERS = ROOT / "evaluation/runs/m3_parser_answers_10pdf_20261017"
DEFAULT_REVIEWS = ROOT / "evaluation/runs/m6_answer_review_20261018.csv"
DEFAULT_OUTPUT = ROOT / "evaluation/runs/strict_gate_review_20261104"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 44:
        raise ValueError(f"Expected 44 review rows, got {len(rows)}")
    if any(row["answer_correct"] not in {"yes", "no"} or
           row["citations_supported"] not in {"yes", "no"} for row in rows):
        raise ValueError("All 44 review judgments must be complete")
    keys = [(row["document_id"], row["question_id"], row["variant"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate review key")
    return dict(zip(keys, rows))


def call_verifier(*, model: str, base_url: str, key: str, payload: dict) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": RAG_SUPPORT_VERIFIER_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }, ensure_ascii=False).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    req = request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    })
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=120) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        # Do not save provider response bodies: they can contain request data.
        raise RuntimeError(f"Verifier HTTP status {exc.code}") from None
    content = result["choices"][0]["message"]["content"]
    return {"raw": content, "usage": result.get("usage") or {},
            "seconds": round(time.perf_counter() - started, 4)}


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def audit_case(source: Path, review: dict[str, str], *, model: str,
               base_url: str, key: str, allow_call: bool) -> dict:
    case = json.loads(source.read_text(encoding="utf-8"))
    if case.get("status") != "completed":
        raise ValueError(f"Case is incomplete: {source.name}")
    answer = case["answer"]
    sources = case.get("sources") or []
    chunks = [{"chunk_id": cite["chunk_id"], "paper_id": cite["paper_id"],
               "text": cite["excerpt"]} for cite in case.get("citations") or []]
    before = strict_gate(answer, sources, chunks, paper_ids=[case["document_id"]])
    result = {
        "case_id": source.stem,
        "source_sha256": digest(source),
        "split": case["split"], "variant": case["variant"],
        "document_id": case["document_id"], "question_id": case["question_id"],
        "human_answer_correct": review["answer_correct"],
        "human_citations_supported": review["citations_supported"],
        "baseline_answer": answer, "baseline_sources": sources,
        "baseline_generation_seconds": case.get("seconds"),
        "baseline_generation_usage": case.get("token_usage") or {},
        "precheck": before,
        "verifier_model": model, "verifier_base_url": base_url,
    }
    if before["action"] != "verify":
        result["final"] = before
        result["status"] = "completed_no_call"
        return result
    if not allow_call:
        result["status"] = "pending_call"
        return result
    payload = {"question": case["question"], "retrieval_question": case["question"],
               "proposed_answer": answer, "paper_ids": [case["document_id"]],
               "cited_chunks": chunks}
    try:
        call = call_verifier(model=model, base_url=base_url, key=key, payload=payload)
        result["verifier"] = call
        try:
            verdict = json.loads(call["raw"])
            result["final"] = strict_gate(answer, sources, chunks,
                                           paper_ids=[case["document_id"]], verdict=verdict)
        except (ValueError, TypeError):
            result["final"] = {"action": "abstain", "reason": "invalid_verifier_response"}
        result["status"] = "completed_called"
    except (OSError, RuntimeError, KeyError, IndexError, TypeError) as exc:
        result["status"] = "call_failed"
        result["final"] = {"action": "abstain", "reason": "verifier_unavailable"}
        result["call_error_type"] = type(exc).__name__
    return result


def summarize(records: list[dict]) -> dict:
    done = [item for item in records if item.get("status", "").startswith("completed")]
    accepted = [item for item in done if item["final"]["action"] == "answer"]
    baseline_wrong = [item for item in done if item["human_answer_correct"] == "no"]
    calls = [item for item in done if item["status"] == "completed_called"]
    def usage(name: str) -> int:
        return sum(int(item["verifier"]["usage"].get(name) or 0) for item in calls)
    return {
        "total_expected": 44, "completed": len(done),
        "baseline_correct": sum(item["human_answer_correct"] == "yes" for item in done),
        "baseline_wrong_or_refused": len(baseline_wrong),
        "baseline_citations_supported": sum(item["human_citations_supported"] == "yes" for item in done),
        "accepted": len(accepted),
        "accepted_human_correct": sum(item["human_answer_correct"] == "yes" for item in accepted),
        "accepted_human_wrong": sum(item["human_answer_correct"] == "no" for item in accepted),
        "accepted_human_supported": sum(item["human_citations_supported"] == "yes" for item in accepted),
        "refused_human_correct": sum(item["human_answer_correct"] == "yes" and
                                      item["final"]["action"] != "answer" for item in done),
        "verifier_calls": len(calls),
        "verifier_prompt_tokens": usage("prompt_tokens"),
        "verifier_completion_tokens": usage("completion_tokens"),
        "verifier_seconds_total": round(sum(item["verifier"]["seconds"] for item in calls), 4),
        "call_failures": sum(item.get("status") == "call_failed" for item in records),
        "reasons": {reason: sum(item["final"]["reason"] == reason for item in done)
                    for reason in sorted({item["final"]["reason"] for item in done})},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=44)
    parser.add_argument("--max-calls", type=int, default=44)
    args = parser.parse_args()
    if not 1 <= args.limit <= 44 or not 0 <= args.max_calls <= 44:
        parser.error("limit must be 1..44 and max-calls 0..44")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    key = os.getenv("DEEPSEEK_API_KEY", "")
    reviews = reviewed_rows(args.reviews)
    paths = sorted(args.answers.glob("[0-9][0-9][0-9]_*.json"))
    if len(paths) != 44:
        raise ValueError(f"Expected 44 cached answers, got {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = {"protocol": "fixed-cached-answers-gate-only-v1",
                "answers_manifest_sha256": digest(args.answers / "manifest.json"),
                "review_sha256": digest(args.reviews),
                "gate_sha256": digest(ROOT / "rag_reader/support_gate.py"),
                "prompt_sha256": digest(ROOT / "rag_reader/prompts.py"),
                "model": model, "base_url": base_url, "expected_cases": 44,
                "scope": "gate-only; cached DeepSeek answers and old cited excerpts; not current end-to-end RAG"}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Evaluation inputs changed; choose a new output directory")
    else:
        atomic_json(manifest_path, manifest)
    used_calls = sum(json.loads(path.read_text(encoding="utf-8")).get("status") in
                     {"completed_called", "call_failed"}
                     for path in args.output.glob("[0-9][0-9][0-9]_*.json"))
    for source in paths[:args.limit]:
        output = args.output / source.name
        if output.exists():
            continue
        case = json.loads(source.read_text(encoding="utf-8"))
        review = reviews[(case["document_id"], case["question_id"], case["variant"])]
        allow_call = used_calls < args.max_calls and bool(key)
        record = audit_case(source, review, model=model, base_url=base_url,
                            key=key, allow_call=allow_call)
        if record["status"] in {"completed_called", "call_failed"}:
            used_calls += 1
        atomic_json(output, record)
        print(f'{source.stem}: {record["status"]} / '
              f'{record.get("final", {}).get("reason", "pending")}', flush=True)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in
               sorted(args.output.glob("[0-9][0-9][0-9]_*.json"))]
    report = summarize(records)
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
