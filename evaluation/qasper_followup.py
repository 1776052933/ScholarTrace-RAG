"""Frozen-evidence QASPER diagnosis and bounded local/API model comparison.

No gold answers or annotated evidence enter model requests. Existing runs remain
read-only; every provider request is recorded before dispatch and never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "evaluation/runs/qasper_qwen_dev200_20261021"
NEW = ROOT / "evaluation/runs/qasper_followup_20261022"
SOURCE = ROOT / "data/benchmarks/qasper/qasper-dev-v0.3.json"
EVALUATOR = ROOT / "data/benchmarks/qasper/qasper_evaluator.py"
PROMPT = """You answer questions about one research paper using only supplied excerpts. Excerpts may describe earlier work; do not attribute those findings to this paper. For numbers, compare the same metric and report the actual value, not a relative improvement. If the excerpts do not establish the answer, abstain. Return only a JSON object with these fields: short_answer (a concise English answer; exactly Yes or No for yes/no questions, exactly Unanswerable when evidence is insufficient), explanation (one or two English sentences), sources (an array of chunk_id strings that directly support the answer). Cite only supplied chunks. Do not infer missing values from placeholders such as [FORMULA]."""
SEED = 20261022
COUNTS = {"extractive": 10, "abstractive": 8, "boolean": 6, "none": 6}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def official():
    spec = importlib.util.spec_from_file_location("qasper_official_followup", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(directory):
    rows = [read(path) for path in (OLD / directory).glob("*.json")]
    return {row["case_id"]: row for row in rows}


def original():
    manifest = read(OLD / "manifest.json")
    ids = manifest["case_ids"]
    answers, retrievals = records("answer"), records("retrieve")
    assert len(ids) == len(set(ids)) == len(answers) == len(retrievals) == 200
    assert all(answers[q]["status"] == retrievals[q]["status"] == "completed" for q in ids)
    source = read(SOURCE)
    question_map = {q["question_id"]: (pid, paper, q)
                    for pid, paper in source.items() for q in paper["qas"]}
    return manifest, ids, answers, retrievals, source, question_map


def normalize(value):
    return " ".join(value.split())


def project(excerpts, paragraphs):
    selected = []
    for excerpt in excerpts:
        text = normalize(excerpt)
        selected.extend(p for p in paragraphs if text and normalize(p) and
                        (normalize(p) in text or (len(text) >= 40 and text in normalize(p))))
    return list(dict.fromkeys(selected))


def diagnostic():
    _, ids, answers, retrievals, source, question_map = original()
    evaluator = official()
    gold = evaluator.get_answers_and_evidence(source, False)
    rows = []
    for qid in ids:
        pid, paper, question = question_map[qid]
        paragraphs = [p for section in paper["full_text"] for p in section["paragraphs"]]
        paragraph_keys = {normalize(p) for p in paragraphs}
        # A table/caption-only label is outside this structured text index.
        text_references = [set(normalize(p) for p in ref["evidence"] if normalize(p) in paragraph_keys)
                           for ref in gold[qid]]
        text_references = [group for group in text_references if group]
        ranked = retrievals[qid]["retrieved"]
        mapped = [set(normalize(p) for p in project([chunk["text"]], paragraphs)) for chunk in ranked]
        def scores(group):
            ranks = [i + 1 for i, matches in enumerate(mapped) if matches & group]
            union = set().union(*mapped) if mapped else set()
            return (int(bool(ranks) and ranks[0] <= 1), int(bool(ranks)),
                    len(union & group) / len(group), 1 / ranks[0] if ranks else 0)
        metric = max((scores(group) for group in text_references), key=lambda item: (item[1],item[2],item[3]), default=None)
        answer = answers[qid]["answer"]["answer"]
        best = max(gold[qid], key=lambda ref: evaluator.token_f1_score(answer, ref["answer"]))
        row = {"question_id": qid, "paper_id": pid, "question": question["question"],
               "best_gold_type": best["type"], "gold_types": sorted({ref["type"] for ref in gold[qid]}),
               "answer_f1": evaluator.token_f1_score(answer, best["answer"]),
               "retrieval_status": "text_gold" if metric else "no_indexable_text_gold",
               "hit_at_1": metric[0] if metric else None,
               "hit_at_5": metric[1] if metric else None,
               "text_recall_at_5": metric[2] if metric else None,
               "mrr_at_5": metric[3] if metric else None,
               "answer_seconds": answers[qid]["seconds"], "retrieve_seconds": retrievals[qid]["seconds"],
               "input_tokens": answers[qid].get("usage", {}).get("prompt_tokens", 0),
               "output_tokens": answers[qid].get("usage", {}).get("completion_tokens", 0),
               "citations": len(answers[qid]["answer"].get("citations", [])),
               "has_formula_placeholder": any("[FORMULA]" in chunk["text"] for chunk in ranked),
               "review_priority": ("answer_zero_with_text_evidence" if metric and metric[1] and
                                   evaluator.token_f1_score(answer, best["answer"]) == 0 else
                                   "text_evidence_missed" if metric and not metric[1] else
                                   "source_not_indexable" if not metric else "other")}
        rows.append(row)
    valid = [row for row in rows if row["retrieval_status"] == "text_gold"]
    summary = {"baseline": str(OLD), "count": len(rows), "source_sha256": sha(SOURCE),
               "manifest_sha256": sha(OLD / "manifest.json"),
               "answer_f1": statistics.mean(row["answer_f1"] for row in rows),
               "text_gold_questions": len(valid), "no_indexable_text_gold": len(rows)-len(valid),
               "hit_at_1": statistics.mean(row["hit_at_1"] for row in valid),
               "hit_at_5": statistics.mean(row["hit_at_5"] for row in valid),
               "text_recall_at_5": statistics.mean(row["text_recall_at_5"] for row in valid),
               "mrr_at_5": statistics.mean(row["mrr_at_5"] for row in valid),
               "priority_counts": dict(Counter(row["review_priority"] for row in rows)),
               "note": "Text-only source paragraph projection, max over annotators; not semantic evidence judgment."}
    write(NEW / "baseline_diagnostics.json", summary)
    write(NEW / "baseline_cases.json", rows)
    return summary


def freeze():
    _, ids, answers, _, source, question_map = original()
    evaluator = official()
    gold = evaluator.get_answers_and_evidence(source, False)
    groups = defaultdict(list)
    for qid in ids:
        types = {ref["type"] for ref in gold[qid]}
        if len(types) == 1:
            groups[next(iter(types))].append(qid)
    rng = random.Random(SEED)
    chosen = []
    for kind, count in COUNTS.items():
        options = sorted(groups[kind])
        rng.shuffle(options)
        if kind == "extractive":
            numeric = [q for q in options if re.search(r"\b(how many|how much|percentage|percent|score|number|size|accuracy|f1)\b", question_map[q][2]["question"], re.I)]
            selected = numeric[:4] + [q for q in options if q not in numeric[:4]][:count-4]
        else:
            selected = options[:count]
        assert len(selected) == count
        chosen.extend((qid, kind) for qid in selected)
    assert len(chosen) == 30 and len({q for q, _ in chosen}) == 30
    cases = []
    for qid, kind in chosen:
        row = answers[qid]
        pid, _, question = question_map[qid]
        chunks = [{"chunk_id": c["chunk_id"], "paper_id": c["paper_id"], "section": c["section"],
                   "text": c["text"]} for c in row["retrieved"]]
        messages = [{"role": "system", "content": PROMPT},
                    {"role": "user", "content": json.dumps({"paper_id": pid, "question": question["question"],
                                                           "chunks": chunks}, ensure_ascii=False)}]
        cases.append({"question_id": qid, "paper_id": pid, "stratum": kind, "question": question["question"],
                      "chunks": chunks, "messages": messages,
                      "message_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    output = {"seed": SEED, "counts": COUNTS, "source_sha256": sha(SOURCE),
              "old_manifest_sha256": sha(OLD / "manifest.json"),
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "cases": cases,
              "rule": "Unanimous gold answer type only; type used for stratification and never appears in model messages."}
    path = NEW / "frozen_30.json"
    if path.exists():
        assert read(path) == output, "Frozen sample changed"
    else:
        write(path, output)
    return {"cases": len(cases), "counts": dict(Counter(c["stratum"] for c in cases)),
            "input_chars_max": max(len(json.dumps(c["messages"], ensure_ascii=False)) for c in cases),
            "prompt_sha256": output["prompt_sha256"]}


def parse_model_json(content, valid_ids, question):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            break
    else:
        raise ValueError("No JSON object in model output")
    short = value.get("short_answer")
    explanation = value.get("explanation")
    sources = value.get("sources")
    if not isinstance(short, str) or not short.strip() or not isinstance(explanation, str) or not isinstance(sources, list):
        raise ValueError("Invalid answer contract")
    if any(not isinstance(s, str) or s not in valid_ids for s in sources):
        raise ValueError("Unknown source id")
    if re.match(r"^\s*(Do|Does|Did|Is|Are|Was|Were|Can|Could|Will|Would|Has|Have|Had|Should)\b", question, re.I):
        if short.strip() not in {"Yes", "No", "Unanswerable"}:
            raise ValueError("Binary question requires Yes, No, or Unanswerable")
    elif short.strip() in {"Yes", "No"}:
        raise ValueError("Content question requires a factual short answer")
    if short.strip() != "Unanswerable" and re.search(
        r"\b(?:not (?:mentioned|stated|specified|provided|given)|"
        r"does not (?:mention|state|specify|provide)|cannot be determined)\b",
        short, re.I,
    ):
        raise ValueError("Abstention must use exact Unanswerable label")
    return {"short_answer": short.strip(), "explanation": explanation.strip(),
            "sources": list(dict.fromkeys(sources))}


def run(model, limit=30, retry_failed_local=False):
    from openai import OpenAI
    frozen = read(NEW / "frozen_30.json")
    if frozen["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest():
        raise ValueError("Prompt changed after freeze")
    if model == "qwen":
        os.environ["NO_PROXY"] = ",".join(filter(None, [os.getenv("NO_PROXY", ""), "127.0.0.1", "localhost"]))
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
        key = Path("F:/qwen/local_api_key.txt").read_text(encoding="utf-8").strip()
        client = OpenAI(api_key=key, base_url="http://127.0.0.1:18510/v1", timeout=180, max_retries=0)
        model_name, extra = "qwen3-4b-local", {}
    elif model == "deepseek":
        if os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") != "https://api.deepseek.com":
            raise ValueError("DeepSeek base URL is not official")
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is missing")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180, max_retries=0)
        model_name, extra = "deepseek-flash", {"extra_body": {"thinking": {"type": "disabled"}}}
    else:
        raise ValueError(model)
    folder = NEW / model
    folder.mkdir(parents=True, exist_ok=True)
    results = Counter()
    for index, case in enumerate(frozen["cases"][:limit], 1):
        path = folder / (case["question_id"] + ".json")
        if path.exists():
            prior = read(path)
            if (retry_failed_local and model == "qwen" and prior.get("status") == "failed"
                    and prior.get("error_type") == "InternalServerError"
                    and prior.get("error_detail") == "Error code: 502"):
                attempts = folder / "attempts"
                attempts.mkdir(exist_ok=True)
                path.replace(attempts / path.name)
            else:
                results[prior["status"]] += 1
                continue
        record = {"question_id": case["question_id"], "model": model_name,
                  "requested_at": datetime.now(timezone.utc).isoformat(),
                  "message_sha256": case["message_sha256"], "status": "started", "max_tokens": 512,
                  "thinking": "disabled"}
        write(path, record)  # A started call is never silently sent a second time.
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(model=model_name, messages=case["messages"],
                max_tokens=512, temperature=0, **extra)
            record["response_model"] = response.model
            record["finish_reason"] = response.choices[0].finish_reason
            record["raw_output"] = response.choices[0].message.content or ""
            record["usage"] = response.usage.model_dump() if response.usage else None
            try:
                record["parsed"] = parse_model_json(record["raw_output"], {c["chunk_id"] for c in case["chunks"]}, case["question"])
                record["status"] = "completed" if record["finish_reason"] == "stop" else "truncated"
            except ValueError as error:
                record["status"], record["parse_error"] = "invalid_format", str(error)
        except Exception as error:
            record["status"], record["error_type"] = "failed", type(error).__name__
            record["error_detail"] = str(error)[:300]
        record["seconds"] = time.perf_counter() - started
        write(path, record)
        results[record["status"]] += 1
        print(f"{model} {index}/{limit} {record['status']} {record['seconds']:.1f}s", flush=True)
    return {"model": model, "statuses": dict(results), "folder": str(folder)}


def score(model):
    frozen = read(NEW / "frozen_30.json")
    source = read(SOURCE)
    evaluator = official()
    gold = evaluator.get_answers_and_evidence(source, False)
    predictions, cases = {}, []
    for case in frozen["cases"]:
        qid = case["question_id"]
        path = NEW / model / (qid + ".json")
        row = read(path) if path.exists() else {}
        status = row.get("status", "missing")
        parsed = row.get("parsed") if status == "completed" else None
        if parsed:
            try:
                parsed = parse_model_json(row["raw_output"],
                    {c["chunk_id"] for c in case["chunks"]}, case["question"])
            except ValueError:
                parsed, status = None, "invalid_format_audit"
        paper = source[case["paper_id"]]
        paragraphs = [p for section in paper["full_text"] for p in section["paragraphs"]]
        excerpt_map = {c["chunk_id"]: c["text"] for c in case["chunks"]}
        evidence = project([excerpt_map[s] for s in parsed["sources"]], paragraphs) if parsed else []
        answer = parsed["short_answer"] if parsed else ""
        predictions[qid] = {"answer": answer, "evidence": evidence}
        best_f1 = max(evaluator.token_f1_score(answer, ref["answer"]) for ref in gold[qid])
        cases.append({"question_id": qid, "stratum": case["stratum"], "status": status,
                      "short_answer": answer, "answer_f1": best_f1,
                      "gold_answers": list(dict.fromkeys(ref["answer"] for ref in gold[qid])),
                      "sources": parsed["sources"] if parsed else [], "seconds": row.get("seconds"),
                      "input_tokens": (row.get("usage") or {}).get("prompt_tokens", 0),
                      "output_tokens": (row.get("usage") or {}).get("completion_tokens", 0)})
    official_scores = evaluator.evaluate({c["question_id"]: gold[c["question_id"]] for c in frozen["cases"]}, predictions)
    completed = [r for r in cases if r["status"] == "completed"]
    report = {"model": model, "questions": 30, "statuses": dict(Counter(c["status"] for c in cases)),
              "official_scores": official_scores, "score_denominator": "all 30 frozen questions; missing/failed are zero",
              "input_tokens": sum(c["input_tokens"] for c in cases),
              "output_tokens": sum(c["output_tokens"] for c in cases),
              "elapsed_completed_seconds": sum(c["seconds"] for c in completed),
              "by_stratum": {kind: {"count": len(group), "answer_f1": statistics.mean(c["answer_f1"] for c in group)}
                             for kind in COUNTS for group in [[c for c in cases if c["stratum"] == kind]]},
              "note": "Short-answer F1 on a new common prompt; compare Qwen vs DeepSeek, not directly with old long-answer F1."}
    write(NEW / ("score_" + model + ".json"), report)
    write(NEW / ("cases_" + model + ".json"), cases)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["diagnostic", "freeze", "run", "score"])
    parser.add_argument("--model", choices=["qwen", "deepseek"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--retry-failed-local", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error("--limit must be 1..30")
    if args.action in {"run", "score"} and not args.model:
        parser.error("--model is required")
    result = {"diagnostic": diagnostic, "freeze": freeze}.get(args.action)
    print(json.dumps(result() if result else run(args.model, args.limit, args.retry_failed_local) if args.action == "run" else score(args.model),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
