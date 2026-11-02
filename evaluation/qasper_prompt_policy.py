"""Predeclared question-form routing: v1 for yes/no questions, v3 otherwise.

The routing reads only question text, never gold or model answers. It is scored
on the frozen 200 and the locked 60-question holdout from existing model calls.
"""
from __future__ import annotations

from collections import Counter
import json
import re
import statistics

from . import qasper_followup as base
from .qasper_prompt_ablation import directory

BINARY_START = re.compile(
    r"^\s*(Do|Does|Did|Is|Are|Was|Were|Can|Could|Will|Would|Has|Have|Had|Should)\b", re.I)


def version(question: str) -> str:
    return "v1" if BINARY_START.match(question) else "v3"


def score(cohort: str):
    a = base.read(directory(cohort, "v1") / "inputs.json")
    b = base.read(directory(cohort, "v3") / "inputs.json")
    if [c["question_id"] for c in a["cases"]] != [c["question_id"] for c in b["cases"]]:
        raise ValueError("Prompt cohorts differ")
    source = base.read(base.SOURCE)
    gold = base.official().get_answers_and_evidence(source, False)
    evaluator = base.official()
    predictions, rows = {}, []
    for case in a["cases"]:
        qid = case["question_id"]
        chosen = version(case["question"])
        record = base.read(directory(cohort, chosen) / "qwen" / f"{qid}.json")
        status = record.get("status", "missing")
        parsed = record.get("parsed") if status == "completed" else None
        if parsed:
            try:
                parsed = base.parse_model_json(record["raw_output"],
                    {c["chunk_id"] for c in case["chunks"]}, case["question"])
            except ValueError:
                parsed, status = None, "invalid_format_audit"
        answer = parsed["short_answer"] if parsed else ""
        paragraphs = [p for section in source[case["paper_id"]]["full_text"]
                      for p in section["paragraphs"]]
        chunks = {c["chunk_id"]: c["text"] for c in case["chunks"]}
        evidence = base.project([chunks[s] for s in parsed["sources"]], paragraphs) if parsed else []
        predictions[qid] = {"answer": answer, "evidence": evidence}
        rows.append({"question_id": qid, "prompt": chosen, "status": status,
                     "question": case["question"], "short_answer": answer,
                     "answer_f1": max(evaluator.token_f1_score(answer, ref["answer"]) for ref in gold[qid]),
                     "gold_answers": list(dict.fromkeys(ref["answer"] for ref in gold[qid]))})
    scores = evaluator.evaluate({c["question_id"]: gold[c["question_id"]]
                                 for c in a["cases"]}, predictions)
    report = {"cohort": cohort, "questions": len(rows), "policy": "v1 if leading auxiliary verb, else v3",
              "routing_counts": dict(Counter(r["prompt"] for r in rows)),
              "statuses": dict(Counter(r["status"] for r in rows)),
              "official_scores": scores,
              "note": "Policy selected before holdout answers were scored; QASPER lexical F1 is not independent factual correctness."}
    base.write(base.NEW / f"prompt_policy_{cohort}.json", report)
    base.write(base.NEW / f"prompt_policy_cases_{cohort}.json", rows)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=["200", "holdout60"], required=True)
    args = p.parse_args()
    print(json.dumps(score(args.cohort), ensure_ascii=False, indent=2))
