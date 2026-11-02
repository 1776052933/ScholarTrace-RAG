"""Local diagnostics for controlled QASPER results; no model or judge calls."""
from __future__ import annotations

from collections import Counter
import json
import re
import statistics

from . import qasper_followup as base


def report(cohort="200"):
    folder = base.NEW
    v1 = base.read(folder / f"prompt_v1_{cohort}" / "cases.json")
    v3 = base.read(folder / f"prompt_v3_{cohort}" / "cases.json")
    assert len(v1) == len(v3)
    old = {r["question_id"]: r for r in base.read(folder / "baseline_cases.json")}
    source = base.read(base.SOURCE)
    gold = base.official().get_answers_and_evidence(source, False)
    id_order = [r["question_id"] for r in v1]
    assert id_order == [r["question_id"] for r in v3]
    details = []
    for a,b in zip(v1,v3):
        qid = a["question_id"]
        types = {ref["type"] for ref in gold[qid]}
        answers = {ref["answer"] for ref in gold[qid]}
        unanimous = next(iter(types)) if len(types) == 1 else "mixed"
        binary = bool(re.match(r"^\s*(Do|Does|Did|Is|Are|Was|Were|Can|Could|Will|Would|Has|Have|Had|Should)\b",
                               a["question"], re.I))
        details.append({"question_id": qid, "type": unanimous, "question": a["question"],
                        "retrieval": old.get(qid, {}).get("review_priority"),
                        "v1_answer": a["short_answer"], "v3_answer": b["short_answer"],
                        "v1_f1": a["answer_f1"], "v3_f1": b["answer_f1"],
                        "delta": b["answer_f1"] - a["answer_f1"],
                        "v1_status": a["status"], "v3_status": b["status"],
                        "binary_question": binary,
                        "unanimous_unanswerable": answers == {"Unanswerable"},
                        "unanimous_answerable": "Unanswerable" not in answers})
    groups = {}
    for kind in ("extractive", "abstractive", "boolean", "none", "mixed"):
        rows = [r for r in details if r["type"] == kind]
        groups[kind] = {"count": len(rows),
                        "v1_mean_f1": statistics.mean(r["v1_f1"] for r in rows) if rows else None,
                        "v3_mean_f1": statistics.mean(r["v3_f1"] for r in rows) if rows else None}
    answerable = [r for r in details if r["unanimous_answerable"]]
    none = [r for r in details if r["unanimous_unanswerable"]]
    def refusal_outcomes(version):
        answers = [r[f"{version}_answer"] for r in none]
        return {
            "wrong_concrete_answer": sum(bool(a.strip()) and a != "Unanswerable" for a in answers),
            "invalid_or_empty": sum(not a.strip() for a in answers),
            "unanswerable": sum(a == "Unanswerable" for a in answers),
        }
    result = {"cohort": cohort, "count": len(details),
              "paired": {"v3_higher": sum(r["delta"] > 1e-9 for r in details),
                         "v1_higher": sum(r["delta"] < -1e-9 for r in details),
                         "tie": sum(abs(r["delta"]) <= 1e-9 for r in details)},
              "unanimous_gold_type": groups,
              "answerable_overabstain": {
                  "denominator": len(answerable),
                  "v1": sum(r["v1_answer"] == "Unanswerable" for r in answerable),
                  "v3": sum(r["v3_answer"] == "Unanswerable" for r in answerable)},
              "unanswerable_false_answer": {
                  "denominator": len(none),
                  "v1": refusal_outcomes("v1"),
                  "v3": refusal_outcomes("v3")},
              "format_status": {"v1": dict(Counter(r["v1_status"] for r in details)),
                                "v3": dict(Counter(r["v3_status"] for r in details))},
              "note": "Counts based on unanimous gold annotations where specified; F1 is lexical, not manual correctness."}
    base.write(folder / f"prompt_comparison_{cohort}.json", result)
    base.write(folder / f"prompt_comparison_cases_{cohort}.json", details)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=["200", "holdout60"], default="200")
    args = p.parse_args()
    print(json.dumps(report(args.cohort), ensure_ascii=False, indent=2))
