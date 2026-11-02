"""Paired summary of saved QASPER phase-3/4 predictions and costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

from .qasper_phase3_selection import PROJECT, RUNS, read, write


ANSWERS = RUNS / "phase34_answers"
NEW = PROJECT / "evaluation/runs/qasper_new_papers_20261022"


def compare(cohort: str, candidate: str = "model_selected5") -> dict:
    root = ANSWERS / cohort
    baseline = read(root / "scores_baseline5.json")
    candidate_score = read(root / f"scores_{candidate}.json")
    old = {row["question_id"]: row for row in read(root / "cases_baseline5.json")}
    new = {row["question_id"]: row for row in read(root / f"cases_{candidate}.json")}
    if set(old) != set(new):
        raise ValueError("Paired cohorts differ")
    ids = list(old)
    old_f1 = [old[qid]["answer_f1"] for qid in ids]
    new_f1 = [new[qid]["answer_f1"] for qid in ids]
    by_paper = {}
    for qid in ids:
        by_paper.setdefault(old[qid]["paper_id"], []).append(
            new[qid]["answer_f1"] - old[qid]["answer_f1"])
    paper_ids = sorted(by_paper)
    rng = random.Random(20261022)
    draws = []
    for _ in range(5000):
        sampled = [rng.choice(paper_ids) for _ in paper_ids]
        differences = [delta for pid in sampled for delta in by_paper[pid]]
        draws.append(statistics.mean(differences))
    draws.sort()
    selector = []
    selector_input_tokens = []
    selector_output_tokens = []
    statuses = {}
    if candidate == "model_selected5":
        for qid in ids:
            record = read(RUNS / "phase3_model_selection" / cohort / f"{qid}.json")
            selector.append(record.get("seconds", 0))
            selector_input_tokens.append((record.get("usage") or {}).get("prompt_tokens", 0))
            selector_output_tokens.append((record.get("usage") or {}).get("completion_tokens", 0))
            statuses[record["status"]] = statuses.get(record["status"], 0) + 1
    retrieval = []
    retrieval_root = (NEW / "retrieval" if cohort == "new_papers" else
                      RUNS / "evidence_budget/holdout60/retrieval")
    for qid in ids:
        retrieval.append(read(retrieval_root / f"{qid}.json").get("seconds", 0))
    result = {
        "cohort": cohort, "candidate": candidate, "questions": len(ids),
        "paired_answer_f1_improved": sum(n > o for o, n in zip(old_f1, new_f1)),
        "paired_answer_f1_worse": sum(n < o for o, n in zip(old_f1, new_f1)),
        "paired_answer_f1_tied": sum(n == o for o, n in zip(old_f1, new_f1)),
        "paired_answer_f1_delta_95pct_paper_bootstrap": [draws[125], draws[4874]],
        "bootstrap_method": "5000 paper-cluster resamples, seed 20261022; exploratory uncertainty interval",
        "baseline_answer_f1": baseline["official_scores"]["Answer F1"],
        "candidate_answer_f1": candidate_score["official_scores"]["Answer F1"],
        "baseline_evidence_f1": baseline["official_scores"]["Evidence F1"],
        "candidate_evidence_f1": candidate_score["official_scores"]["Evidence F1"],
        "baseline_none_nonstandard_nonempty": baseline["none_nonstandard_nonempty"],
        "candidate_none_nonstandard_nonempty": candidate_score["none_nonstandard_nonempty"],
        "none_denominator": baseline["none_denominator"],
        "baseline_answerable_abstentions": baseline["answerable_abstentions"],
        "candidate_answerable_abstentions": candidate_score["answerable_abstentions"],
        "answerable_denominator": baseline["answerable_denominator"],
        "baseline_format_valid": baseline["statuses"].get("completed", 0),
        "candidate_format_valid": candidate_score["statuses"].get("completed", 0),
        "mean_retrieval_seconds": statistics.mean(retrieval),
        "mean_selector_seconds": statistics.mean(selector) if selector else 0,
        "mean_selector_input_tokens": statistics.mean(selector_input_tokens) if selector else 0,
        "mean_selector_output_tokens": statistics.mean(selector_output_tokens) if selector else 0,
        "mean_baseline_answer_input_tokens": baseline["mean_input_tokens"],
        "mean_candidate_answer_input_tokens": candidate_score["mean_input_tokens"],
        "mean_baseline_answer_seconds": baseline["mean_answer_seconds"],
        "mean_candidate_answer_seconds": candidate_score["mean_answer_seconds"],
        "selector_statuses": statuses,
    }
    result["mean_baseline_total_seconds"] = result["mean_retrieval_seconds"] + result["mean_baseline_answer_seconds"]
    result["mean_candidate_total_seconds"] = (result["mean_retrieval_seconds"] +
                                                result["mean_selector_seconds"] +
                                                result["mean_candidate_answer_seconds"])
    write(root / f"comparison_{candidate}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["holdout60", "new_papers"], required=True)
    parser.add_argument("--candidate", choices=["model_selected5", "related_guard5", "numeric_guard5"],
                        default="model_selected5")
    args = parser.parse_args()
    print(json.dumps(compare(args.cohort, args.candidate), ensure_ascii=False, indent=2))
