"""Lock a question-level QASPER dev holdout before final configuration choice.

Uses new questions from the 151 already indexed papers. It is not a new-paper
generalization test. Gold type is used for stratification only and is never
sent to retrieval or generation.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random

from . import qasper_followup as base

SEED = 20261023
COUNTS = {"extractive": 20, "abstractive": 16, "boolean": 12, "none": 12}
PATH = base.NEW / "holdout_60.json"


def freeze():
    manifest, old_ids, _, _, source, questions = base.original()
    indexed_papers = {questions[qid][0] for qid in old_ids}
    gold = base.official().get_answers_and_evidence(source, False)
    groups = defaultdict(list)
    for qid, (pid, _, question) in questions.items():
        if qid in old_ids or pid not in indexed_papers:
            continue
        answer_types = {row["type"] for row in gold[qid]}
        if len(answer_types) == 1:
            groups[next(iter(answer_types))].append(qid)
    rng = random.Random(SEED)
    cases = []
    for kind, count in COUNTS.items():
        available = sorted(groups[kind])
        if len(available) < count:
            raise ValueError(f"Only {len(available)} available {kind} questions")
        rng.shuffle(available)
        for qid in available[:count]:
            pid, _, question = questions[qid]
            cases.append({"question_id": qid, "paper_id": pid,
                          "question": question["question"], "stratum": kind})
    payload = {"seed": SEED, "counts": COUNTS, "source_sha256": base.sha(base.SOURCE),
               "baseline_manifest_sha256": base.sha(base.OLD / "manifest.json"),
               "collection": manifest["settings"]["collection_name"],
               "sample_scope": "new dev questions, previously indexed paper IDs",
               "cases": cases}
    if PATH.exists():
        if base.read(PATH) != payload:
            raise ValueError("Holdout definition changed")
    else:
        base.write(PATH, payload)
    return {"questions": len(cases), "paper_count": len({c["paper_id"] for c in cases}),
            "counts": COUNTS, "holdout_sha256": base.sha(PATH),
            "overlap_with_original_200": len({c["question_id"] for c in cases} & set(old_ids))}


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
