"""Freeze a paper-disjoint QASPER dev cohort before any new-paper evaluation.

Selection uses only paper IDs and question IDs. Gold answers and evidence are
not inspected. Once written, the manifest is immutable: changed inputs fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/benchmarks/qasper/qasper-dev-v0.3.json"
BASELINE = PROJECT / "evaluation/runs/qasper_qwen_dev200_20261021/manifest.json"
MANIFEST = PROJECT / "evaluation/runs/qasper_new_papers_20261022/frozen_manifest.json"
TRACKED = PROJECT / "evaluation/fixtures/qasper_new_papers_20261022.json"
SEED = 20261022
PAPERS = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    old = json.loads(BASELINE.read_text(encoding="utf-8"))
    old_questions = set(old["case_ids"])
    indexed_papers = {
        paper_id for paper_id, paper in source.items()
        if any(q["question_id"] in old_questions for q in paper["qas"])
    }
    choices = sorted(set(source) - indexed_papers)
    if len(indexed_papers) != 151 or len(choices) < PAPERS:
        raise ValueError("Previous QASPER corpus or indexed-paper scope changed")
    selected = sorted(random.Random(SEED).sample(choices, PAPERS))
    cases = [{"question_id": q["question_id"], "paper_id": paper_id}
             for paper_id in selected for q in source[paper_id]["qas"]]
    if len({c["question_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate question IDs")
    if {c["question_id"] for c in cases} & old_questions:
        raise ValueError("Question overlap with baseline")
    payload = {
        "protocol": "QASPER dev; 30 uniformly sampled previously unindexed paper IDs; all questions",
        "seed": SEED,
        "source_sha256": digest(SOURCE),
        "baseline_manifest_sha256": digest(BASELINE),
        "previously_indexed_papers_sha256": hashlib.sha256(
            json.dumps(sorted(indexed_papers)).encode()).hexdigest(),
        "paper_ids": selected,
        "cases": cases,
        "primary_metrics": ["Answer F1", "text evidence hit@5"],
        "secondary_metrics": ["Evidence F1", "none-label specific answers",
                              "answerable abstentions", "format valid", "mean answer seconds",
                              "mean input tokens"],
        "interpretation": "Official lexical metrics; text evidence hit excludes questions without indexable annotated body text. No independent factual-accuracy claim.",
    }
    for path in (TRACKED, MANIFEST):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError(f"Frozen paper holdout differs; do not overwrite {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return {"paper_count": len(selected), "question_count": len(cases),
            "manifest_sha256": digest(MANIFEST), "source_sha256": payload["source_sha256"],
            "overlap_with_old_papers": len(set(selected) & indexed_papers)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
