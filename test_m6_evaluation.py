"""M6 offline contract tests for frozen cases and review denominators."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation.m6_metrics import load_multi_cases, score_evidence, summarize_variants
from evaluation.m6_review import FIELDS, summarize_sheet


def test_frozen_cross_paper_cases_are_split_and_traceable() -> None:
    cases, hashes = load_multi_cases()
    assert len(cases) == 4
    assert {case["split"] for case in cases} == {"dev", "test"}
    assert len(hashes) == 4
    for case in cases:
        assert len({ref["document_id"] for ref in case["references"]}) >= 2
        assert all(ref["page"] > 0 and ref["answer_fragments"] for ref in case["references"])


def test_new_dev_and_holdout_references_do_not_reuse_original_test_facts() -> None:
    original, _ = load_multi_cases()
    dev, _ = load_multi_cases(cases_name="m6_dev_pairs.json", require_both_splits=False)
    holdout, _ = load_multi_cases(cases_name="m6_holdout_pairs.json", require_both_splits=False)
    assert len(dev) == len(holdout) == 4
    assert {case["split"] for case in dev} == {"dev"}
    assert {case["split"] for case in holdout} == {"test"}
    used = {(ref["document_id"], ref["source_question_id"])
            for case in original if case["split"] == "test" for ref in case["references"]}
    fresh = {(ref["document_id"], ref["source_question_id"])
             for case in holdout for ref in case["references"]}
    assert used.isdisjoint(fresh)


def test_cross_paper_evidence_requires_both_correct_pages_and_values() -> None:
    refs = [{"document_id": "p1", "page": 2, "answer_fragments": ["0.85"]},
            {"document_id": "p2", "page": 4, "answer_fragments": ["41.37"]}]
    chunks = [{"paper_id": "p1", "page": 2, "text": "precision 0.85"},
              {"paper_id": "p2", "page": 3, "text": "41.37"}]
    partial = score_evidence(chunks, refs)
    assert partial["paper_coverage"] == 1
    assert partial["page_recall"] == 0.5
    assert partial["answer_fragment_recall"] == 0.5
    assert not partial["all_gold_fragments_found"]
    chunks.append({"paper_id": "p2", "page": 4, "text": "average 41.37"})
    assert score_evidence(chunks, refs)["all_gold_fragments_found"]


def test_failed_variants_stay_in_denominator() -> None:
    records = [{"variants": {"plain": {"status": "failed"}}},
               {"variants": {"plain": {"status": "completed", "seconds": 2,
                                        "metrics": {"paper_coverage": 1, "page_recall": 1,
                                                    "answer_fragment_recall": 1,
                                                    "all_gold_fragments_found": True,
                                                    "chunk_count": 5}}}}]
    summary = summarize_variants(records, ["plain"])["plain"]
    assert summary["cases"] == 2 and summary["failed"] == 1
    assert summary["all_gold_fragments_found"] == 1


def test_review_sheet_does_not_count_unreviewed_as_correct(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    base = dict.fromkeys(FIELDS, "")
    base.update(split="test", document_id="p1", question_id="q1", variant="pymupdf")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerow(base)
    result = summarize_sheet(path, expected_counts={"test:pymupdf": 1})
    assert not result["complete"]
    assert result["groups"]["test:pymupdf"]["answer_correct"] == 0
    base.update(answer_correct="yes", citations_supported="yes")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerow(base)
    with pytest.raises(ValueError, match="reviewer"):
        summarize_sheet(path, expected_counts={"test:pymupdf": 1})
