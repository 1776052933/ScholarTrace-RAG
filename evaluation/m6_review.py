"""Create and summarize a human review sheet for cached PDF answers.

Reviewers should open the cited PDF page and judge the answer and citation
independently. Empty fields are unreviewed, never silently counted as correct.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ["split", "document_id", "question_id", "variant", "question", "gold_page",
          "answer_fragments", "answer", "cited_pages", "cited_excerpts",
          "answer_correct", "citations_supported", "error_type", "reviewer", "notes"]
ERROR_TYPES = {"", "retrieval_miss", "parser_miss", "unsupported_claim", "wrong_value",
               "wrong_citation", "appropriate_abstention", "other"}
EXPECTED_COUNTS = {"dev:pymupdf": 12, "dev:docling": 12,
                   "test:pymupdf": 10, "test:docling": 10}


def make_sheet(source: Path, output: Path) -> int:
    rows = []
    for path in sorted(source.glob("[0-9][0-9][0-9]_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "completed":
            continue
        citations = item.get("citations") or []
        rows.append({"split": item["split"], "document_id": item["document_id"],
                     "question_id": item["question_id"], "variant": item["variant"],
                     "question": item["question"], "gold_page": item["gold_page"],
                     "answer_fragments": " | ".join(item["answer_fragments"]),
                     "answer": item["answer"],
                     "cited_pages": " | ".join(str(cite["page"]) for cite in citations),
                     "cited_excerpts": " | ".join(str(cite.get("excerpt") or "") for cite in citations),
                     "answer_correct": "", "citations_supported": "",
                     "error_type": "", "reviewer": "", "notes": ""})
    if not rows:
        raise ValueError(f"No cached answers under {source}")
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing review sheet: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def summarize_sheet(path: Path, *, expected_counts: dict[str, int] = EXPECTED_COUNTS) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected review sheet columns")
        rows = list(reader)
    if len({(r["document_id"], r["question_id"], r["variant"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate review rows")
    if len(rows) != sum(expected_counts.values()):
        raise ValueError("Review sheet lost or added rows")
    groups = {}
    for split in ("dev", "test"):
        for variant in ("pymupdf", "docling"):
            selected = [r for r in rows if r["split"] == split and r["variant"] == variant]
            if len(selected) != expected_counts.get(f"{split}:{variant}", 0):
                raise ValueError("Review sheet has an unexpected split/variant denominator")
            for row in selected:
                if row["answer_correct"] not in {"", "yes", "no"} or row["citations_supported"] not in {"", "yes", "no"}:
                    raise ValueError("Review judgments must be blank, yes or no")
                if row["error_type"] not in ERROR_TYPES:
                    raise ValueError(f"Unknown error_type: {row['error_type']}")
                if (row["answer_correct"] or row["citations_supported"]) and not row["reviewer"].strip():
                    raise ValueError("A reviewed row requires a reviewer name")
            groups[f"{split}:{variant}"] = {
                "rows": len(selected),
                "answer_reviewed": sum(bool(r["answer_correct"]) for r in selected),
                "answer_correct": sum(r["answer_correct"] == "yes" for r in selected),
                "citation_reviewed": sum(bool(r["citations_supported"]) for r in selected),
                "citation_supported": sum(r["citations_supported"] == "yes" for r in selected),
                "error_types": {name: sum(r["error_type"] == name for r in selected)
                                for name in sorted(ERROR_TYPES - {""})},
            }
    return {"groups": groups, "complete": all(g["answer_reviewed"] == g["rows"] and
                                                  g["citation_reviewed"] == g["rows"]
                                                  for g in groups.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evaluation/runs/m3_parser_answers_10pdf_20261017"))
    parser.add_argument("--sheet", type=Path, default=Path("evaluation/runs/m6_answer_review.csv"))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        print(f"Created {make_sheet(args.source, args.sheet)} review rows in {args.sheet}")
    print(json.dumps(summarize_sheet(args.sheet), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
