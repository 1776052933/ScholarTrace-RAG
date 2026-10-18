"""Sort cached answers for human review using transparent risk signals.

Flags identify records that deserve attention; they are not correctness labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import unicodedata


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prioritize(source: Path) -> list[dict]:
    records = []
    for path in sorted(source.glob("[0-9][0-9][0-9]_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "completed":
            continue
        reasons = []
        if not item.get("answer_contains_gold_fragments"):
            reasons.append("answer_missing_annotated_fragment")
        if not item.get("cited_gold_page"):
            reasons.append("gold_page_not_cited")
        cited_gold_text = _norm(" ".join(
            str(cite.get("excerpt") or "") for cite in item.get("citations") or []
            if cite.get("page") == item["gold_page"]))
        if not all(_norm(fragment) in cited_gold_text
                   for fragment in item["answer_fragments"]):
            reasons.append("annotated_fragment_not_contiguous_in_cited_page")
        answer = _norm(item.get("answer") or "")
        if any(marker in answer for marker in ("无法确定", "无法回答", "没有足够", "cannot determine",
                                               "insufficient evidence", "not enough information")):
            reasons.append("possible_abstention")
        if not item.get("citations"):
            reasons.append("no_citations")
        records.append({"split": item["split"], "document_id": item["document_id"],
                        "question_id": item["question_id"], "variant": item["variant"],
                        "reasons": reasons, "priority": 1 if reasons else 2,
                        "answer_preview": str(item.get("answer") or "")[:150]})
    if len(records) != 44:
        raise ValueError(f"Expected all 44 cached PDF answers, found {len(records)}")
    return sorted(records, key=lambda row: (row["priority"],
                                            row["split"] != "test", row["document_id"],
                                            row["question_id"], row["variant"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("evaluation/runs/m3_parser_answers_10pdf_20261017"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs/m6_review_priorities_20261018.json"))
    args = parser.parse_args()
    records = prioritize(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"note": "Flags are not correctness judgments.",
                                       "records": records}, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"flagged": sum(bool(row["reasons"]) for row in records),
                      "total": len(records), "first": records[:8]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
