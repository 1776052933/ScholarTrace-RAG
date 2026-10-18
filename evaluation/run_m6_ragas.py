"""Replay cached M3 answers with Ragas' offline StringPresence metric.

This intentionally makes no LLM calls. Fragment presence is a narrow proxy,
not an answer-correctness or faithfulness judgment. Run on both frozen splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .m6_metrics import digest


DEFAULT_SOURCE = Path("evaluation/runs/m3_parser_answers_10pdf_20261017")
DEFAULT_OUTPUT = Path("evaluation/runs/m6_ragas_cached_answers_20261018.json")


def evaluate(source: Path) -> dict:
    try:
        from ragas.metrics.collections import StringPresence
    except ImportError as exc:
        raise RuntimeError("Install evaluation/requirements-ragas.txt to run this optional check") from exc
    scorer = StringPresence()
    rows = []
    for path in sorted(source.glob("[0-9][0-9][0-9]_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "completed":
            continue
        fragments = item["answer_fragments"]
        if not fragments:
            raise ValueError(f"No gold fragments in {path}")
        scores = [float(scorer.score(reference=fragment, response=item["answer"]).value)
                  for fragment in fragments]
        rows.append({"document_id": item["document_id"], "question_id": item["question_id"],
                     "split": item["split"], "variant": item["variant"],
                     "fragments": fragments, "fragment_scores": scores,
                     "all_fragments_present": all(value == 1.0 for value in scores),
                     "answer_source_sha256": digest(path)})
    if not rows:
        raise ValueError(f"No completed cached answers under {source}")
    summary = {}
    for split in ("dev", "test"):
        for variant in ("pymupdf", "docling"):
            group = [row for row in rows if row["split"] == split and row["variant"] == variant]
            if group:
                summary[f"{split}:{variant}"] = {
                    "cases": len(group),
                    "all_fragments_present": sum(row["all_fragments_present"] for row in group),
                    "mean_fragment_presence": mean(score for row in group
                                                   for score in row["fragment_scores"]),
                }
    return {"metric": "Ragas StringPresence (offline, case-sensitive)",
            "source": str(source), "source_manifest_sha256": digest(source / "manifest.json"),
            "summary": summary, "records": rows,
            "limitation": "A matching string can occur in an incorrect or unsupported answer; this is not faithfulness."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
