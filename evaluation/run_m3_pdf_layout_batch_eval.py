"""Evaluate the frozen ten-PDF first-pass labels against both parser paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_reader.parser import PDFParser
from .run_m3_pdf_layout_eval import evaluate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["docling_table_tp", "docling_table_fp", "docling_table_fn",
            "docling_row_checks", "pymupdf_row_checks", "row_checks_total",
            "docling_evidence_page_fragments", "pymupdf_evidence_page_fragments",
            "evidence_questions_total"]
    totals = {key: sum(report["metrics"][key] for report in reports) for key in keys}
    tp, fp, fn = (totals[key] for key in ("docling_table_tp", "docling_table_fp", "docling_table_fn"))
    totals.update({
        "documents": len(reports),
        "reviewed_pages": sum(len(report["scope"]["reviewed_pages"]) for report in reports),
        "docling_table_detection_precision": tp / (tp + fp) if tp + fp else None,
        "docling_table_detection_recall": tp / (tp + fn) if tp + fn else None,
    })
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/gold/m3_pdf_layout/manifest.json"))
    parser.add_argument("--batch-labels", type=Path, default=Path("evaluation/gold/m3_pdf_layout/batch_labels.json"))
    parser.add_argument("--attention-label", type=Path, default=Path("evaluation/gold/m3_pdf_layout/attention_is_all_you_need.json"))
    parser.add_argument("--docling-dir", type=Path, default=Path("data/m3/layout_batch"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs/m3_pdf_layout_10pdf_20261017/report.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    label_set = json.loads(args.batch_labels.read_text(encoding="utf-8"))
    labels = {entry["id"]: entry for entry in label_set["documents"]}
    attention = json.loads(args.attention_label.read_text(encoding="utf-8"))
    if set(labels) != {entry["id"] for entry in manifest["documents"]} - {"attention"}:
        parser.error("Batch label IDs do not match the frozen PDF manifest")
    docling_runs_path = args.docling_dir / "run_summary.json"
    docling_runs = {item["id"]: item for item in json.loads(docling_runs_path.read_text(encoding="utf-8"))}
    reports = []
    for entry in manifest["documents"]:
        doc_id = entry["id"]
        pdf = Path(entry["pdf"])
        gold = attention if doc_id == "attention" else dict(labels[doc_id])
        gold.setdefault("reviewed_pages", list(range(1, gold["page_count"] + 1)))
        gold.setdefault("review_status", label_set["review_status"])
        if digest(pdf) != gold["pdf_sha256"]:
            parser.error(f"PDF SHA-256 differs from label: {doc_id}")
        docling_path = args.docling_dir / f"{doc_id}_docling.json"
        if docling_runs.get(doc_id, {}).get("status") != "completed" or not docling_path.is_file():
            parser.error(f"Docling run missing or failed: {doc_id}")
        docling = json.loads(docling_path.read_text(encoding="utf-8"))
        started = perf_counter()
        pymupdf = PDFParser().parse(pdf)
        pymupdf_seconds = perf_counter() - started
        if len(pymupdf) != gold["page_count"]:
            parser.error(f"PDF page count differs from label: {doc_id}")
        report = evaluate(gold, docling, pymupdf)
        report.update({"id": doc_id, "split": entry["split"], "pdf": entry["pdf"],
                       "pdf_sha256": gold["pdf_sha256"],
                       "docling_records_sha256": digest(docling_path),
                       "runtime": {"pymupdf_parse_seconds": round(pymupdf_seconds, 3),
                                   "docling_parse_seconds_prior_run": docling_runs[doc_id]["seconds"]}})
        reports.append(report)
    result = {
        "task": "m3_pdf_layout_ten_document_first_pass",
        "label_review_status": label_set["review_status"],
        "manifest_sha256": digest(args.manifest),
        "batch_labels_sha256": digest(args.batch_labels),
        "attention_label_sha256": digest(args.attention_label),
        "summary": aggregate(reports),
        "splits": {split: aggregate([item for item in reports if item["split"] == split])
                   for split in ("dev", "test")},
        "documents": reports,
        "limits": [
            "First-pass visual labels need independent second review before published accuracy claims.",
            "All ten PDFs contain extractable text; scanned-PDF behavior is untested.",
            "Table detection uses caption/page matching plus unambiguous unique-page fallback, not geometric IoU.",
            "Only sampled rows and questions measure value availability; no full-cell transcription is scored.",
            "These results do not measure retrieval or generated answer quality.",
            "Parser timings came from separate runs/images and are exploratory only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": result["summary"],
                      "splits": result["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
