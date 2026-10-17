"""Compare PDF parsing outputs against independently reviewed page labels.

This evaluates table detection for Docling and text/row availability for both
parsers. PyMuPDF's current production path emits page text, not table objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_reader.parser import PDFParser


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def matching_line(text: str, anchor: str, values: list[str]) -> bool:
    anchor = normalized(anchor)
    values = [normalized(value) for value in values]
    return any(
        anchor in normalized(line) and all(value in normalized(line) for value in values)
        for line in text.splitlines()
    )


def contains_fragments(text: str, fragments: list[str]) -> bool:
    whole = normalized(text)
    return all(normalized(fragment) in whole for fragment in fragments)


def evaluate(gold: dict[str, Any], docling: list[dict[str, Any]], pymupdf: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = set(gold["reviewed_pages"])
    if not reviewed or any(page < 1 or page > gold["page_count"] for page in reviewed):
        raise ValueError("reviewed_pages must be valid 1-based PDF pages")
    page_text = {int(item["page"]): str(item["text"]) for item in pymupdf}
    docling_pages: dict[int, str] = {}
    for item in docling:
        page = int(item["page"])
        docling_pages[page] = docling_pages.get(page, "") + "\n" + str(item["text"])

    candidates = [item for item in docling if item.get("metadata", {}).get("item_type") == "table"
                  and int(item["page"]) in reviewed]
    matched: set[int] = set()
    table_matches: dict[str, tuple[int, str]] = {}
    # Match named captions first, then use the page only if exactly one gold
    # table and one Docling candidate remain there. This handles a table whose
    # caption is separate from its grid without guessing among multiple grids.
    for table in gold["tables"]:
        matches = [index for index, candidate in enumerate(candidates)
                   if index not in matched and int(candidate["page"]) == int(table["page"])
                   and normalized(table["caption_prefix"]) in normalized(candidate["text"])]
        if len(matches) == 1:
            table_matches[table["id"]] = (matches[0], "caption_and_page")
            matched.add(matches[0])
    for page in reviewed:
        remaining_gold = [table for table in gold["tables"]
                          if int(table["page"]) == page and table["id"] not in table_matches]
        remaining_candidates = [index for index, candidate in enumerate(candidates)
                                if int(candidate["page"]) == page and index not in matched]
        if len(remaining_gold) == len(remaining_candidates) == 1:
            table_matches[remaining_gold[0]["id"]] = (remaining_candidates[0], "unique_remaining_on_page")
            matched.add(remaining_candidates[0])
    table_results = []
    for table in gold["tables"]:
        page = int(table["page"])
        if page not in reviewed:
            raise ValueError(f"Gold table on unreviewed page: {table['id']}")
        match = table_matches.get(table["id"])
        candidate = candidates[match[0]] if match else None
        row_results = []
        for check in table["row_checks"]:
            row_results.append({
                "row_contains": check["row_contains"],
                "values": check["values"],
                "docling_table_row": bool(candidate and matching_line(candidate["text"], check["row_contains"], check["values"])),
                "pymupdf_page_row": matching_line(page_text.get(page, ""), check["row_contains"], check["values"]),
            })
        table_results.append({"id": table["id"], "page": page,
                              "docling_detected": candidate is not None,
                              "match_method": match[1] if match else None,
                              "docling_source_ref": candidate.get("metadata", {}).get("source_ref") if candidate else None,
                              "row_checks": row_results})

    false_positives = [
        {"page": int(item["page"]), "source_ref": item.get("metadata", {}).get("source_ref"),
         "text_preview": re.sub(r"\s+", " ", str(item["text"]))[:160]}
        for index, item in enumerate(candidates) if index not in matched
    ]
    non_table_results = [
        {"id": item["id"], "page": item["page"],
         "docling_misclassified_as_table": any(
             int(candidate["page"]) == item["page"]
             and normalized(item["anchor"]) in normalized(candidate["text"])
             for candidate in candidates)}
        for item in gold.get("non_tables", [])
    ]
    heading_results = [
        {"page": item["page"], "text": item["text"],
         "docling_section_header": any(
             int(record["page"]) == item["page"]
             and record.get("metadata", {}).get("item_type") == "section_header"
             and normalized(record["text"]) == normalized(item["text"])
             for record in docling),
         "pymupdf_page_text": normalized(item["text"]) in normalized(page_text.get(item["page"], ""))}
        for item in gold.get("headings", [])
    ]
    evidence_results = []
    table_lookup = {table["id"]: table for table in gold["tables"]}
    for item in gold.get("evidence_questions", []):
        page = int(item["page"])
        if item["source_type"] == "table":
            table = table_lookup[item["table_id"]]
            match = table_matches.get(table["id"])
            matching_table = candidates[match[0]] if match else None
            if item.get("row_contains"):
                docling_found = bool(matching_table and matching_line(
                    matching_table["text"], item["row_contains"], item["answer_fragments"]))
                pymupdf_found = matching_line(page_text.get(page, ""), item["row_contains"], item["answer_fragments"])
            else:
                docling_found = bool(matching_table and contains_fragments(
                    matching_table["text"], item["answer_fragments"]))
                pymupdf_found = contains_fragments(page_text.get(page, ""), item["answer_fragments"])
        else:
            docling_found = contains_fragments(docling_pages.get(page, ""), item["answer_fragments"])
            pymupdf_found = contains_fragments(page_text.get(page, ""), item["answer_fragments"])
        evidence_results.append({"id": item["id"], "page": page,
                                 "docling_correct_page_fragments": docling_found,
                                 "pymupdf_correct_page_fragments": pymupdf_found})

    true_positives = len(matched)
    false_positives_count = len(false_positives)
    false_negatives = len(gold["tables"]) - true_positives
    all_rows = [row for table in table_results for row in table["row_checks"]]
    return {
        "scope": {"reviewed_pages": sorted(reviewed), "review_status": gold["review_status"],
                  "gold_tables": len(gold["tables"]), "docling_table_candidates": len(candidates)},
        "metrics": {
            "docling_table_detection_precision": true_positives / (true_positives + false_positives_count)
            if candidates else None,
            "docling_table_detection_recall": true_positives / len(gold["tables"])
            if gold["tables"] else None,
            "docling_table_tp": true_positives,
            "docling_table_fp": false_positives_count,
            "docling_table_fn": false_negatives,
            "docling_row_checks": sum(row["docling_table_row"] for row in all_rows),
            "pymupdf_row_checks": sum(row["pymupdf_page_row"] for row in all_rows),
            "row_checks_total": len(all_rows),
            "docling_heading_labels_found": sum(item["docling_section_header"] for item in heading_results),
            "pymupdf_heading_text_found": sum(item["pymupdf_page_text"] for item in heading_results),
            "headings_total": len(heading_results),
            "docling_evidence_page_fragments": sum(item["docling_correct_page_fragments"] for item in evidence_results),
            "pymupdf_evidence_page_fragments": sum(item["pymupdf_correct_page_fragments"] for item in evidence_results),
            "evidence_questions_total": len(evidence_results),
        },
        "tables": table_results, "docling_false_positive_tables": false_positives,
        "non_tables": non_table_results, "headings": heading_results,
        "evidence_questions": evidence_results,
        "metric_limits": [
            "Only reviewed pages form the table-detection denominator.",
            "PyMuPDF emits page text, so its table detection precision/recall is not defined here.",
            "Row checks and answer fragments test same-line availability where a row anchor exists, not exact cell geometry, retrieval, or generated answer quality.",
            "This pilot document and its labels need an independent second review before a published claim.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--docling-records", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    docling = json.loads(args.docling_records.read_text(encoding="utf-8"))
    pdf_hash = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if pdf_hash != gold["pdf_sha256"]:
        parser.error(f"PDF SHA-256 differs from label: {pdf_hash}")
    started = perf_counter()
    pymupdf = PDFParser().parse(args.pdf)
    elapsed = perf_counter() - started
    if len(pymupdf) != gold["page_count"]:
        parser.error(f"PDF page count differs from label: {len(pymupdf)}")
    report = evaluate(gold, docling, pymupdf)
    report["inputs"] = {"pdf_sha256": pdf_hash,
                        "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
                        "docling_records_sha256": hashlib.sha256(args.docling_records.read_bytes()).hexdigest()}
    report["runtime"] = {"pymupdf_parse_seconds": round(elapsed, 3),
                         "note": "Docling records are loaded from a prior run; this script does not time Docling."}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
