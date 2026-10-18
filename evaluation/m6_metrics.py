"""Frozen M6 PDF cases and deterministic evidence metrics.

These metrics check whether an annotated page and answer fragment are present.
They do not decide whether a generated claim is supported by its citation.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence
import unicodedata


GOLD_ROOT = Path(__file__).parent / "gold"


def digest(path: Path) -> str:
    # Git may check these JSON fixtures out with CRLF on Windows. M3 indexed
    # their LF bytes inside Linux, so compare canonical text across platforms.
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_multi_cases(root: Path = GOLD_ROOT, *, cases_name: str = "m6_multi_paper.json",
                     require_both_splits: bool = True) -> tuple[list[dict[str, Any]], dict[str, str]]:
    layout = root / "m3_pdf_layout"
    paths = {
        "pdf_manifest": layout / "manifest.json",
        "batch_labels": layout / "batch_labels.json",
        "attention_label": layout / "attention_is_all_you_need.json",
        "multi_cases": root / cases_name,
    }
    files = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    documents = {row["id"]: row for row in files["pdf_manifest"]["documents"]}
    labels = {row["id"]: row for row in files["batch_labels"]["documents"]}
    labels["attention"] = files["attention_label"]
    if set(documents) != set(labels):
        raise ValueError("PDF manifest and label document IDs differ")
    if files["multi_cases"].get("review_status") != "derived_from_m3_first_pass_labels_needs_independent_review":
        raise ValueError("unexpected M6 gold review status")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in files["multi_cases"]["cases"]:
        case_id = row["id"]
        if case_id in seen or row["split"] not in {"dev", "test"}:
            raise ValueError(f"duplicate case ID or invalid split: {case_id}")
        seen.add(case_id)
        if len(row["references"]) < 2 or len({ref["document_id"] for ref in row["references"]}) != len(row["references"]):
            raise ValueError(f"case needs at least two distinct papers: {case_id}")
        references = []
        for ref in row["references"]:
            document_id = ref["document_id"]
            if document_id not in documents or documents[document_id]["split"] != row["split"]:
                raise ValueError(f"cross-split or unknown PDF in {case_id}: {document_id}")
            matches = [item for item in labels[document_id]["evidence_questions"]
                       if item["id"] == ref["source_question_id"]]
            if len(matches) != 1 or not matches[0].get("answer_fragments"):
                raise ValueError(f"missing source question in {case_id}: {ref}")
            question = matches[0]
            references.append({
                "document_id": document_id,
                "source_question_id": question["id"],
                "page": question["page"],
                "answer_fragments": question["answer_fragments"],
            })
        cases.append({"id": case_id, "split": row["split"],
                      "question": row["question"], "references": references})
    if not cases or (require_both_splits and not {"dev", "test"}.issubset({row["split"] for row in cases})):
        raise ValueError("M6 cases must contain both dev and test questions")
    return cases, {name: digest(path) for name, path in paths.items()}


def score_evidence(chunks: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score page and answer-fragment hits separately for each selected paper."""
    if not references:
        raise ValueError("At least one gold reference is required")
    hits = []
    for gold in references:
        paper_id = gold["document_id"]
        matching_paper = [(index, chunk) for index, chunk in enumerate(chunks, 1)
                          if chunk.get("paper_id") == paper_id]
        matching_page = [(index, chunk) for index, chunk in matching_paper
                         if chunk.get("page") == gold["page"]]
        matching_fragments = [index for index, chunk in matching_page
                              if all(_normalized(fragment) in _normalized(str(chunk.get("text") or ""))
                                     for fragment in gold["answer_fragments"])]
        hits.append({"document_id": paper_id,
                     "paper_present": bool(matching_paper),
                     "page_present": bool(matching_page),
                     "answer_fragment_present": bool(matching_fragments),
                     "first_answer_rank": matching_fragments[0] if matching_fragments else None})
    count = len(hits)
    return {
        "paper_coverage": sum(item["paper_present"] for item in hits) / count,
        "page_recall": sum(item["page_present"] for item in hits) / count,
        "answer_fragment_recall": sum(item["answer_fragment_present"] for item in hits) / count,
        "all_gold_fragments_found": all(item["answer_fragment_present"] for item in hits),
        "chunk_count": len(chunks),
        "per_paper": hits,
    }


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[ceil(0.95 * len(ordered)) - 1]


def summarize_variants(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        rows = [record["variants"][name] for record in records if name in record["variants"]]
        completed = [item for item in rows if item.get("status") == "completed"]
        seconds = [float(item["seconds"]) for item in completed]
        result[name] = {
            "cases": len(rows), "completed": len(completed), "failed": len(rows) - len(completed),
            "paper_coverage": (sum(item["metrics"]["paper_coverage"] for item in completed) / len(completed)
                               if completed else None),
            "page_recall": (sum(item["metrics"]["page_recall"] for item in completed) / len(completed)
                            if completed else None),
            "answer_fragment_recall": (sum(item["metrics"]["answer_fragment_recall"] for item in completed) / len(completed)
                                       if completed else None),
            "all_gold_fragments_found": sum(item["metrics"]["all_gold_fragments_found"] for item in completed),
            "mean_chunk_count": (sum(item["metrics"]["chunk_count"] for item in completed) / len(completed)
                                 if completed else None),
            "latency_p50_seconds": median(seconds) if seconds else None,
            "latency_p95_seconds": _p95(seconds),
        }
    return result
