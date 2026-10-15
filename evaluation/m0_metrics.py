"""Small, explicitly non-official proxies for the M0 QASPER run."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any, Mapping, Sequence


def tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)


def token_f1(prediction: str, reference: str) -> float:
    a, b = Counter(tokens(prediction)), Counter(tokens(reference))
    if not a and not b:
        return 1.0
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values())) if overlap else 0.0


def reference_answer(annotation: Mapping[str, Any]) -> str:
    answer = annotation.get("answer") or {}
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    spans = answer.get("extractive_spans") or []
    if spans:
        return " ".join(str(span) for span in spans)
    return str(answer.get("free_form_answer") or "").strip()


def evidence_coverage(prediction: str, reference: str) -> float:
    """Fraction of gold evidence tokens covered by one retrieved chunk."""
    a, b = Counter(tokens(prediction)), Counter(tokens(reference))
    return sum((a & b).values()) / sum(b.values()) if b else 0.0


def evidence_recall(
    annotations: Sequence[Mapping[str, Any]],
    retrieved_texts: Sequence[str],
    *,
    threshold: float = 0.8,
) -> float | None:
    """Best annotator recall, matching highlighted spans by token coverage.

    This is a diagnostic proxy, not QASPER's official evidence score. Empty
    evidence on unanswerable questions is excluded from the denominator.
    """
    scores = []
    for annotation in annotations:
        answer = annotation.get("answer") or {}
        evidence = answer.get("highlighted_evidence") or answer.get("evidence") or []
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        if not evidence:
            continue
        hits = sum(
            any(evidence_coverage(chunk, gold) >= threshold for chunk in retrieved_texts)
            for gold in evidence
        )
        scores.append(hits / len(evidence))
    return max(scores) if scores else None


def answer_f1(prediction: str, annotations: Sequence[Mapping[str, Any]]) -> float | None:
    references = [reference_answer(annotation) for annotation in annotations]
    references = [reference for reference in references if reference]
    return max((token_f1(prediction, reference) for reference in references), default=None)


