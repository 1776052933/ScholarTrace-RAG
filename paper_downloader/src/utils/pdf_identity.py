"""Verify that a downloaded PDF belongs to the requested paper title."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Optional


_STOP_WORDS = {
    "a", "an", "and", "by", "for", "from", "in", "of", "on", "the", "to", "with",
}


def pdf_matches_title(
    file_path: str | Path,
    expected_title: str,
    *,
    threshold: float = 0.82,
) -> Optional[bool]:
    """Return True/False for a readable PDF, or None when text cannot be inspected."""
    score = pdf_title_similarity(file_path, expected_title)
    return None if score is None else score >= threshold


def pdf_title_similarity(file_path: str | Path, expected_title: str) -> Optional[float]:
    expected = _normalized(expected_title)
    if not expected:
        return None
    metadata_title, first_page = _pdf_identity_text(Path(file_path))
    if not metadata_title and not first_page:
        return None
    candidates = []
    if metadata_title:
        candidates.append(metadata_title)
    candidates.extend(_title_region_candidates(first_page, len(expected.split())))
    return max(
        (title_similarity(expected, candidate) for candidate in candidates),
        default=0.0,
    )


def title_similarity(left: str, right: str) -> float:
    """Return deterministic title similarity without optional dependencies."""
    left_normalized = _normalized(left)
    right_normalized = _normalized(right)
    if not left_normalized or not right_normalized:
        return 0.0
    direct = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    token_sorted = SequenceMatcher(
        None,
        " ".join(sorted(left_normalized.split())),
        " ".join(sorted(right_normalized.split())),
    ).ratio()
    return max(direct, token_sorted)


def _pdf_identity_text(path: Path) -> tuple[str, str]:
    if not path.is_file() or path.suffix.casefold() != ".pdf":
        return "", ""
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        metadata = getattr(reader, "metadata", None)
        metadata_title = str(getattr(metadata, "title", "") or "")
        first_page = reader.pages[0].extract_text() if reader.pages else ""
        return metadata_title, first_page or ""
    except Exception:
        return "", ""


def _title_region_candidates(text: str, expected_tokens: int) -> list[str]:
    """Extract likely title lines before the abstract/body begins."""
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title_lines = []
    for line in lines[:40]:
        normalized = _normalized(line)
        marker = normalized.split()[:2]
        if marker and marker[0] in {"abstract", "introduction", "keywords", "summary"}:
            break
        title_lines.append(line)

    candidates = []
    for start in range(len(title_lines)):
        for width in range(1, min(5, len(title_lines) - start) + 1):
            candidates.append(" ".join(title_lines[start:start + width]))

    region = " ".join(title_lines)
    tokens = _normalized(region).split()[:250]
    if tokens and expected_tokens:
        for width in range(max(3, expected_tokens - 2), expected_tokens + 3):
            for start in range(0, max(0, len(tokens) - width + 1)):
                candidates.append(" ".join(tokens[start:start + width]))
    return candidates


def _normalized(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[-\u2010-\u2015]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
