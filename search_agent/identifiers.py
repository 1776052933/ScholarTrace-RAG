"""Paper identifier extraction and stable identity keys."""

from __future__ import annotations

import re
from typing import Any, Iterable


_DOI_PATTERN = re.compile(r"(?i)(10\.\d{4,9}/[^\s\"'<>?#]+)")
_ARXIV_PATTERN = re.compile(
    r"(?i)(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)"
    r"([a-z-]+(?:\.[a-z-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?"
)
_OPENALEX_PATTERN = re.compile(r"(?i)(?:openalex\.org/)?(W\d+)")


def normalize_doi(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"(?i)^doi\s*:\s*", "", text)
    text = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", text)
    match = _DOI_PATTERN.search(text)
    if match:
        text = match.group(1)
    text = text.rstrip(".,;:)]}").casefold()
    return text if text.startswith("10.") and "/" in text else None


def extract_doi(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_doi(value)
        if normalized:
            return normalized
    return None


def normalize_arxiv_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _ARXIV_PATTERN.search(text)
    if match:
        return match.group(1).casefold()
    text = re.sub(r"(?i)^arxiv\s*:\s*", "", text)
    text = re.sub(r"(?i)^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text)
    text = re.sub(r"(?i)\.pdf(?:\?.*)?$", "", text)
    text = re.sub(r"(?i)v\d+$", "", text).strip()
    if re.fullmatch(r"(?:[a-z-]+(?:\.[a-z-]+)?/\d{7}|\d{4}\.\d{4,5})", text, re.I):
        return text.casefold()
    return None


def extract_arxiv_id(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_arxiv_id(value)
        if normalized:
            return normalized
    return None


def normalize_openalex_id(value: Any) -> str | None:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return None
    match = _OPENALEX_PATTERN.search(text.rsplit("/", 1)[-1])
    return match.group(1).upper() if match else None


def extract_openalex_id(*values: Any) -> str | None:
    for value in values:
        normalized = normalize_openalex_id(value)
        if normalized:
            return normalized
    return None


def normalize_title(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum()
    )


def enrich_paper_identifiers(paper: Any) -> Any:
    """Fill identifiers that are embedded in URLs without changing other metadata."""
    values = (
        getattr(paper, "doi", None),
        getattr(paper, "arxiv_id", None),
        getattr(paper, "openalex_id", None),
        getattr(paper, "url", None),
        getattr(paper, "pdf_url", None),
        getattr(paper, "paper_id", None),
    )
    updates = {
        "doi": extract_doi(*values),
        "arxiv_id": extract_arxiv_id(*values),
        "openalex_id": extract_openalex_id(*values),
    }
    updates = {
        key: value
        for key, value in updates.items()
        if value and hasattr(paper, key) and not getattr(paper, key, None)
    }
    return paper.model_copy(update=updates) if updates else paper


def paper_identity_keys(paper: Any, *, include_title: bool = True) -> set[str]:
    enriched = enrich_paper_identifiers(paper)
    keys: set[str] = set()
    doi = normalize_doi(getattr(enriched, "doi", None))
    arxiv_id = normalize_arxiv_id(getattr(enriched, "arxiv_id", None))
    openalex_id = normalize_openalex_id(getattr(enriched, "openalex_id", None))
    if doi:
        keys.add(f"doi:{doi}")
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")
    if openalex_id:
        keys.add(f"openalex:{openalex_id.casefold()}")
    if include_title:
        title = normalize_title(getattr(enriched, "title", ""))
        if title:
            keys.add(f"title:{title}")
    return keys


def canonical_paper_key(paper: Any) -> str | None:
    keys = paper_identity_keys(paper)
    for prefix in ("doi:", "arxiv:", "openalex:", "title:"):
        match = next((key for key in keys if key.startswith(prefix)), None)
        if match:
            return match
    return None


def unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result
