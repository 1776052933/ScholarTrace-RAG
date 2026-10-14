"""Resolve AutoPaper metadata to LitSearch Semantic Scholar CorpusIds."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import requests


API_ROOT = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarResolver:
    """Resolve DOI/arXiv identifiers in batches and use title matching as fallback."""

    def __init__(
        self,
        cache_path: Path,
        *,
        api_key: str | None = None,
        timeout: int = 25,
        minimum_interval: float = 1.1,
    ) -> None:
        self.cache_path = cache_path.resolve()
        self.timeout = timeout
        self.minimum_interval = minimum_interval
        self.session = requests.Session()
        key = (api_key or _project_key()).strip()
        if key:
            self.session.headers["x-api-key"] = key
        self.session.headers["User-Agent"] = "AutoPaper-LitSearch-Evaluator/1.0"
        self.cache = self._load_cache()
        self.last_request_at = 0.0
        self.api_call_count = 0

    def resolve_many(
        self,
        papers: Sequence[dict[str, Any]],
        *,
        allow_title_matching: bool = True,
        max_title_lookups: int = 10,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(papers)
        pending_identifiers: list[tuple[int, str]] = []
        for index, paper in enumerate(papers):
            identifier = _paper_identifier(paper)
            cache_key = f"identifier:{identifier}" if identifier else ""
            if cache_key and cache_key in self.cache:
                results[index] = dict(self.cache[cache_key])
            elif identifier:
                pending_identifiers.append((index, identifier))

        if pending_identifiers:
            payload = self._request_json(
                "POST",
                "/paper/batch",
                params={"fields": "paperId,corpusId,title,year,externalIds"},
                json_body={"ids": [item[1] for item in pending_identifiers]},
            )
            if isinstance(payload, list):
                for (index, identifier), item in zip(pending_identifiers, payload):
                    if not isinstance(item, dict) or item.get("corpusId") is None:
                        continue
                    resolution = _resolution(item, "identifier", papers[index])
                    results[index] = resolution
                    self.cache[f"identifier:{identifier}"] = resolution

        title_lookups = 0
        if allow_title_matching:
            for index, paper in enumerate(papers):
                if results[index] is not None or title_lookups >= max_title_lookups:
                    continue
                title = str(paper.get("title") or "").strip()
                if not title:
                    continue
                cache_key = f"title:{_normalize_title(title)}|{paper.get('year') or ''}"
                if cache_key in self.cache:
                    results[index] = dict(self.cache[cache_key])
                    continue
                title_lookups += 1
                item = self._request_json(
                    "GET",
                    "/paper/search/match",
                    params={
                        "query": title,
                        "fields": "paperId,corpusId,title,year,externalIds",
                    },
                )
                if not isinstance(item, dict) or item.get("corpusId") is None:
                    continue
                confidence = _title_similarity(title, str(item.get("title") or ""))
                if confidence < 0.90 or not _year_matches(paper.get("year"), item.get("year")):
                    continue
                resolution = _resolution(item, "title", paper, confidence=confidence)
                results[index] = resolution
                self.cache[cache_key] = resolution

        self._save_cache()
        return [
            result or {
                "corpus_id": None,
                "paper_id": None,
                "matched_title": None,
                "method": "unresolved",
                "confidence": 0.0,
            }
            for result in results
        ]

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        for attempt in range(3):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            self.last_request_at = time.monotonic()
            self.api_call_count += 1
            try:
                response = self.session.request(
                    method,
                    API_ROOT + path,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                if attempt == 2:
                    return None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429:
                if attempt == 2:
                    return None
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(min(max(delay, 1.0), 20.0))
                continue
            if response.status_code >= 400:
                return None
            try:
                return response.json()
            except ValueError:
                return None
        return None

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


def _paper_identifier(paper: dict[str, Any]) -> str | None:
    doi = _normalize_doi(str(paper.get("doi") or ""))
    if doi:
        return f"DOI:{doi}"

    values = [str(paper.get(field) or "") for field in ("url", "pdf_url", "paper_id")]
    for value in values:
        match = re.search(
            r"(?i)(?:doi\.org/|/doi/(?:abs/|pdf/)?|/content/pdf/|/article/)"
            r"(10\.\d{4,9}/[^/?#\s]+)",
            value,
        )
        if match:
            doi = _normalize_doi(match.group(1))
            if doi:
                return f"DOI:{doi}"
        match = re.fullmatch(r"\s*(10\.\d{4,9}/\S+)\s*", value, flags=re.IGNORECASE)
        if match:
            doi = _normalize_doi(match.group(1))
            if doi:
                return f"DOI:{doi}"

    arxiv_id = _normalize_arxiv_id(str(paper.get("arxiv_id") or ""))
    if not arxiv_id:
        for value in values:
            match = re.search(
                r"(?i)arxiv(?:\.org/(?:abs|pdf)/|:)"
                r"([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
                value,
            )
            if match:
                arxiv_id = _normalize_arxiv_id(match.group(1))
                break
    if arxiv_id:
        return f"ARXIV:{arxiv_id}"

    for value in values:
        match = re.search(
            r"(?i)(?:aclanthology\.org/|aclweb\.org/anthology/)"
            r"([a-z0-9][a-z0-9.-]*?)(?:\.pdf)?(?:[/?#]|$)",
            value,
        )
        if match:
            return f"ACL:{match.group(1)}"
    return None


def _normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"(?i)^doi:\s*", "", value)
    value = value.split("?", 1)[0].split("#", 1)[0]
    value = re.sub(r"(?i)\.pdf$", "", value)
    return value.rstrip("/.,; ")


def _normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"(?i)^arxiv:\s*", "", value)
    value = value.split("?", 1)[0].split("#", 1)[0]
    value = re.sub(r"(?i)\.pdf$", "", value)
    value = re.sub(r"(?i)v\d+$", "", value)
    return value.rstrip("/ ")


def _resolution(
    item: dict[str, Any],
    method: str,
    paper: dict[str, Any],
    *,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "corpus_id": int(item["corpusId"]),
        "paper_id": item.get("paperId"),
        "matched_title": item.get("title"),
        "method": method,
        "confidence": round(float(confidence), 6),
        "input_title": paper.get("title"),
    }


def _normalize_title(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize_title(left), _normalize_title(right)).ratio()


def _year_matches(left: Any, right: Any) -> bool:
    if left in (None, "") or right in (None, ""):
        return True
    try:
        return abs(int(left) - int(right)) <= 1
    except (TypeError, ValueError):
        return True


def _project_key() -> str:
    value = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if value:
        return value
    try:
        from API_key.key_manager import key_get

        return str(key_get("semantic_scholar") or "").strip()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return ""
