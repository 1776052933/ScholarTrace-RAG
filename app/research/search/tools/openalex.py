"""OpenAlex Works API paper search."""

import json
import logging
import os
import re
from typing import Dict, List
from urllib import error, parse, request

from ..schema import PaperMetadata


logger = logging.getLogger(__name__)


class OpenAlexTool:
    source = "openalex"

    def __init__(
        self,
        mailto: str = "",
        timeout_seconds: float = 20.0,
        api_key: str = "",
    ):
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key.strip()

    def search(
        self,
        query: str,
        limit: int = 10,
        publication_date_to: str | None = None,
    ) -> List[PaperMetadata]:
        if not query.strip() or limit < 1:
            return []
        values = {"search": query, "per-page": min(limit, 100)}
        cutoff = _iso_date(publication_date_to)
        if cutoff:
            values["filter"] = f"to_publication_date:{cutoff}"
        if self.api_key:
            values["api_key"] = self.api_key
        if self.mailto:
            values["mailto"] = self.mailto
        req = request.Request(
            f"https://api.openalex.org/works?{parse.urlencode(values)}",
            headers={"User-Agent": "Auto-paper-search-agent/1.0"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.warning("OpenAlex API request failed: %s", _safe_error(exc))
            return []

        papers = []
        for item in payload.get("results", []):
            try:
                paper = paper_from_openalex_work(item)
                if paper is not None:
                    papers.append(paper)
            except (TypeError, ValueError):
                logger.debug("Skipping malformed OpenAlex work", exc_info=True)
        return papers


def search(query: str, limit: int = 10) -> List[PaperMetadata]:
    return OpenAlexTool(
        mailto=os.getenv("OPENALEX_MAILTO", ""),
        api_key=os.getenv("OPENALEX_API_KEY", ""),
    ).search(query, limit)


def paper_from_openalex_work(
    item: dict,
    *,
    source: str = "openalex",
) -> PaperMetadata | None:
    if not item.get("title"):
        return None
    primary = item.get("primary_location") or {}
    best_oa = item.get("best_oa_location") or {}
    venue = (primary.get("source") or {}).get("display_name") or ""
    ids = item.get("ids") or {}
    return PaperMetadata(
        title=item["title"],
        authors=[
            authorship.get("author", {}).get("display_name", "")
            for authorship in item.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ],
        abstract=_abstract(item.get("abstract_inverted_index") or {}),
        year=item.get("publication_year"),
        publication_date=item.get("publication_date"),
        venue=venue,
        citation_count=item.get("cited_by_count"),
        url=primary.get("landing_page_url") or item.get("id") or "",
        source=source,
        doi=_normalize_doi(item.get("doi")),
        arxiv_id=_normalize_arxiv(ids.get("arxiv")) or _arxiv_from_locations(item),
        openalex_id=_normalize_openalex_id(item.get("id")),
        pdf_url=best_oa.get("pdf_url") or primary.get("pdf_url") or None,
        sources=_unique(["openalex", source]),
    )


def _abstract(index: Dict[str, List[int]]) -> str:
    positions = [
        (position, word)
        for word, values in index.items()
        for position in values
    ]
    return " ".join(word for _, word in sorted(positions))


def _normalize_doi(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        value.strip(),
        flags=re.I,
    )


def _normalize_arxiv(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(
        r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/",
        "",
        value.strip(),
        flags=re.I,
    )
    return text.removesuffix(".pdf")


def _normalize_openalex_id(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    identifier = value.strip().rstrip("/").rsplit("/", 1)[-1]
    return identifier if re.fullmatch(r"W\d+", identifier, re.I) else None


def _arxiv_from_locations(item: dict) -> str | None:
    for location in item.get("locations") or []:
        url = str((location or {}).get("landing_page_url") or "")
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
            url,
            re.I,
        )
        if match:
            return match.group(1)
    return None


def _iso_date(value: str | None) -> str:
    digits = "".join(character for character in (value or "") if character.isdigit())
    if len(digits) < 8:
        return ""
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, error.URLError):
        return f"network error: {exc.reason}"
    return type(exc).__name__


def _unique(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
