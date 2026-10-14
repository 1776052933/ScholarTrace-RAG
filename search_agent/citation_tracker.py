"""Budgeted two-hop OpenAlex citation expansion with BGE filtering and caching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any, Iterable, Mapping, Optional
from urllib import error, parse, request

from ranking_agent.embedding import EmbeddingModel
from ranking_agent.similarity import SimilarityCalculator

from .identifiers import (
    canonical_paper_key,
    enrich_paper_identifiers,
    normalize_arxiv_id,
    normalize_doi,
    normalize_openalex_id,
    paper_identity_keys,
    unique_strings,
)
from .schema import PaperMetadata
from .searcher import deduplicate_papers
from .tools.openalex import paper_from_openalex_work


logger = logging.getLogger(__name__)
OPENALEX_API_URL = "https://api.openalex.org"


@dataclass(frozen=True)
class CitationExpansion:
    papers: list[PaperMetadata]
    summary: dict[str, Any]


class CitationCache:
    def __init__(self, path: str | Path, *, ttl_days: int = 30) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=max(1, int(ttl_days)))
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS citation_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get(self, key: str) -> tuple[bool, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, updated_at FROM citation_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return False, None
        try:
            updated = datetime.fromisoformat(row[1])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - updated > self.ttl:
                return False, None
            return True, json.loads(row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            return False, None

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        updated = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO citation_cache(cache_key, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (key, payload, updated),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)


class OpenAlexCitationClient:
    def __init__(
        self,
        api_key: str,
        *,
        mailto: str = "",
        timeout_seconds: float = 20.0,
        reference_limit: int = 24,
        citation_limit: int = 20,
    ) -> None:
        self.api_key = api_key.strip()
        self.mailto = mailto.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.reference_limit = max(1, min(100, int(reference_limit)))
        self.citation_limit = max(1, min(100, int(citation_limit)))
        self.api_call_count = 0
        self._counter_lock = Lock()

    def resolve(self, paper: PaperMetadata) -> dict[str, Any] | None:
        if paper.openalex_id:
            return self._get_work(paper.openalex_id)
        if paper.doi:
            return self._get_work(f"doi:{_normalize_doi(paper.doi)}")
        arxiv_id = _normalize_arxiv_id(paper.arxiv_id)
        if not arxiv_id:
            return None
        payload = self._get_json(
            "/works",
            {
                "filter": (
                    "locations.landing_page_url:"
                    f"https://arxiv.org/abs/{arxiv_id}"
                ),
                "per-page": 5,
            },
        )
        for item in payload.get("results", []) if isinstance(payload, dict) else []:
            parsed = paper_from_openalex_work(item)
            if parsed and _normalize_arxiv_id(parsed.arxiv_id) == arxiv_id:
                return item
        return None

    def neighbors(
        self,
        work: Mapping[str, Any],
        *,
        publication_date_to: str | None = None,
    ) -> dict[str, list[dict[str, Any]]] | None:
        openalex_id = _short_openalex_id(work.get("id"))
        if not openalex_id:
            return {"references": [], "citations": []}
        reference_ids = [
            _short_openalex_id(value)
            for value in work.get("referenced_works") or []
        ]
        reference_ids = [value for value in reference_ids if value][:self.reference_limit]
        references = self._works_by_ids(reference_ids, publication_date_to)

        filters = [f"cites:{openalex_id}"]
        if publication_date_to:
            filters.append(f"to_publication_date:{publication_date_to}")
        payload = self._get_json(
            "/works",
            {
                "filter": ",".join(filters),
                "sort": "cited_by_count:desc",
                "per-page": self.citation_limit,
            },
        )
        if references is None or not isinstance(payload, dict):
            return None
        citations = payload.get("results", [])
        return {"references": references, "citations": citations}

    def _works_by_ids(
        self,
        identifiers: list[str],
        publication_date_to: str | None,
    ) -> list[dict[str, Any]] | None:
        if not identifiers:
            return []
        filters = [f"openalex_id:{'|'.join(identifiers)}"]
        if publication_date_to:
            filters.append(f"to_publication_date:{publication_date_to}")
        payload = self._get_json(
            "/works",
            {"filter": ",".join(filters), "per-page": min(100, len(identifiers))},
        )
        return payload.get("results", []) if isinstance(payload, dict) else None

    def _get_work(self, identifier: str) -> dict[str, Any] | None:
        path_identifier = parse.quote(identifier, safe="")
        payload = self._get_json(f"/works/{path_identifier}", {})
        return payload if isinstance(payload, dict) and payload.get("id") else None

    def _get_json(self, path: str, values: dict[str, Any]) -> Any:
        params = dict(values)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{OPENALEX_API_URL}{path}"
        if params:
            url += "?" + parse.urlencode(params)
        req = request.Request(
            url,
            headers={"User-Agent": "AutoPaper-citation-tracker/1.0"},
        )
        with self._counter_lock:
            self.api_call_count += 1
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            logger.warning("OpenAlex citation request failed: HTTP %s", exc.code)
        except error.URLError as exc:
            logger.warning("OpenAlex citation request failed: network error %s", exc.reason)
        except (TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.warning("OpenAlex citation request failed: %s", type(exc).__name__)
        return None


class CitationTracker:
    def __init__(
        self,
        *,
        embedding_model: Any | None = None,
        api_key: str | None = None,
        mailto: str | None = None,
        cache_path: str | Path | None = None,
        seed_min: int = 5,
        seed_max: int = 10,
        retain_min: int = 20,
        retain_max: int = 40,
        second_seed_max: int = 5,
        second_retain_max: int = 20,
        max_workers: int = 4,
        api_client: OpenAlexCitationClient | None = None,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.similarity = SimilarityCalculator()
        self.seed_min = max(1, int(seed_min))
        self.seed_max = max(self.seed_min, int(seed_max))
        self.retain_min = max(1, int(retain_min))
        self.retain_max = max(self.retain_min, int(retain_max))
        self.second_seed_max = max(1, int(second_seed_max))
        self.second_retain_max = max(1, int(second_retain_max))
        self.max_workers = max(1, int(max_workers))
        resolved_key = (api_key if api_key is not None else os.getenv("OPENALEX_API_KEY", ""))
        resolved_mailto = (
            mailto if mailto is not None else os.getenv("OPENALEX_MAILTO", "")
        )
        self.client = api_client or OpenAlexCitationClient(
            str(resolved_key or ""),
            mailto=str(resolved_mailto or ""),
            timeout_seconds=float(os.getenv("SEARCH_AGENT_TIMEOUT", "20")),
        )
        path = cache_path or os.getenv(
            "CITATION_CACHE_PATH",
            "./data/citation_graph.sqlite3",
        )
        self.cache = CitationCache(path)

    def expand(
        self,
        query: str,
        papers: Iterable[PaperMetadata],
        *,
        publication_date_to: str | None = None,
    ) -> CitationExpansion:
        initial = deduplicate_papers(enrich_paper_identifiers(paper) for paper in papers)
        summary = {
            "enabled": True,
            "initial_paper_count": len(initial),
            "seed_count": 0,
            "identifiable_seed_count": 0,
            "resolved_seed_count": 0,
            "raw_neighbor_count": 0,
            "unique_neighbor_count": 0,
            "retained_count": 0,
            "merged_paper_count": len(initial),
            "api_call_count": 0,
            "cache_hit_count": 0,
            "publication_date_to": publication_date_to,
            "seed_titles": [],
            "max_depth": 2,
            "first_layer": {},
            "second_layer": {},
        }
        if not initial:
            summary["skipped_reason"] = "no initial papers"
            return CitationExpansion(initial, summary)
        if not getattr(self.client, "api_key", ""):
            summary["skipped_reason"] = "OPENALEX_API_KEY is not configured"
            return CitationExpansion(initial, summary)

        query_vector = self.embedding_model.encode_query(query)
        ranked_initial = self._rank_semantic(query_vector, initial)
        seeds = self._identifiable_seeds(
            ranked_initial,
            self._seed_count(len(initial)),
        )
        summary["seed_count"] = len(seeds)
        summary["seed_titles"] = [paper.title for paper in seeds]
        summary["identifiable_seed_count"] = len(seeds)
        calls_before = self.client.api_call_count
        first, first_summary = self._expand_layer(
            query_vector,
            seeds,
            existing=initial,
            depth=1,
            publication_date_to=publication_date_to,
            retain_limit=self.retain_max,
        )
        second_seed_target = min(
            self.second_seed_max,
            max(1, math.ceil(len(first) * 0.20)),
        ) if first else 0
        second_seeds = self._identifiable_seeds(first, second_seed_target)
        second, second_summary = self._expand_layer(
            query_vector,
            second_seeds,
            existing=[*initial, *first],
            depth=2,
            publication_date_to=publication_date_to,
            retain_limit=self.second_retain_max,
        )
        merged = deduplicate_papers([*initial, *first, *second])
        summary["first_layer"] = first_summary
        summary["second_layer"] = second_summary
        summary["resolved_seed_count"] = (
            first_summary["resolved_seed_count"]
            + second_summary["resolved_seed_count"]
        )
        summary["raw_neighbor_count"] = (
            first_summary["raw_neighbor_count"]
            + second_summary["raw_neighbor_count"]
        )
        summary["unique_neighbor_count"] = (
            first_summary["unique_neighbor_count"]
            + second_summary["unique_neighbor_count"]
        )
        summary["cache_hit_count"] = (
            first_summary["cache_hit_count"]
            + second_summary["cache_hit_count"]
        )
        summary["retained_count"] = len(first) + len(second)
        summary["first_layer_retained_count"] = len(first)
        summary["second_layer_retained_count"] = len(second)
        summary["merged_paper_count"] = len(merged)
        summary["api_call_count"] = self.client.api_call_count - calls_before
        return CitationExpansion(merged, summary)

    def _expand_layer(
        self,
        query_vector: Any,
        seeds: list[PaperMetadata],
        *,
        existing: list[PaperMetadata],
        depth: int,
        publication_date_to: str | None,
        retain_limit: int,
    ) -> tuple[list[PaperMetadata], dict[str, int]]:
        layer_summary = {
            "seed_count": len(seeds),
            "resolved_seed_count": 0,
            "raw_neighbor_count": 0,
            "unique_neighbor_count": 0,
            "retained_count": 0,
            "cache_hit_count": 0,
        }
        if not seeds:
            return [], layer_summary

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(seeds))) as pool:
            resolved_values = list(pool.map(self._resolve_cached, seeds))
        resolved = [
            (paper, work)
            for paper, (work, _) in zip(seeds, resolved_values)
            if isinstance(work, dict) and work.get("id")
        ]
        layer_summary["cache_hit_count"] += sum(hit for _, hit in resolved_values)
        layer_summary["resolved_seed_count"] = len(resolved)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(resolved) or 1)) as pool:
            relation_values = list(pool.map(
                lambda item: self._neighbors_cached(item[1], publication_date_to),
                resolved,
            ))
        layer_summary["cache_hit_count"] += sum(hit for _, hit in relation_values)

        expanded: list[PaperMetadata] = []
        for (seed, work), (relations, _) in zip(resolved, relation_values):
            parent_key = canonical_paper_key(seed)
            seed_id = _short_openalex_id(work.get("id")) or parent_key or seed.title
            if not parent_key:
                parent_key = f"openalex:{str(seed_id).casefold()}"
            for direction, relation_key in (("reference", "references"), ("citation", "citations")):
                values = relations.get(relation_key, []) if isinstance(relations, dict) else []
                for value in values:
                    try:
                        paper = paper_from_openalex_work(value, source="citation_tracking")
                    except (TypeError, ValueError):
                        paper = None
                    if paper is None or not _within_cutoff(paper, publication_date_to):
                        continue
                    paper = enrich_paper_identifiers(paper).model_copy(update={
                        "citation_seed_ids": [seed_id],
                        "citation_directions": [direction],
                        "citation_parent_ids": [parent_key],
                        "citation_relations": [{
                            "parent_id": parent_key,
                            "direction": direction,
                            "depth": depth,
                        }],
                        "citation_depth": depth,
                    })
                    self._prime_resolve_cache(paper, value)
                    expanded.append(paper)

        layer_summary["raw_neighbor_count"] = len(expanded)
        unique = _deduplicate_by_identifiers(expanded)
        existing_keys = {key for paper in existing for key in _paper_keys(paper)}
        unique = [
            paper for paper in unique
            if not existing_keys.intersection(_paper_keys(paper))
        ]
        layer_summary["unique_neighbor_count"] = len(unique)
        keep_count = min(retain_limit, self._retain_count(len(unique)))
        retained = self._top_semantic_vector(query_vector, unique, keep_count)
        layer_summary["retained_count"] = len(retained)
        return retained, layer_summary

    def _resolve_cached(self, paper: PaperMetadata) -> tuple[Any, bool]:
        key = _seed_cache_key(paper)
        if not key:
            return None, False
        cache_key = f"resolve:{key}"
        hit, value = self.cache.get(cache_key)
        if hit:
            return value, True
        value = self.client.resolve(paper)
        if isinstance(value, dict) and value.get("id"):
            self.cache.set(cache_key, value)
        return value, False

    def _prime_resolve_cache(self, paper: PaperMetadata, work: Mapping[str, Any]) -> None:
        if not work.get("id"):
            return
        for key in _paper_keys(paper):
            if not key.startswith("title:"):
                self.cache.set(f"resolve:{key}", work)

    def _neighbors_cached(
        self,
        work: Mapping[str, Any],
        publication_date_to: str | None,
    ) -> tuple[Any, bool]:
        openalex_id = _short_openalex_id(work.get("id"))
        if not openalex_id:
            return {"references": [], "citations": []}, False
        cache_key = f"neighbors:{openalex_id}:{publication_date_to or '*'}"
        hit, value = self.cache.get(cache_key)
        if hit:
            return value, True
        value = self.client.neighbors(
            work,
            publication_date_to=publication_date_to,
        )
        if isinstance(value, dict):
            self.cache.set(cache_key, value)
            return value, False
        return {"references": [], "citations": []}, False

    def _top_semantic(
        self,
        query: str,
        papers: list[PaperMetadata],
        count: int,
    ) -> list[PaperMetadata]:
        if count <= 0 or not papers:
            return []
        query_vector = self.embedding_model.encode_query(query)
        return self._top_semantic_vector(query_vector, papers, count)

    def _top_semantic_vector(
        self,
        query_vector: Any,
        papers: list[PaperMetadata],
        count: int,
    ) -> list[PaperMetadata]:
        if count <= 0 or not papers:
            return []
        vectors = self.embedding_model.encode_papers(
            (paper.title, paper.abstract) for paper in papers
        )
        scores = self.similarity.calculate_batch(query_vector, vectors)
        ranked = sorted(
            zip(scores, papers),
            key=lambda item: (
                item[0],
                item[1].citation_count or 0,
                item[1].year or 0,
            ),
            reverse=True,
        )
        return [paper for _, paper in ranked[:count]]

    def _rank_semantic(
        self,
        query_vector: Any,
        papers: list[PaperMetadata],
    ) -> list[PaperMetadata]:
        return self._top_semantic_vector(query_vector, papers, len(papers))

    @staticmethod
    def _identifiable_seeds(
        ranked: list[PaperMetadata],
        count: int,
    ) -> list[PaperMetadata]:
        if count <= 0:
            return []
        return [paper for paper in ranked if _seed_cache_key(paper)][:count]

    def _seed_count(self, total: int) -> int:
        desired = max(self.seed_min, math.ceil(total * 0.10))
        return min(total, self.seed_max, desired)

    def _retain_count(self, total: int) -> int:
        if total <= 0:
            return 0
        desired = max(self.retain_min, math.ceil(total * 0.20))
        return min(total, self.retain_max, desired)


def _seed_cache_key(paper: PaperMetadata) -> str | None:
    enriched = enrich_paper_identifiers(paper)
    if enriched.openalex_id:
        openalex_id = _short_openalex_id(enriched.openalex_id)
        return f"openalex:{openalex_id.casefold()}" if openalex_id else None
    if enriched.doi:
        return f"doi:{_normalize_doi(enriched.doi)}"
    arxiv_id = _normalize_arxiv_id(enriched.arxiv_id)
    return f"arxiv:{arxiv_id}" if arxiv_id else None


def _paper_keys(paper: PaperMetadata) -> set[str]:
    return paper_identity_keys(paper)


def _deduplicate_by_identifiers(papers: Iterable[PaperMetadata]) -> list[PaperMetadata]:
    return list(deduplicate_papers(papers))


def _within_cutoff(paper: PaperMetadata, cutoff_value: str | None) -> bool:
    if not cutoff_value:
        return True
    try:
        cutoff = date.fromisoformat(cutoff_value)
    except ValueError:
        return True
    if paper.publication_date:
        try:
            return date.fromisoformat(paper.publication_date[:10]) <= cutoff
        except ValueError:
            pass
    return paper.year is None or paper.year <= cutoff.year


def _normalize_doi(value: Any) -> str:
    return normalize_doi(value) or ""


def _normalize_arxiv_id(value: Any) -> str | None:
    return normalize_arxiv_id(value)


def _short_openalex_id(value: Any) -> str | None:
    return normalize_openalex_id(value)


def _normalize_title(value: Any) -> str:
    return "".join(character.casefold() for character in str(value or "") if character.isalnum())


def _unique(values: Iterable[Any]) -> list[str]:
    return unique_strings(values)


def _unique_relations(values: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        item = {
            "parent_id": str(value.get("parent_id") or ""),
            "direction": str(value.get("direction") or ""),
            "depth": int(value.get("depth") or 0),
        }
        key = (item["parent_id"].casefold(), item["direction"].casefold(), item["depth"])
        if item["parent_id"] and key not in seen:
            seen.add(key)
            result.append(item)
    return result
