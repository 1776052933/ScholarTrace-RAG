"""ScholarGym benchmark loading and arXiv-ID evaluation helpers."""

from __future__ import annotations

import json
import calendar
from datetime import date
from pathlib import Path
import random
import re
import shutil
from typing import Any, Iterable, Sequence
from urllib.request import Request, urlopen


SCHOLARGYM_BENCH_URL = (
    "https://raw.githubusercontent.com/shenhao-stu/ScholarGym/"
    "main/data/scholargym_bench.jsonl"
)


class ScholarGymAccessError(RuntimeError):
    """Raised when the public ScholarGym benchmark cannot be downloaded."""


def ensure_scholargym_dataset(path: Path) -> Path:
    """Download the public query benchmark from GitHub when it is absent."""
    path = path.expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = Request(
        SCHOLARGYM_BENCH_URL,
        headers={"User-Agent": "AutoPaper-ScholarGym-Evaluation/1.0"},
    )
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise OSError("downloaded file is empty")
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ScholarGymAccessError(
            "Unable to download the public ScholarGym benchmark from GitHub. "
            "Download scholargym_bench.jsonl manually and pass its path with "
            "--dataset-path. Source: " + SCHOLARGYM_BENCH_URL
        ) from exc
    return path


def load_scholargym_queries(
    path: Path,
    *,
    source: str = "pasa_real",
    split: str | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load one ScholarGym source subset and preserve its benchmark IDs."""
    source_key = _source_key(source)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _source_key(value.get("source")) != source_key:
                continue
            if value.get("valid") is False:
                continue
            row_split = str(value.get("split") or "").strip().lower()
            if split and row_split != split.strip().lower():
                continue
            query = str(value.get("query") or "").strip()
            gold_papers = _gold_papers(value)
            gold_ids = [paper["arxiv_id"] for paper in gold_papers]
            if not query or not gold_ids:
                continue
            date_constraint = value.get("date_constraint") or value.get("date")
            rows.append({
                "query_id": str(
                    value.get("query_id") or value.get("qid") or f"scholargym_{index:04d}"
                ),
                "row_index": index,
                "query": query,
                "gold_arxiv_ids": gold_ids,
                "gold_papers": gold_papers,
                "source": str(value.get("source") or source_key),
                "split": row_split or None,
                "date_constraint": date_constraint,
                "publication_date_to": scholar_gym_cutoff(date_constraint),
            })
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows


def normalize_arxiv_id(value: Any) -> str | None:
    """Return an arXiv identifier without URL prefixes or version suffixes."""
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"(?i)^arxiv:\s*", "", text)
    text = re.sub(r"(?i)^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", "", text)
    text = re.sub(r"(?i)\.pdf(?:[?#].*)?$", "", text)
    text = text.split("?", 1)[0].split("#", 1)[0].strip(" /.")
    match = re.search(
        r"(?i)(\d{4}\.\d{4,5}|[a-z][a-z0-9.-]+/\d{7})(?:v\d+)?$",
        text,
    )
    return match.group(1).lower() if match else None


def paper_arxiv_id(paper: dict[str, Any]) -> str | None:
    """Extract an arXiv ID from common AutoPaper metadata fields."""
    candidates: list[Any] = [paper.get("arxiv_id")]
    external_ids = paper.get("external_ids") or paper.get("externalIds")
    if isinstance(external_ids, dict):
        candidates.extend([
            external_ids.get("ArXiv"),
            external_ids.get("arxiv"),
            external_ids.get("ARXIV"),
        ])
    for field in ("url", "pdf_url", "paper_id", "source_id"):
        candidates.append(paper.get(field))
    for candidate in candidates:
        normalized = normalize_arxiv_id(candidate)
        if normalized:
            return normalized
    return None


def benchmark_paper_arxiv_id(
    paper: dict[str, Any],
    gold_papers: Iterable[dict[str, Any]],
) -> str | None:
    """Resolve an ID directly, then use strict gold-title identity as fallback."""
    direct_id = paper_arxiv_id(paper)
    if direct_id:
        return direct_id
    title = normalize_scholargym_title(paper.get("title"))
    if not title:
        return None
    title_matches = {
        str(gold.get("arxiv_id"))
        for gold in gold_papers
        if normalize_scholargym_title(gold.get("title")) == title
    }
    return next(iter(title_matches)) if len(title_matches) == 1 else None


def normalize_scholargym_title(value: Any) -> str:
    return "".join(character for character in str(value or "") if character.isalnum()).lower()


def deduplicate_arxiv_ids(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_arxiv_id(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def scholar_gym_cutoff(value: Any) -> str | None:
    """Convert ScholarGym's year/month/date constraint to an inclusive ISO cutoff."""
    text = str(value or "").strip()
    full = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if full:
        try:
            return date(*(int(part) for part in full.groups())).isoformat()
        except ValueError:
            return None
    month = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if month:
        year, month_number = (int(part) for part in month.groups())
        try:
            last_day = calendar.monthrange(year, month_number)[1]
            return date(year, month_number, last_day).isoformat()
        except (ValueError, calendar.IllegalMonthError):
            return None
    year = re.fullmatch(r"\d{4}", text)
    return f"{text}-12-31" if year else None


def _gold_papers(value: dict[str, Any]) -> list[dict[str, str]]:
    direct_ids = value.get("gt_arxiv_ids")
    if isinstance(direct_ids, list):
        return [
            {"arxiv_id": arxiv_id, "title": ""}
            for arxiv_id in deduplicate_arxiv_ids(direct_ids)
        ]
    papers = value.get("cited_paper")
    if not isinstance(papers, list):
        return []
    labels = value.get("gt_label")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, paper in enumerate(papers):
        if isinstance(labels, list) and index < len(labels) and not bool(labels[index]):
            continue
        if isinstance(paper, dict):
            arxiv_id = normalize_arxiv_id(paper.get("arxiv_id"))
            title = str(paper.get("title") or "").strip()
        else:
            arxiv_id = normalize_arxiv_id(paper)
            title = ""
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            result.append({"arxiv_id": arxiv_id, "title": title})
    return result


def evaluate_scholargym_query(
    gold_arxiv_ids: Iterable[Any],
    candidate_ids: Iterable[str | None],
    ranked_ids: Sequence[str | None],
    cutoffs: Sequence[int],
    *,
    selected_ids: Iterable[str | None] | None = None,
) -> dict[str, Any]:
    """Calculate crawler, selector, and ranked retrieval metrics by arXiv ID."""
    gold = set(deduplicate_arxiv_ids(gold_arxiv_ids))
    candidates = {value for value in candidate_ids if value}
    ranked = list(ranked_ids)
    candidate_hits = len(candidates & gold)
    selected = {value for value in (selected_ids or []) if value}
    selected_hits = len(selected & gold)
    selector_precision = selected_hits / len(selected) if selected else 0.0
    selector_recall = selected_hits / len(gold) if gold else 0.0
    metrics: dict[str, float] = {
        "crawler_recall": candidate_hits / len(gold) if gold else 0.0,
        "selector_precision": selector_precision,
        "selector_recall": selector_recall,
        "selector_f1": (
            2 * selector_precision * selector_recall / (selector_precision + selector_recall)
            if selector_precision + selector_recall
            else 0.0
        ),
    }
    for cutoff in cutoffs:
        top_ids = {value for value in ranked[:cutoff] if value}
        hits = len(top_ids & gold)
        precision = hits / cutoff
        recall = hits / len(gold) if gold else 0.0
        metrics[f"precision@{cutoff}"] = precision
        metrics[f"recall@{cutoff}"] = recall
        metrics[f"f1@{cutoff}"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[f"hit@{cutoff}"] = float(hits > 0)
    gold_ranks = [
        rank for rank, arxiv_id in enumerate(ranked, start=1) if arxiv_id in gold
    ]
    return {
        "metrics": metrics,
        "gold_ranks": gold_ranks,
        "best_gold_rank": min(gold_ranks) if gold_ranks else None,
        "matched_gold_count": len({value for value in ranked if value} & gold),
        "candidate_matched_gold_count": candidate_hits,
        "selector_matched_gold_count": selected_hits,
    }


def _source_key(value: Any) -> str:
    key = re.sub(r"[^a-z]", "", str(value or "").lower())
    aliases = {
        "pasareal": "pasareal",
        "pasarealscholar": "pasareal",
        "pasaauto": "pasaauto",
        "pasaautoscholar": "pasaauto",
    }
    return aliases.get(key, key)
