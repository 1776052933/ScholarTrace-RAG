"""Official-compatible title matching and retrieval metrics for PaSa."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable, Sequence


def normalize_pasa_title(value: str) -> str:
    """Match PaSa's keep_letters: retain Unicode letters and lowercase."""
    return "".join(character for character in str(value) if character.isalpha()).lower()


def deduplicate_titles(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = normalize_pasa_title(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def evaluate_pasa_query(
    gold_titles: Iterable[str],
    candidate_titles: Iterable[str],
    ranked_titles: Iterable[str],
    cutoffs: Sequence[int],
    selected_titles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate PaSa crawler recall plus ranked Precision/Recall/F1 at K."""
    gold = set(deduplicate_titles(gold_titles))
    candidates = set(deduplicate_titles(candidate_titles))
    ranked = deduplicate_titles(ranked_titles)
    candidate_hits = len(candidates & gold)
    metrics: dict[str, float] = {
        "crawler_recall": candidate_hits / len(gold) if gold else 0.0,
    }
    selected = set(deduplicate_titles(selected_titles or []))
    selected_hits = len(selected & gold)
    selector_precision = selected_hits / len(selected) if selected else 0.0
    selector_recall = selected_hits / len(gold) if gold else 0.0
    metrics["selector_precision"] = selector_precision
    metrics["selector_recall"] = selector_recall
    metrics["selector_f1"] = (
        2 * selector_precision * selector_recall / (selector_precision + selector_recall)
        if selector_precision + selector_recall
        else 0.0
    )
    for cutoff in cutoffs:
        selected = set(ranked[:cutoff])
        true_positive = len(selected & gold)
        precision = true_positive / len(selected) if selected else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[f"precision@{cutoff}"] = precision
        metrics[f"recall@{cutoff}"] = recall
        metrics[f"f1@{cutoff}"] = f1
        metrics[f"hit@{cutoff}"] = float(true_positive > 0)
    gold_ranks = [
        rank
        for rank, title in enumerate(ranked, start=1)
        if title in gold
    ]
    return {
        "metrics": metrics,
        "gold_ranks": gold_ranks,
        "best_gold_rank": min(gold_ranks) if gold_ranks else None,
        "matched_gold_count": len(set(ranked) & gold),
        "candidate_matched_gold_count": candidate_hits,
        "selector_matched_gold_count": selected_hits,
    }


def aggregate_pasa_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "completed"]
    metric_names = sorted({
        name
        for record in successful
        for name in (record.get("metrics") or {})
    })
    macro = {
        name: fmean(float(record["metrics"].get(name, 0.0)) for record in successful)
        for name in metric_names
    } if successful else {}
    return {
        "query_count": len(records),
        "successful_query_count": len(successful),
        "failed_query_count": len(records) - len(successful),
        "macro_metrics": macro,
        "average_workflow_seconds": (
            fmean(float(record.get("workflow_seconds") or 0.0) for record in successful)
            if successful else 0.0
        ),
        "total_workflow_seconds": sum(
            float(record.get("workflow_seconds") or 0.0) for record in successful
        ),
        "average_search_paper_count": (
            fmean(float(record.get("search_paper_count") or 0.0) for record in successful)
            if successful else 0.0
        ),
        "average_ranked_paper_count": (
            fmean(float(record.get("ranked_paper_count") or 0.0) for record in successful)
            if successful else 0.0
        ),
    }
