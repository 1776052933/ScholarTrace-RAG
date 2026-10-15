"""Set-retrieval metrics for LitSearch predictions."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Iterable, Sequence


def evaluate_query(
    gold_corpus_ids: Iterable[int],
    predicted_corpus_ids: Sequence[int | None],
    cutoffs: Sequence[int],
) -> dict[str, float]:
    """Calculate P/R/F1, Hit and reciprocal rank at each requested cutoff."""
    gold = {int(value) for value in gold_corpus_ids}
    predictions = list(predicted_corpus_ids)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        selected = predictions[:cutoff]
        hits = len({value for value in selected if value is not None and value in gold})
        # Missing result slots are non-relevant, so Precision@K always uses K.
        precision = hits / cutoff
        recall = hits / len(gold) if gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        reciprocal_rank = 0.0
        for rank, corpus_id in enumerate(selected, start=1):
            if corpus_id in gold:
                reciprocal_rank = 1.0 / rank
                break
        suffix = f"@{cutoff}"
        metrics[f"precision{suffix}"] = precision
        metrics[f"recall{suffix}"] = recall
        metrics[f"f1{suffix}"] = f1
        metrics[f"hit{suffix}"] = float(hits > 0)
        metrics[f"mrr{suffix}"] = reciprocal_rank
    return metrics


def aggregate_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Create macro averages and operational statistics for successful rows."""
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
    total_predictions = sum(int(record.get("prediction_count") or 0) for record in successful)
    resolved_predictions = sum(int(record.get("resolved_prediction_count") or 0) for record in successful)
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
        "prediction_resolution_rate": (
            resolved_predictions / total_predictions if total_predictions else 0.0
        ),
        "total_predictions": total_predictions,
        "resolved_predictions": resolved_predictions,
    }
