"""Evaluate the unchanged AutoPaper workflow on a ScholarGym subset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from time import perf_counter
import sys
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.pasa_metrics import aggregate_pasa_results
from evaluation.scholargym import (
    ScholarGymAccessError,
    benchmark_paper_arxiv_id,
    ensure_scholargym_dataset,
    evaluate_scholargym_query,
    load_scholargym_queries,
    paper_arxiv_id,
)


LOGGER = logging.getLogger("autopaper.evaluation.scholargym")
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "data" / "scholargym" / "scholargym_bench.jsonl"
)
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "evaluation" / "runs"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoffs = _parse_cutoffs(args.cutoffs)
    _validate_args(args, cutoffs)
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else DEFAULT_RUNS_ROOT / datetime.now().strftime("scholargym_pasa_real_%Y%m%d_%H%M%S")
    )
    records_dir = run_dir / "records"
    raw_dir = run_dir / "raw_results"
    records_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "benchmark.log", args.log_level)

    try:
        dataset_path = ensure_scholargym_dataset(Path(args.dataset_path))
    except ScholarGymAccessError as exc:
        raise SystemExit(str(exc)) from exc
    queries = load_scholargym_queries(
        dataset_path,
        source=args.source,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    selected = queries[args.offset:]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit(
            f"No ScholarGym rows matched source={args.source!r}, split={args.split!r}"
        )

    os.environ["RANKING_CACHE_PATH"] = str(run_dir / "ranking_evaluations.sqlite3")
    if args.search_agent_llm:
        os.environ["SEARCH_AGENT_USE_LLM"] = "true"
    from main import AutoPaperWorkflow

    workflow = AutoPaperWorkflow(
        use_query_llm=not args.no_query_llm,
        max_search_rounds=args.max_search_rounds,
        database_url=f"sqlite:///{(run_dir / 'literature.db').resolve().as_posix()}",
        enable_rag=False,
        enable_citation_tracking=not args.skip_citation_tracking,
    )
    manifest = {
        "benchmark": "ScholarGym PaSa-RealScholar subset",
        "dataset_source": "shenhao-stu/ScholarGym/data/scholargym_bench.jsonl",
        "evaluation_mode": "live AutoPaper backends matched against ScholarGym arXiv IDs",
        "comparison_scope": (
            "Project-internal live-search evaluation; not directly comparable with ScholarGym "
            "static-corpus results or the original PaSa benchmark score."
        ),
        "temporal_cutoff_enforced": not args.no_temporal_cutoff,
        "temporal_cutoff_note": (
            "ScholarGym date constraints are applied before ranking. Month-only constraints "
            "use the inclusive final day of that month."
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path.resolve()),
        "expected_query_count": len(selected),
        "source": args.source,
        "split": args.split,
        "top_k": args.top_k,
        "cutoffs": cutoffs,
        "max_search_rounds": args.max_search_rounds,
        "query_llm_enabled": not args.no_query_llm,
        "search_agent_llm_enabled": bool(args.search_agent_llm),
        "citation_tracking_enabled": not args.skip_citation_tracking,
        "citation_tracking_max_depth": 0 if args.skip_citation_tracking else 2,
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    LOGGER.info("ScholarGym evaluation started: queries=%d, run_dir=%s", len(selected), run_dir)
    for position, item in enumerate(selected, start=1):
        record_path = records_dir / f"{item['query_id']}.json"
        if args.resume and record_path.is_file():
            LOGGER.info("[%d/%d] Resume skip: %s", position, len(selected), item["query_id"])
            continue
        LOGGER.info("[%d/%d] Query: %s", position, len(selected), item["query"])
        record = _evaluate_one(
            workflow,
            item,
            raw_dir=raw_dir,
            top_k=args.top_k,
            cutoffs=cutoffs,
            publication_date_to=(
                None if args.no_temporal_cutoff else item.get("publication_date_to")
            ),
        )
        _write_json(record_path, record)
        _write_reports(run_dir, _load_records(records_dir), manifest)
        if record["status"] == "completed":
            LOGGER.info(
                "[%d/%d] Completed: crawler_recall=%.3f recall@%d=%.3f time=%.1fs",
                position,
                len(selected),
                record["metrics"]["crawler_recall"],
                max(cutoffs),
                record["metrics"][f"recall@{max(cutoffs)}"],
                record["workflow_seconds"],
            )
        else:
            LOGGER.error("[%d/%d] Failed: %s", position, len(selected), record["error"])

    summary = _write_reports(run_dir, _load_records(records_dir), manifest)
    _print_summary(summary, run_dir, cutoffs)
    return 0 if summary["successful_query_count"] else 1


def _evaluate_one(
    workflow: Any,
    item: dict[str, Any],
    *,
    raw_dir: Path,
    top_k: int,
    cutoffs: list[int],
    publication_date_to: str | None,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = workflow.run(
            item["query"],
            max_papers=top_k,
            download_papers=False,
            ingest_rag=False,
            publication_date_to=publication_date_to,
        )
        workflow_seconds = perf_counter() - started
        raw_path = raw_dir / f"{item['query_id']}.json"
        _write_json(raw_path, result)
        stages = result.get("stages", {})
        search_stage = stages.get("search_agent", {}).get("result", {})
        ranking_stage = stages.get("ranking_agent", {}).get("result", {})
        search_papers = [
            paper for paper in search_stage.get("papers", []) or [] if isinstance(paper, dict)
        ]
        ranked_papers = [
            paper for paper in ranking_stage.get("papers", []) or [] if isinstance(paper, dict)
        ]
        highly_relevant_count = int(ranking_stage.get("highly_relevant_count") or 0)
        search_ids = [
            benchmark_paper_arxiv_id(paper, item["gold_papers"])
            for paper in search_papers
        ]
        ranked_ids = [
            benchmark_paper_arxiv_id(paper, item["gold_papers"])
            for paper in ranked_papers
        ]
        diagnostic = evaluate_scholargym_query(
            item["gold_arxiv_ids"],
            search_ids,
            ranked_ids,
            cutoffs,
            selected_ids=ranked_ids[:highly_relevant_count],
        )
        gold = set(item["gold_arxiv_ids"])
        predictions = []
        for rank, paper in enumerate(ranked_papers[:top_k], start=1):
            arxiv_id = paper_arxiv_id(paper)
            evaluation_arxiv_id = benchmark_paper_arxiv_id(paper, item["gold_papers"])
            predictions.append({
                "rank": rank,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "url": paper.get("url"),
                "doi": paper.get("doi"),
                "arxiv_id": arxiv_id,
                "evaluation_arxiv_id": evaluation_arxiv_id,
                "identifier_match": (
                    "arxiv_id" if arxiv_id else "exact_gold_title" if evaluation_arxiv_id else None
                ),
                "source": paper.get("source"),
                "is_gold": evaluation_arxiv_id in gold if evaluation_arxiv_id else False,
            })
        return {
            **item,
            "status": "completed",
            "workflow_seconds": round(workflow_seconds, 6),
            "reported_workflow_seconds": float(result.get("elapsed_seconds") or 0.0),
            "search_paper_count": len(search_papers),
            "search_rounds_completed": int(search_stage.get("rounds_completed") or 0),
            "ranked_paper_count": int(ranking_stage.get("total") or len(ranked_papers)),
            "llm_evaluated_count": int(ranking_stage.get("llm_evaluated_count") or 0),
            "prediction_count": len(predictions),
            "resolved_arxiv_id_count": sum(
                paper_arxiv_id(paper) is not None for paper in ranked_papers[:top_k]
            ),
            "selected_paper_count": highly_relevant_count,
            "metrics": diagnostic["metrics"],
            "gold_ranks": diagnostic["gold_ranks"],
            "best_gold_rank": diagnostic["best_gold_rank"],
            "matched_gold_count": diagnostic["matched_gold_count"],
            "candidate_matched_gold_count": diagnostic["candidate_matched_gold_count"],
            "selector_matched_gold_count": diagnostic["selector_matched_gold_count"],
            "predictions": predictions,
            "raw_result_path": str(raw_path.resolve()),
        }
    except Exception as exc:
        return {
            **item,
            "status": "failed",
            "workflow_seconds": round(perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _write_reports(
    run_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    summary = {**manifest, **aggregate_pasa_results(records)}
    completed = [record for record in records if record.get("status") == "completed"]
    prediction_count = sum(int(record.get("prediction_count") or 0) for record in completed)
    resolved_count = sum(int(record.get("resolved_arxiv_id_count") or 0) for record in completed)
    summary["arxiv_id_resolution_rate"] = (
        resolved_count / prediction_count if prediction_count else 0.0
    )
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(run_dir / "summary.json", summary)
    metric_names = sorted(summary.get("macro_metrics", {}))
    fieldnames = [
        "query_id", "query", "status", "gold_count", "search_paper_count",
        "ranked_paper_count", "prediction_count", "resolved_arxiv_id_count",
        "matched_gold_count", "candidate_matched_gold_count", "best_gold_rank",
        "gold_ranks", "workflow_seconds", *metric_names, "error",
    ]
    with (run_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "query_id": record.get("query_id"),
                "query": record.get("query"),
                "status": record.get("status"),
                "gold_count": len(record.get("gold_arxiv_ids") or []),
                "search_paper_count": record.get("search_paper_count"),
                "ranked_paper_count": record.get("ranked_paper_count"),
                "prediction_count": record.get("prediction_count"),
                "resolved_arxiv_id_count": record.get("resolved_arxiv_id_count"),
                "matched_gold_count": record.get("matched_gold_count"),
                "candidate_matched_gold_count": record.get("candidate_matched_gold_count"),
                "best_gold_rank": record.get("best_gold_rank"),
                "gold_ranks": ";".join(str(value) for value in record.get("gold_ranks") or []),
                "workflow_seconds": record.get("workflow_seconds"),
                "error": record.get("error"),
            }
            row.update(record.get("metrics") or {})
            writer.writerow(row)
    return summary


def _print_summary(summary: dict[str, Any], run_dir: Path, cutoffs: list[int]) -> None:
    macro = summary.get("macro_metrics", {})
    print("\nScholarGym PaSa-RealScholar evaluation completed")
    print(f"Run directory: {run_dir.resolve()}")
    print(
        f"Successful/total: {summary['successful_query_count']}/{summary['query_count']} | "
        f"Average workflow time: {summary['average_workflow_seconds']:.1f}s | "
        f"Crawler Recall: {macro.get('crawler_recall', 0.0):.4f} | "
        f"arXiv ID resolution: {summary.get('arxiv_id_resolution_rate', 0.0):.2%}"
    )
    for cutoff in cutoffs:
        print(
            f"@{cutoff}: Precision={macro.get(f'precision@{cutoff}', 0.0):.4f} | "
            f"Recall={macro.get(f'recall@{cutoff}', 0.0):.4f} | "
            f"F1={macro.get(f'f1@{cutoff}', 0.0):.4f} | "
            f"Hit={macro.get(f'hit@{cutoff}', 0.0):.4f}"
        )
    print("Note: this uses live search backends, not ScholarGym's static 570K-paper corpus.")


def _load_records(records_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parse_cutoffs(value: str) -> list[int]:
    try:
        cutoffs = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoffs must be comma-separated integers") from exc
    if not cutoffs or cutoffs[0] < 1:
        raise argparse.ArgumentTypeError("cutoffs must contain positive integers")
    return cutoffs


def _validate_args(args: argparse.Namespace, cutoffs: list[int]) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    if args.offset < 0:
        raise SystemExit("--offset must not be negative")
    if args.top_k < max(cutoffs):
        raise SystemExit("--top-k must be greater than or equal to every cutoff")
    if args.max_search_rounds < 1:
        raise SystemExit("--max-search-rounds must be positive")


def _configure_logging(path: Path, level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AutoPaper on the public ScholarGym PaSa-RealScholar subset.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of queries to evaluate")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--cutoffs", default="20,50,100")
    parser.add_argument("--max-search-rounds", type=int, default=1)
    parser.add_argument("--source", default="pasa_real")
    parser.add_argument("--split", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-query-llm", action="store_true")
    parser.add_argument("--search-agent-llm", action="store_true")
    parser.add_argument(
        "--skip-citation-tracking",
        action="store_true",
        help="Disable two-hop OpenAlex citation expansion for an A/B comparison",
    )
    parser.add_argument(
        "--no-temporal-cutoff",
        action="store_true",
        help="Disable benchmark publication-date filtering for an A/B comparison",
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
