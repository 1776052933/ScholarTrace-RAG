"""Run the unchanged AutoPaper workflow on LitSearch queries."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.litsearch import ensure_query_dataset, load_queries
from evaluation.metrics import aggregate_results, evaluate_query
from evaluation.semantic_scholar import SemanticScholarResolver


LOGGER = logging.getLogger("autopaper.evaluation")
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "data" / "litsearch_query.parquet"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "evaluation" / "runs"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cutoffs = _parse_cutoffs(args.cutoffs)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    if args.offset < 0:
        raise SystemExit("--offset must not be negative")
    if args.top_k < 1 or args.max_search_rounds < 1:
        raise SystemExit("--top-k and --max-search-rounds must be positive")
    if args.max_title_lookups < 0:
        raise SystemExit("--max-title-lookups must not be negative")
    if args.top_k < max(cutoffs):
        raise SystemExit("--top-k must be greater than or equal to every cutoff")

    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else DEFAULT_RUNS_ROOT / datetime.now().strftime("litsearch_%Y%m%d_%H%M%S")
    )
    records_dir = run_dir / "records"
    raw_dir = run_dir / "raw_results"
    records_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "benchmark.log", args.log_level)

    dataset_path = ensure_query_dataset(Path(args.dataset_path))
    queries = load_queries(
        dataset_path,
        query_set=args.query_set,
        minimum_quality=args.minimum_quality,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    selected = queries[args.offset:]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise SystemExit("No LitSearch queries matched the requested selection")

    os.environ["RANKING_CACHE_PATH"] = str(run_dir / "ranking_evaluations.sqlite3")
    if args.search_agent_llm:
        os.environ["SEARCH_AGENT_USE_LLM"] = "true"
    from main import AutoPaperWorkflow

    workflow = AutoPaperWorkflow(
        use_query_llm=not args.no_query_llm,
        max_search_rounds=args.max_search_rounds,
        database_url=f"sqlite:///{(run_dir / 'literature.db').resolve().as_posix()}",
        enable_rag=False,
    )
    resolver = SemanticScholarResolver(run_dir / "semantic_scholar_cache.json")
    manifest = {
        "benchmark": "LitSearch",
        "evaluation_mode": "official queries with live AutoPaper search backends",
        "official_fixed_corpus_used": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path.resolve()),
        "query_count": len(selected),
        "top_k": args.top_k,
        "cutoffs": cutoffs,
        "max_search_rounds": args.max_search_rounds,
        "query_llm_enabled": not args.no_query_llm,
        "search_agent_llm_enabled": bool(args.search_agent_llm),
        "title_resolution_enabled": not args.no_title_resolution,
        "max_title_lookups": args.max_title_lookups,
        "selection": {
            "offset": args.offset,
            "limit": args.limit,
            "query_set": args.query_set,
            "minimum_quality": args.minimum_quality,
            "shuffle": args.shuffle,
            "seed": args.seed,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    LOGGER.info("LitSearch evaluation started: queries=%d, run_dir=%s", len(selected), run_dir)
    for position, item in enumerate(selected, start=1):
        record_path = records_dir / f"{item['query_id']}.json"
        if args.resume and record_path.is_file():
            LOGGER.info("[%d/%d] Resume skip: %s", position, len(selected), item["query_id"])
            continue
        LOGGER.info("[%d/%d] Query: %s", position, len(selected), item["query"])
        record = _evaluate_one(
            workflow,
            resolver,
            item,
            raw_dir=raw_dir,
            top_k=args.top_k,
            cutoffs=cutoffs,
            allow_title_matching=not args.no_title_resolution,
            max_title_lookups=args.max_title_lookups,
        )
        _write_json(record_path, record)
        records = _load_records(records_dir)
        _write_reports(run_dir, records, manifest)
        if record["status"] == "completed":
            metrics = record["metrics"]
            LOGGER.info(
                "[%d/%d] Completed: recall@%d=%.3f f1@%d=%.3f time=%.1fs resolved=%d/%d",
                position,
                len(selected),
                max(cutoffs),
                metrics[f"recall@{max(cutoffs)}"],
                max(cutoffs),
                metrics[f"f1@{max(cutoffs)}"],
                record["workflow_seconds"],
                record["resolved_prediction_count"],
                record["prediction_count"],
            )
        else:
            LOGGER.error("[%d/%d] Failed: %s", position, len(selected), record["error"])

    records = _load_records(records_dir)
    summary = _write_reports(run_dir, records, manifest)
    print_summary(summary, run_dir, cutoffs)
    return 0 if summary["successful_query_count"] else 1


def _evaluate_one(
    workflow: Any,
    resolver: SemanticScholarResolver,
    item: dict[str, Any],
    *,
    raw_dir: Path,
    top_k: int,
    cutoffs: list[int],
    allow_title_matching: bool,
    max_title_lookups: int,
) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = workflow.run(
            item["query"],
            max_papers=top_k,
            download_papers=False,
            ingest_rag=False,
        )
        workflow_seconds = perf_counter() - started
        raw_path = raw_dir / f"{item['query_id']}.json"
        _write_json(raw_path, result)
        ranked = (
            result.get("stages", {})
            .get("ranking_agent", {})
            .get("result", {})
            .get("papers", [])
        )
        ranked_candidates = [dict(paper) for paper in ranked if isinstance(paper, dict)]
        predictions = ranked_candidates[:top_k]
        resolver_calls_before = resolver.api_call_count
        resolution_started = perf_counter()
        ranked_resolutions = resolver.resolve_many(
            ranked_candidates,
            allow_title_matching=allow_title_matching,
            max_title_lookups=max_title_lookups,
        )
        resolution_seconds = perf_counter() - resolution_started
        resolved_rows = []
        corpus_ids: list[int | None] = []
        for rank, (paper, resolution) in enumerate(
            zip(predictions, ranked_resolutions[:top_k]),
            start=1,
        ):
            corpus_id = resolution.get("corpus_id")
            corpus_ids.append(int(corpus_id) if corpus_id is not None else None)
            resolved_rows.append({
                "rank": rank,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "url": paper.get("url"),
                "corpus_id": corpus_id,
                "resolution_method": resolution.get("method"),
                "resolution_confidence": resolution.get("confidence"),
                "matched_title": resolution.get("matched_title"),
            })
        gold = {int(value) for value in item["gold_corpus_ids"]}
        ranked_corpus_ids = [
            int(resolution["corpus_id"])
            if resolution.get("corpus_id") is not None
            else None
            for resolution in ranked_resolutions
        ]
        gold_ranks = [
            rank
            for rank, corpus_id in enumerate(ranked_corpus_ids, start=1)
            if corpus_id in gold
        ]
        candidate_hits = len({value for value in ranked_corpus_ids if value in gold})
        search_stage = result.get("stages", {}).get("search_agent", {}).get("result", {})
        ranking_stage = result.get("stages", {}).get("ranking_agent", {}).get("result", {})
        return {
            **item,
            "status": "completed",
            "workflow_seconds": round(workflow_seconds, 6),
            "reported_workflow_seconds": float(result.get("elapsed_seconds") or 0.0),
            "resolution_seconds": round(resolution_seconds, 6),
            "resolution_api_calls": resolver.api_call_count - resolver_calls_before,
            "search_paper_count": len(search_stage.get("papers", []) or []),
            "search_rounds_completed": int(search_stage.get("rounds_completed") or 0),
            "ranked_paper_count": int(ranking_stage.get("total") or len(ranked)),
            "llm_evaluated_count": int(ranking_stage.get("llm_evaluated_count") or 0),
            "prediction_count": len(predictions),
            "resolved_prediction_count": sum(value is not None for value in corpus_ids),
            "ranked_resolved_count": sum(value is not None for value in ranked_corpus_ids),
            "candidate_recall": candidate_hits / len(gold) if gold else 0.0,
            "best_gold_rank": min(gold_ranks) if gold_ranks else None,
            "gold_ranks": gold_ranks,
            "metrics": evaluate_query(item["gold_corpus_ids"], corpus_ids, cutoffs),
            "predictions": resolved_rows,
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
    summary = {**manifest, **aggregate_results(records)}
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(run_dir / "summary.json", summary)
    metric_names = sorted(summary.get("macro_metrics", {}))
    fieldnames = [
        "query_id", "query_set", "query", "status", "gold_count",
        "search_paper_count", "ranked_paper_count", "prediction_count",
        "resolved_prediction_count", "search_rounds_completed",
        "ranked_resolved_count", "candidate_recall", "best_gold_rank", "gold_ranks",
        "llm_evaluated_count", "workflow_seconds", "resolution_seconds",
        *metric_names, "error",
    ]
    with (run_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "query_id": record.get("query_id"),
                "query_set": record.get("query_set"),
                "query": record.get("query"),
                "status": record.get("status"),
                "gold_count": len(record.get("gold_corpus_ids") or []),
                "search_paper_count": record.get("search_paper_count"),
                "ranked_paper_count": record.get("ranked_paper_count"),
                "prediction_count": record.get("prediction_count"),
                "resolved_prediction_count": record.get("resolved_prediction_count"),
                "ranked_resolved_count": record.get("ranked_resolved_count"),
                "candidate_recall": record.get("candidate_recall"),
                "best_gold_rank": record.get("best_gold_rank"),
                "gold_ranks": ";".join(str(value) for value in record.get("gold_ranks") or []),
                "search_rounds_completed": record.get("search_rounds_completed"),
                "llm_evaluated_count": record.get("llm_evaluated_count"),
                "workflow_seconds": record.get("workflow_seconds"),
                "resolution_seconds": record.get("resolution_seconds"),
                "error": record.get("error"),
            }
            row.update(record.get("metrics") or {})
            writer.writerow(row)
    return summary


def print_summary(summary: dict[str, Any], run_dir: Path, cutoffs: list[int]) -> None:
    print("\nLitSearch 初步测评完成")
    print(f"运行目录: {run_dir.resolve()}")
    print(
        f"成功/总数: {summary['successful_query_count']}/{summary['query_count']} | "
        f"平均工作流耗时: {summary['average_workflow_seconds']:.1f} 秒 | "
        f"论文身份解析率: {summary['prediction_resolution_rate']:.1%}"
    )
    macro = summary.get("macro_metrics", {})
    for cutoff in cutoffs:
        print(
            f"@{cutoff}: Precision={macro.get(f'precision@{cutoff}', 0.0):.4f} | "
            f"Recall={macro.get(f'recall@{cutoff}', 0.0):.4f} | "
            f"F1={macro.get(f'f1@{cutoff}', 0.0):.4f} | "
            f"Hit={macro.get(f'hit@{cutoff}', 0.0):.4f} | "
            f"MRR={macro.get(f'mrr@{cutoff}', 0.0):.4f}"
        )
    print("说明: 这是官方查询集上的联网检索结果，不是 LitSearch 固定语料官方榜单成绩。")


def _load_records(records_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(records_dir.glob("litsearch_*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_cutoffs(value: str) -> list[int]:
    try:
        cutoffs = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cutoffs must be comma-separated integers") from exc
    if not cutoffs or cutoffs[0] < 1:
        raise argparse.ArgumentTypeError("cutoffs must contain positive integers")
    return cutoffs


def _configure_logging(path: Path, level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(path, encoding="utf-8"),
        ],
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the unchanged AutoPaper workflow on LitSearch queries.",
    )
    parser.add_argument("--limit", type=int, default=1, help="查询数量，默认先测 1 条")
    parser.add_argument("--offset", type=int, default=0, help="从筛选结果的第几条开始")
    parser.add_argument("--top-k", type=int, default=20, help="保留多少篇精排论文")
    parser.add_argument("--cutoffs", default="5,10,20", help="指标截断，例如 5,10,20")
    parser.add_argument("--max-search-rounds", type=int, default=1)
    parser.add_argument(
        "--query-set",
        choices=["inline_acl", "inline_nonacl", "manual_acl", "manual_iclr"],
        default=None,
    )
    parser.add_argument("--minimum-quality", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true", help="按固定种子打乱查询")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-query-llm", action="store_true")
    parser.add_argument(
        "--search-agent-llm",
        action="store_true",
        help="显式启用搜索规划/评价/反思中的 DeepSeek；默认保持项目当前配置",
    )
    parser.add_argument("--no-title-resolution", action="store_true")
    parser.add_argument("--max-title-lookups", type=int, default=10)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--run-dir", default=None, help="指定目录可配合 --resume 断点续跑")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
