"""Command-line entry point for the reusable ScholarTrace research workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional, Sequence

from rag_reader.errors import error_info

from app.services.workflow import (
    AutoPaperWorkflow,
    _configure_project_api_keys,
    _elapsed,
    _json_value,
    _merge_upstream_subqueries,
    _model_data,
    _normalized_query,
    _optional_iso_date,
    _replace_papers,
    _required_text,
    _run_rag_tasks,
    _stage,
    _subquery_type,
    _timed,
    _understanding_for_decomposition,
    _value,
)

logger = logging.getLogger("auto_paper")

def build_workflow(args: argparse.Namespace) -> AutoPaperWorkflow:
    """根据命令行参数创建默认工作流。"""
    return AutoPaperWorkflow(
        use_query_llm=not args.no_query_llm,
        max_search_rounds=args.max_search_rounds,
        database_url=args.database_url,
        chroma_directory=args.chroma_directory,
        enable_rag=not args.skip_download and not args.skip_rag,
        enable_citation_tracking=not args.skip_citation_tracking,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口；成功返回 0，失败返回 1。"""
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    query = " ".join(args.query).strip()
    try:
        workflow = build_workflow(args)
        result = workflow.run(
            query,
            max_papers=args.max_papers,
            download_papers=not args.skip_download,
            ingest_rag=not args.skip_download and not args.skip_rag,
            papers_directory=args.papers_directory,
            text_directory=args.text_directory,
            download_engine=args.download_engine,
            download_timeout=args.download_timeout,
            retry_failed_downloads=args.retry_failed_downloads,
            question=args.question,
            summarize=args.summarize,
            extract_knowledge=args.extract_knowledge,
            publication_date_to=args.publication_date_to,
        )
        content = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            output_path = Path(args.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            logger.info("工作流结果已写入：%s", output_path.resolve())
        print(content)
        return 0
    except Exception as exc:
        logger.exception("Auto_paper 工作流失败")
        error = {
            "status": "failed",
            "query": query,
            "error_type": type(exc).__name__,
            "error_code": error_info(exc)["code"],
            "retryable": error_info(exc)["retryable"],
            "error": str(exc),
        }
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto_paper 端到端科研论文搜索、下载、精排与 RAG 工作流",
    )
    parser.add_argument("query", nargs="+", help="自然语言科研查询")
    parser.add_argument("--max-papers", type=int, default=5, help="最多下载和摄取的论文数")
    parser.add_argument(
        "--max-search-rounds",
        type=int,
        default=3,
        help="Search Agent 最大迭代轮次",
    )
    parser.add_argument("--papers-directory", default="./papers", help="PDF 保存目录")
    parser.add_argument(
        "--text-directory",
        default="./papers/text",
        help="解析文本保存目录",
    )
    parser.add_argument(
        "--chroma-directory",
        default="./data/chroma",
        help="Chroma 持久化目录",
    )
    parser.add_argument("--database-url", default=None, help="SQLAlchemy DATABASE_URL")
    parser.add_argument(
        "--download-engine",
        default="academic_oa",
        choices=[
            "academic_oa",
            "auto",
            "arxiv",
            "openalex",
            "crossref",
            "scholar",
        ],
        help="论文下载搜索来源",
    )
    parser.add_argument("--download-timeout", type=int, default=30, help="下载超时秒数")
    parser.add_argument(
        "--retry-failed-downloads",
        action="store_true",
        help="重试数据库中已有的失败下载任务",
    )
    parser.add_argument("--skip-download", action="store_true", help="只运行到论文注册阶段")
    parser.add_argument("--skip-rag", action="store_true", help="下载 PDF 但不执行 RAG 摄取")
    parser.add_argument(
        "--no-query-llm",
        action="store_true",
        help="查询理解和拆解阶段使用规则，不调用 DeepSeek",
    )
    parser.add_argument("--summarize", action="store_true", help="为成功摄取的论文生成摘要")
    parser.add_argument(
        "--extract-knowledge",
        action="store_true",
        help="为成功摄取的论文抽取结构化科研知识",
    )
    parser.add_argument("--question", default=None, help="向每篇成功摄取的论文提问")
    parser.add_argument("--output", default=None, help="将完整 JSON 结果写入文件")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    parser.add_argument(
        "--skip-citation-tracking",
        action="store_true",
        help="禁用两层 OpenAlex 引文追踪扩展",
    )
    parser.add_argument(
        "--publication-date-to",
        default=None,
        help="只检索该日期及以前发表的论文，格式为 YYYY-MM-DD",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
