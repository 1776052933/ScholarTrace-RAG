"""Auto_paper 端到端科研论文工作流。

数据流：

    Query Understanding
        → Query Decomposition
        → Search Agent
        → Ranking Agent
        → Literature Database / Paper Downloader
        → RAG Reader

命令行示例：

    python main.py "近五年基于大语言模型的药物发现研究"

    python main.py "Transformer 医学图像分割" \
        --max-papers 3 --summarize --question "论文提出了什么方法？"
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import logging
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Dict, Optional

from literature_database.database import DatabaseManager
from app.config import ModelSettings
from literature_database.service import LiteratureService
from query_decomposition.pipeline import QueryDecompositionPipeline
from query_decomposition.schema import SubQuery as DecompositionSubQuery
from query_understanding.pipeline import QueryUnderstandingPipeline
from rag_reader.errors import error_info
from rag_reader.pipeline import RAGReader
from rag_reader.store_factory import create_vector_store
from ranking_agent.pipeline import RankingAgent
from search_agent.citation_tracker import CitationTracker
from search_agent.pipeline import SearchAgent


logger = logging.getLogger("auto_paper")


class WorkflowCancelled(RuntimeError):
    """Raised when a persisted M2 job is cancelled between workflow stages."""


class AutoPaperWorkflow:
    """组合 Auto_paper 全部 Agent 和资源处理模块。

    所有组件都支持注入，既便于测试，也允许后续替换某个 Agent 而不修改主流程。
    """

    def __init__(
        self,
        query_understanding: Optional[Any] = None,
        query_decomposition: Optional[Any] = None,
        search_agent: Optional[Any] = None,
        ranking_agent: Optional[Any] = None,
        literature_service: Optional[LiteratureService] = None,
        rag_reader: Optional[Any] = None,
        *,
        use_query_llm: bool = True,
        max_search_rounds: int = 3,
        database_url: Optional[str] = None,
        chroma_directory: str = "./data/chroma",
        enable_rag: bool = True,
        rag_workers: int = 3,
        citation_tracker: Optional[Any] = None,
        enable_citation_tracking: bool = True,
        model_settings: Optional[ModelSettings] = None,
    ) -> None:
        if (
            isinstance(max_search_rounds, bool)
            or not isinstance(max_search_rounds, int)
            or max_search_rounds < 1
        ):
            raise ValueError("max_search_rounds must be a positive integer")
        if isinstance(rag_workers, bool) or not isinstance(rag_workers, int) or rag_workers < 1:
            raise ValueError("rag_workers must be a positive integer")

        _configure_project_api_keys()
        if model_settings is not None and model_settings.api_key.get_secret_value():
            provider = model_settings.client_kwargs()
            if query_understanding is None and use_query_llm:
                from query_understanding.llm_client import DeepSeekClient as UnderstandingClient
                query_understanding = QueryUnderstandingPipeline(
                    llm_client=UnderstandingClient(**provider, timeout_seconds=model_settings.timeout_seconds),
                    use_llm=True,
                    fallback_to_rules=True,
                )
            if query_decomposition is None and use_query_llm:
                from query_decomposition.llm_client import DeepSeekClient as DecompositionClient
                query_decomposition = QueryDecompositionPipeline(
                    llm_client=DecompositionClient(**provider, timeout_seconds=model_settings.timeout_seconds),
                    use_llm=True,
                    fallback_to_rules=True,
                )
            if search_agent is None:
                from search_agent.config import SearchAgentConfig
                from search_agent.evaluator import SearchEvaluator
                from search_agent.llm_client import DeepSeekClient as SearchClient
                from search_agent.planner import SearchPlanner
                from search_agent.reflector import SearchReflector
                search_config = SearchAgentConfig.from_env()
                search_client = SearchClient(**provider, timeout_seconds=model_settings.timeout_seconds)
                search_agent = SearchAgent(
                    config=search_config,
                    max_round=max_search_rounds,
                    planner=SearchPlanner(search_config, llm_client=search_client),
                    evaluator=SearchEvaluator(search_config, llm_client=search_client),
                    reflector=SearchReflector(llm_client=search_client, use_llm=search_config.use_llm),
                )
            if ranking_agent is None:
                from ranking_agent.cache import EvaluationCache
                from ranking_agent.evaluator import PaperEvaluator
                from ranking_agent.llm_client import DeepSeekClient as RankingClient
                ranking_agent = RankingAgent(evaluator=PaperEvaluator(
                    llm_client=RankingClient(**provider, timeout=model_settings.timeout_seconds),
                    cache=EvaluationCache(),
                ))
        self.query_understanding = query_understanding or QueryUnderstandingPipeline(
            use_llm=use_query_llm,
            fallback_to_rules=True,
        )
        self.query_decomposition = query_decomposition or QueryDecompositionPipeline(
            use_llm=use_query_llm,
            fallback_to_rules=True,
        )
        self.search_agent = search_agent or SearchAgent(max_round=max_search_rounds)
        if isinstance(self.search_agent, SearchAgent):
            from app.workflows.search import GraphSearchAgent
            self.search_agent = GraphSearchAgent(
                self.search_agent,
                checkpoint_path=Path(chroma_directory).parent / "search_graph.sqlite3",
            )
        self.ranking_agent = ranking_agent or RankingAgent()
        self.citation_tracker = citation_tracker
        if self.citation_tracker is None and enable_citation_tracking:
            self.citation_tracker = CitationTracker(
                embedding_model=getattr(self.ranking_agent, "embedding_model", None),
            )
        self.literature_service = literature_service or LiteratureService(
            DatabaseManager(database_url),
        )
        self.rag_reader = rag_reader
        self.rag_workers = rag_workers
        if self.rag_reader is None and enable_rag:
            from rag_reader.config import RAGSettings
            rag_settings = RAGSettings.from_env()
            if model_settings is not None:
                rag_settings = rag_settings.model_copy(update={
                    "llm_model": model_settings.model,
                    "llm_base_url": model_settings.base_url,
                    "llm_timeout_seconds": model_settings.timeout_seconds,
                })
            self.rag_reader = RAGReader(
                vector_store=create_vector_store(rag_settings, chroma_directory=chroma_directory),
                settings=rag_settings,
            )

    def run(
        self,
        query: str,
        *,
        max_papers: int = 5,
        download_papers: bool = True,
        ingest_rag: bool = True,
        papers_directory: str = "./papers",
        text_directory: str = "./papers/text",
        download_engine: str = "academic_oa",
        download_timeout: int = 30,
        retry_failed_downloads: bool = False,
        question: Optional[str] = None,
        summarize: bool = False,
        extract_knowledge: bool = False,
        publication_date_to: Optional[str] = None,
        should_cancel: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行完整工作流并返回可直接 JSON 序列化的结果。"""
        query = _required_text(query, "query")
        publication_date_to = _optional_iso_date(publication_date_to)
        if isinstance(max_papers, bool) or not isinstance(max_papers, int) or max_papers < 1:
            raise ValueError("max_papers must be a positive integer")
        if not download_papers and ingest_rag:
            raise ValueError("ingest_rag requires download_papers=True")
        if ingest_rag and self.rag_reader is None:
            raise RuntimeError("RAG Reader is disabled for this workflow")
        if (question or summarize or extract_knowledge) and not ingest_rag:
            raise ValueError(
                "question, summarize and extract_knowledge require ingest_rag=True"
            )

        def check_cancel() -> None:
            if should_cancel is not None and should_cancel():
                raise WorkflowCancelled("research job was cancelled")

        check_cancel()
        started = perf_counter()
        stages: Dict[str, Any] = {}

        analysis, elapsed = _timed(
            "query_understanding",
            self.query_understanding.run,
            query,
        )
        stages["query_understanding"] = _stage(analysis, elapsed)
        check_cancel()

        decomposition, elapsed = _timed(
            "query_decomposition",
            self.query_decomposition.run,
            _understanding_for_decomposition(analysis),
        )
        decomposition = _merge_upstream_subqueries(decomposition, analysis)
        stages["query_decomposition"] = _stage(decomposition, elapsed)
        check_cancel()

        search_input = _model_data(decomposition)
        if publication_date_to:
            search_input["publication_date_to"] = publication_date_to
        search_result, elapsed = _timed(
            "search_agent",
            self.search_agent.run,
            search_input,
        )
        search_elapsed = elapsed

        if self.citation_tracker is not None:
            citation_expansion, elapsed = _timed(
                "citation_tracking",
                self.citation_tracker.expand,
                query,
                list(_value(search_result, "papers") or []),
                publication_date_to=publication_date_to,
            )
            search_result = _replace_papers(search_result, citation_expansion.papers)
            stages["citation_tracking"] = _stage(citation_expansion.summary, elapsed)

        stages["search_agent"] = _stage(search_result, search_elapsed)
        check_cancel()

        ranking_result, elapsed = _timed(
            "ranking_agent",
            self.ranking_agent.run,
            _model_data(analysis),
            search_result,
        )
        stages["ranking_agent"] = _stage(ranking_result, elapsed)
        check_cancel()

        ranked_papers = list(_value(ranking_result, "papers") or [])
        selected_papers = ranked_papers[:max_papers]
        ranking_candidates = ranked_papers if download_papers else selected_papers
        candidate_ids = {
            str(_value(paper, "paper_id"))
            for paper in ranking_candidates
            if _value(paper, "paper_id")
        }
        scores = {
            paper_id: score
            for paper_id, score in getattr(self.ranking_agent, "last_scores", {}).items()
            if paper_id in candidate_ids
        }
        ranking_payload = {"papers": [_model_data(paper) for paper in ranking_candidates]}

        check_cancel()
        literature_started = perf_counter()
        logger.info("开始阶段：literature_database")
        ingestion_results = []
        downloadable_resources = []
        if download_papers and ingest_rag:
            ingestion_results = self.literature_service.prepare_and_ingest(
                ranking_payload,
                self.rag_reader,
                scores=scores,
                target_count=max_papers,
                output_dir=papers_directory,
                text_output_dir=text_directory,
                engine=download_engine,
                timeout=download_timeout,
                retry_failed=retry_failed_downloads,
            )
        elif download_papers:
            downloadable_resources = self.literature_service.prepare_for_reader(
                ranking_payload,
                scores=scores,
                target_count=max_papers,
                output_dir=papers_directory,
                engine=download_engine,
                timeout=download_timeout,
                retry_failed=retry_failed_downloads,
            )
        else:
            ingestion_results = self.literature_service.register_ranked_papers(
                ranking_payload,
                scores=scores,
            )
        download_results = getattr(self.literature_service, "last_download_results", [])
        original_ranks = {
            str(_value(paper, "paper_id")): index
            for index, paper in enumerate(ranked_papers, start=1)
            if _value(paper, "paper_id")
        }
        title_ranks = {
            str(_value(paper, "title") or "").strip().casefold(): index
            for index, paper in enumerate(ranked_papers, start=1)
            if _value(paper, "title")
        }
        for item in download_results:
            paper_id = str(_value(item, "paper_id") or "")
            title_key = str(_value(item, "title") or "").strip().casefold()
            rank = original_ranks.get(paper_id) or title_ranks.get(title_key)
            if isinstance(item, dict) and rank is not None:
                item["original_rank"] = rank
        if download_papers:
            selected_records = []
            for item in download_results:
                if not bool(_value(item, "success")):
                    continue
                rank = _value(item, "original_rank")
                if not isinstance(rank, int) or rank < 1 or rank > len(ranked_papers):
                    continue
                paper = _model_data(ranked_papers[rank - 1])
                paper["paper_id"] = str(_value(item, "paper_id") or paper.get("paper_id") or "")
                paper["original_rank"] = rank
                selected_records.append(paper)
                if len(selected_records) >= max_papers:
                    break
            selected_papers = selected_records
        else:
            selected_papers = [
                {**_model_data(paper), "original_rank": rank}
                for rank, paper in enumerate(selected_papers, start=1)
            ]
        selected_paper_ids = [
            str(_value(paper, "paper_id"))
            for paper in selected_papers
            if _value(paper, "paper_id")
        ]
        stages["literature_database"] = {
            "elapsed_seconds": _elapsed(literature_started),
            "selected_paper_count": len(selected_papers),
            "requested_paper_count": max_papers,
            "selected_paper_ids": selected_paper_ids,
            "selected_papers": _json_value(selected_papers),
            "attempted_paper_count": len(download_results),
            "download_enabled": download_papers,
            "rag_ingest_enabled": ingest_rag,
            "download_results": _json_value(
                download_results
            ),
            "download_success_count": sum(
                bool(_value(item, "success"))
                for item in download_results
            ),
            "ingestion_results": _json_value(ingestion_results),
            "downloadable_resources": _json_value(downloadable_resources),
            "missing_core_papers": _json_value(
                self.literature_service.get_missing_core_papers()
            ),
        }

        logger.info("完成阶段：literature_database，耗时 %.3f 秒", _elapsed(literature_started))
        check_cancel()
        rag_started = perf_counter()
        logger.info("开始阶段：rag_reader")
        rag_outputs = _run_rag_tasks(
            self.rag_reader,
            ingestion_results,
            summarize=summarize,
            extract_knowledge=extract_knowledge,
            question=question,
            max_workers=self.rag_workers,
        ) if ingest_rag else {}
        stages["rag_reader"] = {
            "elapsed_seconds": _elapsed(rag_started),
            "processed_paper_count": sum(
                bool(_value(item, "success")) for item in ingestion_results
            ),
            "outputs": rag_outputs,
        }

        logger.info("完成阶段：rag_reader，耗时 %.3f 秒", _elapsed(rag_started))
        check_cancel()
        return {
            "status": "completed",
            "query": query,
            "elapsed_seconds": _elapsed(started),
            "stages": stages,
        }


def _run_rag_tasks(
    reader: Any,
    ingestion_results: Sequence[Any],
    *,
    summarize: bool,
    extract_knowledge: bool,
    question: Optional[str],
    max_workers: int,
) -> Dict[str, Any]:
    paper_ids = [
        str(_value(item, "paper_id"))
        for item in ingestion_results
        if bool(_value(item, "success")) and _value(item, "paper_id")
    ]
    outputs: Dict[str, Dict[str, Any]] = {
        paper_id: {} for paper_id in dict.fromkeys(paper_ids)
    }
    jobs = []
    for paper_id in outputs:
        if extract_knowledge:
            jobs.append((
                paper_id,
                "knowledge",
                "knowledge_error",
                "论文知识抽取",
                reader.extract_knowledge,
                (paper_id,),
            ))
        if summarize:
            jobs.append((
                paper_id,
                "summary",
                "summary_error",
                "论文摘要",
                reader.summarize,
                (paper_id,),
            ))
        if question:
            jobs.append((
                paper_id,
                "answer",
                "answer_error",
                "论文问答",
                reader.ask,
                (paper_id, question),
            ))
    if not jobs:
        return outputs

    worker_count = min(max_workers, len(jobs))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="rag-analysis",
    ) as executor:
        submitted = [
            (job, executor.submit(job[4], *job[5]))
            for job in jobs
        ]
        for job, future in submitted:
            paper_id, result_key, error_key, label, _, _ = job
            try:
                outputs[paper_id][result_key] = _json_value(future.result())
            except Exception as exc:
                logger.exception("%s失败：paper_id=%s", label, paper_id)
                outputs[paper_id][error_key] = f"{type(exc).__name__}: {exc}"
                outputs[paper_id][f"{result_key}_error_info"] = error_info(exc)
    return outputs


def _configure_project_api_keys() -> None:
    """将项目 Key Manager 中的密钥映射到各模块统一使用的环境变量。"""
    try:
        from API_key.key_manager import key_get
    except ImportError:
        return
    for service, environment_name in (
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("serpapi", "SERPAPI_API_KEY"),
        ("openalex", "OPENALEX_API_KEY"),
    ):
        if os.getenv(environment_name):
            continue
        try:
            value = str(key_get(service) or "").strip()
        except Exception:
            logger.debug("无法从 Key Manager 读取 %s", service, exc_info=True)
            continue
        if value:
            os.environ[environment_name] = value


def _timed(name: str, function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = perf_counter()
    logger.info("开始阶段：%s", name)
    result = function(*args, **kwargs)
    elapsed = _elapsed(started)
    logger.info("完成阶段：%s，耗时 %.3f 秒", name, elapsed)
    return result, elapsed


def _stage(result: Any, elapsed: float) -> Dict[str, Any]:
    return {
        "elapsed_seconds": elapsed,
        "result": _json_value(result),
    }


def _model_data(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    raise TypeError("pipeline result must be a mapping or Pydantic model")


def _replace_papers(value: Any, papers: list[Any]) -> Any:
    if hasattr(value, "model_copy"):
        return value.model_copy(update={"papers": papers})
    if isinstance(value, Mapping):
        updated = dict(value)
        updated["papers"] = papers
        if "total" in updated:
            updated["total"] = len(papers)
        return updated
    raise TypeError("search result must be a mapping or Pydantic model")


def _understanding_for_decomposition(value: Any) -> Dict[str, Any]:
    """适配查询理解与独立拆解模块间的可空字段差异。"""
    data = _model_data(value)
    data["task_type"] = data.get("task_type") or ""
    return data


def _merge_upstream_subqueries(decomposition: Any, understanding: Any) -> Any:
    """保留查询理解阶段已经生成、但独立拆解阶段未覆盖的检索方向。"""
    upstream = _model_data(understanding).get("sub_queries") or []
    if not upstream or not hasattr(decomposition, "model_copy"):
        return decomposition

    existing = list(_value(decomposition, "sub_queries") or [])
    seen = {_normalized_query(_value(item, "query")) for item in existing}
    keywords = [
        str(item).strip()
        for item in _model_data(understanding).get("keywords") or []
        if str(item).strip()
    ][:12]
    for raw_query in upstream:
        query = re.sub(
            r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])",
            " ",
            str(raw_query).strip(),
        )
        query = re.sub(r"\s+", " ", query).strip()
        normalized = _normalized_query(query)
        if not query or normalized in seen:
            continue
        existing.append(DecompositionSubQuery(
            query=query,
            purpose="补充查询理解阶段识别出的独立检索方向",
            query_type=_subquery_type(query),
            priority="high",
            keywords=keywords,
        ))
        seen.add(normalized)
    return decomposition.model_copy(update={"sub_queries": existing})


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = " ".join(value.strip().split())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _optional_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) == 8:
        raw = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError("publication_date_to must use YYYY-MM-DD format") from exc


def _subquery_type(query: str) -> str:
    if re.search(r"比较|对比|区别|compare|versus|\bvs\b", query, re.I):
        return "comparison"
    if re.search(r"趋势|发展|演进|trend|evolution", query, re.I):
        return "trend_analysis"
    if re.search(r"数据集|dataset|benchmark", query, re.I):
        return "dataset_search"
    if re.search(r"应用|场景|application|use case", query, re.I):
        return "application"
    if re.search(r"性能|评价|评估|指标|performance|evaluation|metric", query, re.I):
        return "evaluation"
    return "method_search"


def _normalized_query(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _elapsed(started: float) -> float:
    return round(perf_counter() - started, 6)
