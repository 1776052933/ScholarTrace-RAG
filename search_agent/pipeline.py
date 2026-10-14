"""自主多轮论文搜索 Agent。"""

from dataclasses import replace
import logging
from typing import Any, Dict, Mapping, Optional, Union

from .config import SearchAgentConfig
from .evaluator import SearchEvaluator
from .memory import SearchMemory
from .planner import SearchPlanner
from .reflector import SearchReflector
from .schema import Reflection, SearchAgentResult, SearchInput, SearchIteration, SearchPlan, SearchResult
from .searcher import PaperSearcher, SearchTool, to_internal_paper
from .tools import ArxivTool, GoogleScholarTool, OpenAlexTool


logger = logging.getLogger(__name__)


class SearchAgent:
    """规划、检索、评估、反思并迭代，直到质量达标或达到轮次上限。"""

    def __init__(
        self,
        config: Optional[SearchAgentConfig] = None,
        *,
        max_round: Optional[int] = None,
        tools: Optional[Dict[str, SearchTool]] = None,
        planner: Optional[SearchPlanner] = None,
        searcher: Optional[PaperSearcher] = None,
        evaluator: Optional[SearchEvaluator] = None,
        reflector: Optional[SearchReflector] = None,
        memory: Optional[SearchMemory] = None,
    ) -> None:
        self.config = config or SearchAgentConfig.from_env()
        if max_round is not None:
            if isinstance(max_round, bool) or not isinstance(max_round, int) or max_round < 1:
                raise ValueError("max_round must be a positive integer")
            self.config = replace(self.config, max_rounds=max_round)
        if self.config.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if searcher is not None and tools is not None:
            raise ValueError("pass either searcher or tools, not both")
        if tools is None and searcher is None:
            tools = {
                "google_scholar": GoogleScholarTool(
                    self.config.serpapi_api_key,
                    self.config.request_timeout_seconds,
                ),
                "arxiv": ArxivTool(self.config.request_timeout_seconds),
                "openalex": OpenAlexTool(
                    self.config.openalex_mailto,
                    self.config.request_timeout_seconds,
                    self.config.openalex_api_key,
                ),
            }
        self.planner = planner or SearchPlanner(self.config)
        self.searcher = searcher or PaperSearcher(
            tools,
            results_per_query=self.config.results_per_task,
        )
        self.evaluator = evaluator or SearchEvaluator(self.config)
        self.reflector = reflector or SearchReflector(use_llm=self.config.use_llm)
        self.memory = memory or SearchMemory()

    def run(self, decomposition_result: Union[SearchInput, Mapping[str, Any]]) -> SearchAgentResult:
        search_input = (
            decomposition_result
            if isinstance(decomposition_result, SearchInput)
            else SearchInput.model_validate(dict(decomposition_result))
        )
        self.memory.clear()
        pending_plan: Optional[SearchPlan] = None
        stopped_reason = "达到最大搜索轮次"
        final_evaluation = None

        for round_number in range(1, self.config.max_rounds + 1):
            logger.info("开始第 %d 轮论文搜索", round_number)
            strategy = pending_plan or self.planner.plan(search_input, round_number)
            pending_plan = None
            search_result = self.searcher.search(
                strategy,
                publication_date_to=getattr(search_input, "publication_date_to", None),
            )
            found = [to_internal_paper(paper) for paper in search_result.papers]
            new_count = self.memory.add_papers(found)
            self.memory.add_search_result(search_result)
            cumulative_result = SearchResult(
                papers=self.memory.papers,
                total=len(self.memory.papers),
                query_used=search_result.query_used,
                query_sources=search_result.query_sources,
            )
            try:
                final_evaluation = (
                    self.evaluator.evaluate(cumulative_result, strategy)
                    if self.config.use_llm
                    else self.evaluator.evaluate_with_rules(cumulative_result, strategy)
                )
            except Exception:
                logger.warning("DeepSeek 搜索评估失败，使用规则回退", exc_info=True)
                final_evaluation = self.evaluator.evaluate_with_rules(cumulative_result, strategy)
            pending_plan = self.reflector.reflect(strategy, cumulative_result, final_evaluation)
            reflection = Reflection(
                should_continue=pending_plan is not None,
                reason=(
                    pending_plan.strategy
                    if pending_plan is not None
                    else "搜索覆盖度已达到停止阈值"
                ),
                add_keywords=(
                    [query.query for query in pending_plan.queries]
                    if pending_plan is not None
                    else []
                ),
                exclude_keywords=[],
                source_adjustments=(
                    list(dict.fromkeys(query.source for query in pending_plan.queries))
                    if pending_plan is not None
                    else []
                ),
            )

            if round_number >= self.config.max_rounds:
                reflection.should_continue = False
                reflection.reason = "达到最大搜索轮次"
            iteration = SearchIteration(
                round_number=round_number,
                plan=strategy,
                new_paper_count=new_count,
                total_paper_count=len(self.memory.papers),
                evaluation=final_evaluation,
                reflection=reflection,
            )
            self.memory.add_iteration(iteration)

            if not final_evaluation.need_refine:
                stopped_reason = "搜索结果质量达到阈值"
                break
            if not reflection.should_continue:
                stopped_reason = reflection.reason
                break

        assert final_evaluation is not None
        papers = sorted(
            self.memory.papers,
            key=lambda paper: (paper.relevance_score, paper.citation_count or 0),
            reverse=True,
        )
        return SearchAgentResult(
            original_query=search_input.original_query,
            rounds_completed=len(self.memory.iterations),
            stopped_reason=stopped_reason,
            papers=papers,
            iterations=self.memory.iterations,
            final_evaluation=final_evaluation,
        )


# 兼容 search_agent 早期版本的类名。
SearchAgentPipeline = SearchAgent
