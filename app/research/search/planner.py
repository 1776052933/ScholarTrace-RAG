"""根据查询拆解结果生成论文搜索计划。"""

import json
import logging
from typing import Any, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .config import SearchAgentConfig
from .llm_client import DeepSeekClient
from .prompts import SEARCH_PLANNER_PROMPT
from .schema import Reflection, SearchInput, SearchPlan, SearchQuery, SearchStrategy, SearchTask


logger = logging.getLogger(__name__)


class _PlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    strategy: str
    queries: list[SearchQuery] = Field(min_length=3)


class SearchPlanner:
    """使用 DeepSeek 或规则为拆解后的科研查询生成搜索计划。"""

    sources = ("google_scholar", "arxiv", "openalex")

    def __init__(
        self,
        config: Optional[SearchAgentConfig] = None,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:
        self.config = config or SearchAgentConfig.from_env()
        self.llm_client = llm_client

    def create_plan(
        self,
        decomposition_result: Union[SearchInput, Mapping[str, Any]],
        *,
        search_round: int = 1,
    ) -> SearchPlan:
        """调用 DeepSeek，从查询拆解结果生成经过校验的 SearchPlan。"""
        if search_round < 1:
            raise ValueError("search_round must be at least 1")
        search_input, context = _normalize_input(decomposition_result)
        client = self.llm_client or DeepSeekClient()
        self.llm_client = client

        logger.info("调用 DeepSeek 创建第 %d 轮搜索计划", search_round)
        response = client.complete_json(
            SEARCH_PLANNER_PROMPT,
            json.dumps(context, ensure_ascii=False),
            temperature=0.1,
            max_tokens=3000,
        )
        planned = _PlanResponse.model_validate(response)
        return SearchPlan(
            original_query=search_input.original_query,
            search_round=search_round,
            queries=_dedupe_queries(planned.queries),
            strategy=planned.strategy,
        )

    def plan(
        self,
        search_input: SearchInput,
        round_number: int,
        reflection: Optional[Reflection] = None,
    ) -> SearchStrategy:
        """为自主多轮 Pipeline 生成带内部任务字段的可执行计划。"""
        if self.config.use_llm or self.llm_client is not None:
            public_plan = self.create_plan(search_input, search_round=round_number)
            return _executable_plan(public_plan, search_input, self.config.results_per_task)
        return self._rule_plan(search_input, round_number, reflection)

    def _rule_plan(
        self,
        search_input: SearchInput,
        round_number: int,
        reflection: Optional[Reflection],
    ) -> SearchStrategy:
        extra = reflection.add_keywords if reflection else []
        tasks = []
        task_number = 1
        for index, sub_query in enumerate(search_input.sub_queries):
            base_query = sub_query.english_query or sub_query.query
            keywords = _unique([*sub_query.keywords, *extra])
            query = " ".join([base_query, *extra]).strip()
            tasks.append(SearchTask(
                id=f"r{round_number}t{task_number}",
                sub_query_index=index,
                source="all",
                query=query,
                purpose=sub_query.purpose,
                priority=sub_query.priority,
                keywords=keywords,
                limit=self.config.results_per_task,
            ))
            task_number += 1
        return SearchStrategy(
            original_query=search_input.original_query,
            search_round=round_number,
            strategy="首次覆盖全部子查询和数据源" if round_number == 1 else "根据上一轮评估扩展关键词并调整数据源",
            queries=tasks,
        )


def _normalize_input(
    value: Union[SearchInput, Mapping[str, Any]],
) -> tuple[SearchInput, dict[str, Any]]:
    raw = value.model_dump(mode="json") if isinstance(value, SearchInput) else dict(value)
    search_input = value if isinstance(value, SearchInput) else SearchInput.model_validate(raw)
    sub_queries = raw.get("sub_queries") or []
    context = {
        "original_query": search_input.original_query,
        "research_intent": raw.get("research_intent") or search_input.original_query,
        "sub_queries": sub_queries,
        "keywords": raw.get("keywords") or _unique(
            keyword for item in sub_queries for keyword in item.get("keywords", [])
        ),
        "expanded_queries": raw.get("expanded_queries") or [
            item.get("english_query") for item in sub_queries if item.get("english_query")
        ],
        "time_constraints": raw.get("time_constraints") or {},
        "method_constraints": raw.get("method_constraints") or [],
        "task_type": raw.get("task_type") or "",
        "dataset_constraints": raw.get("dataset_constraints") or [],
        "evaluation_requirements": raw.get("evaluation_requirements") or [],
    }
    return search_input, context


def _executable_plan(plan: SearchPlan, search_input: SearchInput, limit: int) -> SearchStrategy:
    tasks = []
    for index, query in enumerate(plan.queries, 1):
        sub_index = _best_sub_query_index(query.query, search_input)
        keywords = search_input.sub_queries[sub_index].keywords
        tasks.append(SearchTask(
            id=f"r{plan.search_round}t{index}",
            sub_query_index=sub_index,
            query=query.query,
            source=query.source,
            purpose=query.purpose,
            priority=query.priority,
            keywords=keywords,
            limit=limit,
        ))
    return SearchStrategy(
        original_query=plan.original_query,
        search_round=plan.search_round,
        queries=tasks,
        strategy=plan.strategy,
    )


def _best_sub_query_index(query: str, search_input: SearchInput) -> int:
    words = set(query.casefold().split())
    scores = []
    for index, sub_query in enumerate(search_input.sub_queries):
        candidate = (sub_query.english_query or sub_query.query).casefold().split()
        scores.append((len(words & set(candidate)), index))
    return max(scores)[1] if scores else 0


def _dedupe_queries(values: list[SearchQuery]) -> list[SearchQuery]:
    seen, result = set(), []
    for value in values:
        key = (value.source.casefold(), value.query.casefold())
        if key not in seen:
            seen.add(key)
            result.append(value)
    if len(result) < 3:
        raise ValueError("DeepSeek must generate at least three distinct search queries")
    return result


def _unique(values):
    seen, result = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result
