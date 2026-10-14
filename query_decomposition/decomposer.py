"""科研查询分析与子查询生成。"""

import logging
import re
from typing import Any, List, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .analyzer import QueryAnalyzer
from .llm_client import DeepSeekClient
from .prompts import QUERY_DECOMPOSE_PROMPT
from .schema import QueryUnderstandingResult, SubQuery

logger = logging.getLogger(__name__)


class _GeneratedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_queries: List[SubQuery] = Field(min_length=1)


class QueryDecomposer:
    """先调用 QueryAnalyzer，再按判断结果返回原查询或 DeepSeek 子查询。"""

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        analyzer: Optional[QueryAnalyzer] = None,
        use_llm: bool = True,
        max_subqueries: int = 8,
    ) -> None:
        if max_subqueries < 1:
            raise ValueError("max_subqueries must be greater than zero")
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.max_subqueries = max_subqueries
        self.analyzer = analyzer or QueryAnalyzer(llm_client, use_llm=use_llm)
        self.last_need_decompose = False
        self.last_reason = ""

    def decompose(
        self,
        query_result: Union[QueryUnderstandingResult, Mapping[str, Any]],
        *,
        need_decompose: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> List[SubQuery]:
        analysis = _coerce_input(query_result)

        # Step 1：规则优先，无法判断时由 QueryAnalyzer 调用 DeepSeek。
        if need_decompose is None:
            self.last_need_decompose = self.analyzer.analyze(analysis)
            self.last_reason = self.analyzer.last_reason
        else:
            self.last_need_decompose = need_decompose
            self.last_reason = reason or ""

        # Step 2：无需拆解时，原始查询是唯一子查询。
        if not self.last_need_decompose:
            return [_direct_query(analysis)]

        if not self.use_llm:
            return _rule_queries(analysis)

        # Step 3：需要拆解时，调用 DeepSeek 生成多个结构化子查询。
        client = self.llm_client or self.analyzer.llm_client or DeepSeekClient()
        self.llm_client = client
        payload = client.complete_json(
            QUERY_DECOMPOSE_PROMPT,
            f"最多生成 {self.max_subqueries} 个子查询。\nquery understanding JSON：\n"
            + analysis.model_dump_json(),
            temperature=0.1,
        )
        response = _GeneratedResponse.model_validate(payload)
        return _ensure_coverage(analysis, _dedupe_queries(response.sub_queries))[: self.max_subqueries]

    def decompose_with_rules(
        self,
        query_result: Union[QueryUnderstandingResult, Mapping[str, Any]],
        *,
        need_decompose: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> List[SubQuery]:
        """离线模式或 DeepSeek 失败时执行相同流程。"""
        analysis = _coerce_input(query_result)
        if need_decompose is None:
            analyzer = QueryAnalyzer(use_llm=False)
            self.last_need_decompose = analyzer.analyze(analysis)
            self.last_reason = analyzer.last_reason
        else:
            self.last_need_decompose = need_decompose
            self.last_reason = reason or ""
        return _rule_queries(analysis) if self.last_need_decompose else [_direct_query(analysis)]


def _ensure_coverage(analysis: QueryUnderstandingResult, generated: List[SubQuery]) -> List[SubQuery]:
    """补齐 LLM 遗漏的方法、应用、比较、评价、趋势等明确维度。"""
    result = list(generated)
    rules = _rule_queries(analysis)
    methods = _method_names(analysis)

    for method in methods:
        if not any(item.query_type == "method_search" and method.casefold() in item.query.casefold() for item in result):
            result.extend(item for item in rules if item.query_type == "method_search" and method.casefold() in item.query.casefold())

    required_types = set()
    query = _signal_text(analysis)
    if len(methods) > 1 or re.search(r"比较|区别|对比|compare|versus|\bvs\b", query, re.I):
        required_types.add("comparison")
    if _has_application_signal(analysis):
        required_types.add("application")
    if re.search(r"评价|评估|性能|benchmark|evaluation|performance", query, re.I) and "comparison" not in required_types:
        required_types.add("evaluation")
    if re.search(r"发展|趋势|演进|trend|evolution", query, re.I):
        required_types.add("trend_analysis")
    if analysis.dataset_constraints or re.search(r"数据集|dataset|benchmark", query, re.I):
        required_types.add("dataset_search")

    existing = {item.query_type for item in result}
    for query_type in required_types - existing:
        result.extend(item for item in rules if item.query_type == query_type)
    if re.search(r"挑战|局限|难点|challenge|limitation", query, re.I) and not any(
        re.search(r"挑战|局限|难点|challenge|limitation", item.query + item.purpose, re.I)
        for item in result
    ):
        result.extend(
            item for item in rules
            if re.search(r"挑战|局限|难点|challenge|limitation", item.query + item.purpose, re.I)
        )
    return _dedupe_queries(result)


def _rule_queries(analysis: QueryUnderstandingResult) -> List[SubQuery]:
    methods = _method_names(analysis)
    context = _primary_context(analysis)
    signal_text = _signal_text(analysis)
    result: List[SubQuery] = []

    for method in methods:
        result.append(SubQuery(
            query=_join(method, context, "models methods performance"),
            purpose=f"分析 {method} 模型、方法及其性能表现",
            query_type="method_search",
            priority="high",
        ))

    is_comparison = len(methods) > 1 or bool(re.search(r"比较|区别|对比|compare|versus|\bvs\b", analysis.original_query, re.I))
    if is_comparison:
        result.append(SubQuery(
            query=_join(" vs ".join(methods), context, "benchmark performance comparison"),
            purpose="比较不同方法的性能与适用条件",
            query_type="comparison",
            priority="high",
        ))

    if _has_application_signal(analysis):
        result.append(SubQuery(
            query=_join(" ".join(methods), context, "applications"),
            purpose="分析相关方法在目标领域中的应用",
            query_type="application",
            priority="medium",
        ))

    if re.search(r"评价|评估|指标|性能|benchmark|evaluation|metric|performance", signal_text, re.I) and not is_comparison:
        result.append(SubQuery(
            query=_join(" ".join(methods), context, "evaluation benchmark performance"),
            purpose="检索性能评价指标与实验结果",
            query_type="evaluation",
            priority="high",
        ))

    for dataset in analysis.dataset_constraints:
        result.append(SubQuery(
            query=_join(" ".join(methods), context, dataset, "benchmark"),
            purpose=f"分析 {dataset} 数据集上的研究和实验结果",
            query_type="dataset_search",
            priority="medium",
        ))

    if not analysis.dataset_constraints and re.search(r"数据集|dataset|benchmark", signal_text, re.I):
        result.append(SubQuery(
            query=_join(" ".join(methods), context, "datasets benchmarks"),
            purpose="检索常用数据集、公开基准及其使用方式",
            query_type="dataset_search",
            priority="medium",
        ))

    if re.search(r"挑战|局限|难点|challenge|limitation", signal_text, re.I):
        result.append(SubQuery(
            query=_join(" ".join(methods), context, "challenges limitations open problems"),
            purpose="分析当前方法面临的挑战、局限和开放问题",
            query_type="evaluation",
            priority="medium",
        ))

    if re.search(r"近[一二三四五六七八九十\d]+年|发展|趋势|演进|trend|evolution", signal_text, re.I):
        result.append(SubQuery(
            query=_join(context, "research trends evolution"),
            purpose="分析研究方向的发展过程与最新趋势",
            query_type="trend_analysis",
            priority="medium",
        ))

    return _dedupe_queries(result) or [_direct_query(analysis)]


def _direct_query(analysis: QueryUnderstandingResult) -> SubQuery:
    return SubQuery(query=analysis.original_query, purpose="直接检索原始科研问题", query_type="method_search", priority="high")


def _method_names(analysis: QueryUnderstandingResult) -> List[str]:
    return _unique(analysis.method_constraints + [e.name for e in analysis.entities if e.type in {"method", "model"}])


def _primary_context(analysis: QueryUnderstandingResult) -> str:
    tasks = [e.name for e in analysis.entities if e.type == "task"]
    domains = analysis.domain_constraints + [e.name for e in analysis.entities if e.type in {"domain", "application"}]
    return " ".join(_unique([*tasks, *domains])) or analysis.research_intent


def _has_application_signal(analysis: QueryUnderstandingResult) -> bool:
    return any(e.type == "application" for e in analysis.entities) or bool(re.search(r"应用|场景|application|use case", analysis.original_query, re.I))


def _signal_text(analysis: QueryUnderstandingResult) -> str:
    """汇总原查询、研究意图和关键词，用于识别需要覆盖的研究维度。"""
    return " ".join([analysis.original_query, analysis.research_intent, *analysis.keywords])


def _coerce_input(value: Union[QueryUnderstandingResult, Mapping[str, Any]]) -> QueryUnderstandingResult:
    return value if isinstance(value, QueryUnderstandingResult) else QueryUnderstandingResult.model_validate(dict(value))


def _join(*values: str) -> str:
    return " ".join(value.strip() for value in values if value and value.strip())


def _unique(values) -> List[str]:
    seen, result = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def _dedupe_queries(values: List[SubQuery]) -> List[SubQuery]:
    seen, result = set(), []
    for value in values:
        if value.query.casefold() not in seen:
            seen.add(value.query.casefold())
            result.append(value)
    return result
