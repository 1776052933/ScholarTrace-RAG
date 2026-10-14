"""科研查询是否需要拆解的规则优先分析器。"""

import logging
import re
from typing import Any, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict

from .llm_client import DeepSeekClient
from .prompts import QUERY_COMPLEXITY_PROMPT
from .schema import ComplexityLevel, ComplexityReport, QueryUnderstandingResult


logger = logging.getLogger(__name__)


class _ComplexityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    need_decompose: bool
    reason: str


class QueryAnalyzer:
    """优先使用确定性规则，无法判断时调用 DeepSeek。"""

    def __init__(self, llm_client: Optional[DeepSeekClient] = None, *, use_llm: bool = True):
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.last_reason = ""
        self.last_source = "rules"

    def analyze(
        self,
        query_result: Union[QueryUnderstandingResult, Mapping[str, Any]],
    ) -> bool:
        """返回查询是否需要拆解。"""
        analysis = _coerce_input(query_result)
        rule_reason = _first_rule_reason(analysis)
        if rule_reason:
            self.last_reason = rule_reason
            self.last_source = "rules"
            logger.info("规则判定需要拆解：%s", rule_reason)
            return True

        if not self.use_llm:
            self.last_reason = "未命中复杂查询规则，离线模式下按简单查询处理"
            self.last_source = "rules"
            return False

        client = self.llm_client or DeepSeekClient()
        self.llm_client = client
        logger.info("规则无法确定查询复杂度，调用 DeepSeek 判断")
        payload = client.complete_json(
            QUERY_COMPLEXITY_PROMPT,
            analysis.model_dump_json(),
            temperature=0.0,
        )
        decision = _ComplexityDecision.model_validate(payload)
        self.last_reason = decision.reason
        self.last_source = "deepseek"
        return decision.need_decompose


def _first_rule_reason(analysis: QueryUnderstandingResult) -> str:
    """按需求中的优先顺序返回第一条命中的规则原因。"""
    if len(analysis.entities) >= 3:
        return f"实体数量为 {len(analysis.entities)}，包含多个研究对象"

    markers = re.findall(r"比较|区别|发展|综述|趋势|分析", analysis.original_query)
    if markers:
        return f"查询包含复杂研究动作：{'、'.join(dict.fromkeys(markers))}"

    if len(analysis.expanded_queries) >= 5:
        return f"扩展查询数量为 {len(analysis.expanded_queries)}，研究目标覆盖范围较广"

    if analysis.task_type and _is_broad_scope(analysis):
        return f"任务类型 {analysis.task_type} 存在且研究范围较大"
    return ""


def _is_broad_scope(analysis: QueryUnderstandingResult) -> bool:
    """识别任务是否同时具有较宽的研究范围。"""
    if len(analysis.domain_constraints) >= 2 or len(analysis.dataset_constraints) >= 2:
        return True
    if len(analysis.original_query) >= 60:
        return True
    return bool(re.search(r"全面|综合|广泛|跨领域|多领域|多场景|所有|整体|系统性", analysis.original_query))


def _coerce_input(
    value: Union[QueryUnderstandingResult, Mapping[str, Any]],
) -> QueryUnderstandingResult:
    if isinstance(value, QueryUnderstandingResult):
        return value
    return QueryUnderstandingResult.model_validate(dict(value))


class ComplexityAnalyzer:
    """兼容早期调用的详细复杂度报告适配器。"""

    def analyze(self, analysis: QueryUnderstandingResult) -> ComplexityReport:
        reasons = []
        first = _first_rule_reason(analysis)
        if first:
            reasons.append(first)
        need_decompose = bool(reasons)
        return ComplexityReport(
            level=ComplexityLevel.COMPLEX if need_decompose else ComplexityLevel.SIMPLE,
            score=2 if need_decompose else 0,
            need_decompose=need_decompose,
            reasons=reasons,
        )

