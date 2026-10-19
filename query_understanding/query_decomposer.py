"""复杂科研问题识别与子查询拆解。"""

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_components.chains import generate_text

from .llm_client import DeepSeekClient
from .prompts import QUERY_DECOMPOSE_PROMPT
from .query_parser import _extract_json_object
from .schema import QueryAnalysis


logger = logging.getLogger(__name__)


class _DecomposedItem(BaseModel):
    """DeepSeek 返回的单个子查询。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class _DecompositionResponse(BaseModel):
    """DeepSeek 拆解结果的内部校验模型。"""

    model_config = ConfigDict(extra="forbid")

    need_decompose: bool
    sub_queries: List[_DecomposedItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sub_queries(self) -> "_DecompositionResponse":
        if self.need_decompose and not self.sub_queries:
            raise ValueError("sub_queries must not be empty when need_decompose is true")
        return self


class QueryDecomposer:
    """判断科研问题复杂度，并使用 DeepSeek 生成可检索子问题。"""

    def __init__(self, llm_client: Optional[DeepSeekClient] = None, max_subqueries: int = 6):
        if max_subqueries < 1:
            raise ValueError("max_subqueries must be greater than zero")
        self.llm_client = llm_client
        self.max_subqueries = max_subqueries

    def decompose(self, analysis: QueryAnalysis) -> QueryAnalysis:
        """分析并更新 ``analysis.sub_queries``，然后返回同一分析对象。

        仅当查询包含多个方法、多个任务、多个研究对象，或包含“比较、分析、
        综述、发展趋势”等复合研究动作时调用 DeepSeek。简单问题不额外拆解。
        """
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("analysis must be a QueryAnalysis instance")

        signals = _complexity_signals(analysis)
        if not signals:
            analysis.sub_queries = []
            logger.info("科研查询无需拆解")
            return analysis

        logger.info("检测到复杂科研查询，拆解信号=%s", ",".join(signals))
        client = self.llm_client or DeepSeekClient()
        self.llm_client = client
        user_content = (
            f"最多生成 {self.max_subqueries} 个子问题。\n"
            f"待拆解的查询分析如下：\n{analysis.to_json()}"
        )
        try:
            response_text = generate_text(client,
                [
                    {"role": "system", "content": QUERY_DECOMPOSE_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            )
            payload, repaired = _extract_json_object(response_text)
            if repaired:
                logger.warning("DeepSeek 拆解响应包含非 JSON 包装，已自动提取")
            response = _DecompositionResponse.model_validate(payload)
        except Exception:
            logger.exception("复杂科研问题拆解失败")
            raise

        if response.need_decompose:
            analysis.sub_queries = [item.question for item in response.sub_queries[: self.max_subqueries]]
        else:
            analysis.sub_queries = []

        # LLM 偶尔会把“多方法比较”误判为单一任务。此类查询属于明确的强制
        # 拆解场景：保留 LLM 已生成的内容，并用确定性规则补齐各方法研究与比较。
        minimum_required = 2 if (
            "multiple_methods" in signals or "compound_research_action" in signals
        ) else 0
        if len(analysis.sub_queries) < minimum_required:
            logger.warning(
                "DeepSeek 返回的子查询不足，期望至少 %d 个，正在执行约束修复",
                minimum_required,
            )
            llm_queries = list(analysis.sub_queries)
            repaired = self.decompose_with_rules(analysis.model_copy(deep=True)).sub_queries
            analysis.sub_queries = _dedupe([*llm_queries, *repaired])[: self.max_subqueries]
        logger.info("科研查询拆解完成，子查询数=%d", len(analysis.sub_queries))
        return analysis

    def decompose_with_rules(self, analysis: QueryAnalysis) -> QueryAnalysis:
        """使用确定性规则拆解，供离线运行或 DeepSeek 失败回退。"""
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("analysis must be a QueryAnalysis instance")
        if not _complexity_signals(analysis):
            analysis.sub_queries = []
            return analysis

        methods = _method_names(analysis)
        research_target = _research_target(analysis)
        sub_queries: List[str] = []
        if len(methods) >= 2:
            sub_queries.extend(
                " ".join(filter(None, [method, research_target, "研究"]))
                for method in methods
            )
            sub_queries.append(f"{' 和 '.join(methods)} 比较")
        else:
            parts = [
                part.strip(" ，。；;")
                for part in re.split(r"(?:，|。|；|;|以及|并且|同时)", analysis.original_query)
                if part.strip(" ，。；;")
            ]
            sub_queries.extend(parts)
        analysis.sub_queries = _dedupe(sub_queries)[: self.max_subqueries]
        logger.info("规则拆解完成，子查询数=%d", len(analysis.sub_queries))
        return analysis


def _complexity_signals(analysis: QueryAnalysis) -> List[str]:
    """返回触发复杂查询拆解的原因。"""
    signals: List[str] = []
    if len(_method_names(analysis)) > 1:
        signals.append("multiple_methods")
    task_names = _entity_names(analysis, {"task"})
    if len(task_names) > 1:
        signals.append("multiple_tasks")
    research_objects = _entity_names(analysis, {"domain", "dataset", "application"})
    if len(research_objects) > 1:
        signals.append("multiple_research_objects")
    if re.search(r"比较|对比|分析|综述|发展趋势|演进趋势", analysis.original_query):
        signals.append("compound_research_action")
    return signals


def _method_names(analysis: QueryAnalysis) -> List[str]:
    values = list(analysis.method_constraints)
    values.extend(_entity_names(analysis, {"method", "model"}))
    comparison = re.search(r"(?:比较|对比)\s*(.+?)(?=在|用于|的应用|的性能|$)", analysis.original_query)
    if comparison:
        values.extend(re.split(r"\s*(?:和|与|及|、|vs\.?|versus)\s*", comparison.group(1), flags=re.I))
    return _dedupe(value for value in values if value)


def _entity_names(analysis: QueryAnalysis, entity_types: set[str]) -> List[str]:
    return _dedupe(entity.name for entity in analysis.entities if entity.type in entity_types)


def _research_target(analysis: QueryAnalysis) -> str:
    """提取方法所针对的任务/研究对象，用于规则回退生成自然子查询。"""
    match = re.search(r"(?:在|用于)\s*(.+?)(?:中|领域|场景|的应用|$)", analysis.original_query)
    if match:
        return match.group(1).strip()
    tasks = [name for name in _entity_names(analysis, {"task"}) if name != analysis.original_query]
    return tasks[0] if tasks else ""


def _dedupe(values) -> List[str]:
    seen, result = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result
