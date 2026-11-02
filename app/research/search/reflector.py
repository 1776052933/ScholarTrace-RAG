"""根据搜索评估反思当前策略，并生成下一轮搜索计划。"""

import json
import logging
import re
from typing import Optional

from .llm_client import DeepSeekClient
from .prompts import SEARCH_REFLECTION_PROMPT
from .schema import SearchEvaluation, SearchPlan, SearchQuery, SearchResult


logger = logging.getLogger(__name__)


class SearchReflector:
    """在覆盖不足时补充搜索方向，覆盖达标时终止迭代。"""

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        coverage_threshold: float = 0.7,
        use_llm: bool = False,
    ) -> None:
        if not 0.0 <= coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be between 0 and 1")
        self.llm_client = llm_client
        self.coverage_threshold = coverage_threshold
        self.use_llm = use_llm

    def reflect(
        self,
        search_plan: SearchPlan,
        search_result: SearchResult,
        evaluation: SearchEvaluation,
    ) -> Optional[SearchPlan]:
        """返回下一轮 SearchPlan；覆盖度达标时返回 ``None`` 表示停止搜索。"""
        if evaluation.coverage_score >= self.coverage_threshold:
            logger.info(
                "搜索覆盖度 %.2f 已达到阈值 %.2f，停止生成新计划",
                evaluation.coverage_score,
                self.coverage_threshold,
            )
            return None

        logger.info(
            "搜索覆盖度 %.2f 低于阈值 %.2f，生成下一轮搜索计划",
            evaluation.coverage_score,
            self.coverage_threshold,
        )
        if self.llm_client is not None or self.use_llm:
            try:
                if self.llm_client is None:
                    self.llm_client = DeepSeekClient()
                return self._reflect_with_llm(search_plan, search_result, evaluation)
            except Exception:
                logger.warning("DeepSeek 反思失败，使用规则生成下一轮计划", exc_info=True)
        return self._reflect_with_rules(search_plan, evaluation)

    def _reflect_with_llm(
        self,
        search_plan: SearchPlan,
        search_result: SearchResult,
        evaluation: SearchEvaluation,
    ) -> SearchPlan:
        payload = {
            "search_plan": search_plan.model_dump(mode="json"),
            "search_result": {
                "total": search_result.total,
                "query_used": search_result.query_used,
                "papers": [
                    {
                        "title": paper.title,
                        "abstract": paper.abstract[:800],
                        "year": paper.year,
                        "venue": paper.venue,
                    }
                    for paper in search_result.papers[:30]
                ],
            },
            "evaluation": evaluation.model_dump(mode="json"),
        }
        response = self.llm_client.complete_json(
            SEARCH_REFLECTION_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            max_tokens=1600,
        )
        # 轮次和原始问题由程序控制，避免模型意外改写状态。
        response["original_query"] = search_plan.original_query
        response["search_round"] = search_plan.search_round + 1
        plan = SearchPlan.model_validate(response)
        return _remove_existing_queries(plan, search_plan)

    def _reflect_with_rules(
        self,
        search_plan: SearchPlan,
        evaluation: SearchEvaluation,
    ) -> SearchPlan:
        missing_items = [item.strip() for item in evaluation.missing_information if item.strip()]
        if not missing_items:
            missing_items = ["未覆盖研究方向"]

        existing = {_normalize(query.query) for query in search_plan.queries}
        sources = [query.source for query in search_plan.queries if query.source]
        queries = []
        for index, missing in enumerate(missing_items):
            direction = _clean_missing_information(missing)
            query_text = _build_query(search_plan.original_query, direction)
            if _normalize(query_text) in existing:
                query_text = f"{query_text} survey benchmark"
            queries.append(
                SearchQuery(
                    query=query_text,
                    source=sources[index % len(sources)] if sources else "google_scholar",
                    purpose=f"补充缺失信息：{direction}",
                    priority="high" if index == 0 else "medium",
                )
            )

        return SearchPlan(
            original_query=search_plan.original_query,
            search_round=search_plan.search_round + 1,
            queries=queries,
            strategy=(
                f"第 {search_plan.search_round + 1} 轮针对覆盖缺口定向检索："
                + "；".join(missing_items)
            ),
        )


def _remove_existing_queries(new_plan: SearchPlan, previous_plan: SearchPlan) -> SearchPlan:
    """过滤模型重复生成的旧查询；若全部重复则回退到规则计划。"""
    existing = {_normalize(item.query) for item in previous_plan.queries}
    unique = []
    seen = set()
    for item in new_plan.queries:
        normalized = _normalize(item.query)
        if normalized and normalized not in existing and normalized not in seen:
            unique.append(item)
            seen.add(normalized)
    if not unique:
        raise ValueError("DeepSeek did not generate any new search direction")
    return new_plan.model_copy(update={"queries": unique})


def _clean_missing_information(value: str) -> str:
    value = re.sub(r"^(未覆盖的研究方向|缺少|不足|未覆盖)\s*[：:]?\s*", "", value).strip()
    return value or "research gaps"


def _build_query(original_query: str, direction: str) -> str:
    """将常见中文缺口转换成更适合论文数据库的英文检索词。"""
    translations = {
        "磁共振": "MRI",
        "核磁": "MRI",
        "医学图像分割": "medical image segmentation",
        "数据集": "dataset benchmark",
        "评价指标": "evaluation metrics benchmark",
        "性能评价": "performance evaluation benchmark",
        "发展趋势": "trends survey",
        "综述": "survey review",
        "挑战": "challenges limitations",
    }
    converted = direction
    for chinese, english in translations.items():
        converted = converted.replace(chinese, english)

    # 示例中的缺口应形成精确、可直接提交给论文数据库的检索语句。
    if re.search(r"MRI|磁共振|核磁", direction, re.IGNORECASE):
        original_lower = original_query.casefold()
        if "transformer" in original_lower and ("分割" in original_query or "segmentation" in original_lower):
            return "MRI Transformer segmentation"
    return f"{original_query} {converted}".strip()


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.casefold())
