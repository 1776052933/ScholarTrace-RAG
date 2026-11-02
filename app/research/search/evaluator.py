"""论文搜索结果相关性、质量与研究问题覆盖度评估。"""

import json
import logging
import math
import re
from typing import Optional

from .config import SearchAgentConfig
from .llm_client import DeepSeekClient
from .prompts import SEARCH_EVALUATOR_PROMPT
from .schema import SearchEvaluation, SearchPlan, SearchResult


logger = logging.getLogger(__name__)


class SearchEvaluator:
    """使用 DeepSeek 评估 SearchResult，也提供确定性规则回退。"""

    def __init__(
        self,
        config: Optional[SearchAgentConfig] = None,
        llm_client: Optional[DeepSeekClient] = None,
    ) -> None:
        self.config = config or SearchAgentConfig.from_env()
        self.llm_client = llm_client

    def evaluate(self, search_result: SearchResult, search_plan: SearchPlan) -> SearchEvaluation:
        """调用 DeepSeek，从相关性、论文质量和问题覆盖度三个维度评估。"""
        client = self.llm_client or DeepSeekClient()
        self.llm_client = client
        payload = {
            "search_plan": search_plan.model_dump(mode="json"),
            "search_result": {
                "total": search_result.total,
                "query_used": search_result.query_used,
                "papers": [
                    {
                        "title": paper.title,
                        "abstract": paper.abstract[:1500],
                        "year": paper.year,
                        "venue": paper.venue,
                        "citation_count": paper.citation_count,
                        "source": paper.source,
                        "matched_queries": paper.matched_queries,
                    }
                    for paper in search_result.papers[:50]
                ],
            },
        }
        logger.info("调用 DeepSeek 评估 %d 篇论文", search_result.total)
        response = client.complete_json(
            SEARCH_EVALUATOR_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            max_tokens=1800,
        )
        return SearchEvaluation.model_validate(response)

    def evaluate_with_rules(self, search_result: SearchResult, search_plan: SearchPlan) -> SearchEvaluation:
        """无需 LLM 的可解释评估，用作离线模式和异常回退。"""
        papers = search_result.papers
        if not papers:
            return SearchEvaluation(
                relevance_score=0.0,
                quality_score=0.0,
                coverage_score=0.0,
                missing_information=["没有检索到论文", "所有研究方向均未覆盖"],
                need_refine=True,
            )

        relevance_values = []
        covered_queries = set()
        for paper in papers:
            paper_tokens = _tokens(paper.title + " " + paper.abstract)
            best = 0.0
            for planned_query in search_plan.queries:
                query_tokens = _tokens(planned_query.query)
                score = len(paper_tokens & query_tokens) / max(1, min(len(query_tokens), 12))
                best = max(best, score)
                if score >= self.config.relevance_threshold:
                    covered_queries.add(planned_query.query)
            relevance_values.append(min(1.0, best))

        relevance_score = sum(relevance_values) / len(relevance_values)
        quality_values = []
        for paper in papers:
            citation = min(1.0, math.log1p(paper.citation_count or 0) / math.log(101))
            venue = 1.0 if paper.venue.strip() else 0.0
            abstract = min(1.0, len(paper.abstract.strip()) / 500)
            quality_values.append(0.45 * citation + 0.25 * venue + 0.30 * abstract)
        quality_score = sum(quality_values) / len(quality_values)
        coverage_score = len(covered_queries) / max(1, len(search_plan.queries))

        missing = []
        if relevance_score < self.config.relevance_threshold:
            missing.append("论文与搜索问题的相关性不足")
        if quality_score < 0.5:
            missing.append("高引用、可靠来源或摘要完整的论文不足")
        uncovered = [q.purpose for q in search_plan.queries if q.query not in covered_queries]
        if uncovered:
            missing.append("未覆盖的研究方向：" + "、".join(dict.fromkeys(uncovered)))
        need_refine = (
            relevance_score < self.config.relevance_threshold
            or quality_score < 0.5
            or coverage_score < 1.0
            or len(papers) < self.config.min_relevant_papers
        )
        return SearchEvaluation(
            relevance_score=round(relevance_score, 4),
            quality_score=round(quality_score, 4),
            coverage_score=round(coverage_score, 4),
            missing_information=missing,
            need_refine=need_refine,
        )


def _tokens(text: str) -> set[str]:
    english = re.findall(r"[a-z][a-z0-9-]{1,}", text.casefold())
    chinese = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    stop = {"and", "the", "for", "with", "from", "using", "study", "paper"}
    return {token for token in [*english, *chinese] if token not in stop}

