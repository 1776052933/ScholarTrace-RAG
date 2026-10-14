"""科研查询自动拆解模块的统一编排入口。"""

import logging
from typing import Any, Mapping, Optional, Union

from .analyzer import QueryAnalyzer
from .decomposer import QueryDecomposer
from .llm_client import DeepSeekClient
from .optimizer import SubQueryOptimizer
from .schema import DecompositionResult, QueryUnderstandingResult


logger = logging.getLogger(__name__)


class QueryDecompositionPipeline:
    """执行复杂度判断、查询拆解和子查询优化。"""

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        use_llm: bool = True,
        fallback_to_rules: bool = True,
        max_subqueries: int = 8,
    ) -> None:
        self.use_llm = use_llm
        self.fallback_to_rules = fallback_to_rules
        self.llm_client = (llm_client or DeepSeekClient()) if use_llm else None

        # Pipeline 显式管理三个阶段组件，并共享同一个 DeepSeekClient。
        self.analyzer = QueryAnalyzer(self.llm_client, use_llm=use_llm)
        self.decomposer = QueryDecomposer(
            self.llm_client,
            analyzer=self.analyzer,
            use_llm=use_llm,
            max_subqueries=max_subqueries,
        )
        self.optimizer = SubQueryOptimizer(self.llm_client)

    def run(
        self,
        query_understanding_result: Union[QueryUnderstandingResult, Mapping[str, Any]],
    ) -> DecompositionResult:
        """接收查询理解 JSON，返回完整 DecompositionResult。"""
        analysis = (
            query_understanding_result
            if isinstance(query_understanding_result, QueryUnderstandingResult)
            else QueryUnderstandingResult.model_validate(dict(query_understanding_result))
        )
        logger.info("查询拆解 Pipeline 开始")

        # 1. 复杂度判断。
        try:
            need_decompose = self.analyzer.analyze(analysis)
            reason = self.analyzer.last_reason
        except Exception:
            if not self.fallback_to_rules:
                logger.exception("复杂度判断失败")
                raise
            logger.warning("DeepSeek 复杂度判断失败，使用规则回退", exc_info=True)
            fallback_analyzer = QueryAnalyzer(use_llm=False)
            need_decompose = fallback_analyzer.analyze(analysis)
            reason = fallback_analyzer.last_reason

        # 2. 查询拆解。将 Analyzer 结果传入，避免 Decomposer 重复判断。
        try:
            sub_queries = self.decomposer.decompose(
                analysis,
                need_decompose=need_decompose,
                reason=reason,
            )
        except Exception:
            if not self.fallback_to_rules:
                logger.exception("查询拆解失败")
                raise
            logger.warning("DeepSeek 查询拆解失败，使用规则回退", exc_info=True)
            sub_queries = self.decomposer.decompose_with_rules(
                analysis,
                need_decompose=need_decompose,
                reason=reason,
            )

        # 3. 子查询搜索表达优化。
        try:
            sub_queries = self.optimizer.optimize(sub_queries)
        except Exception:
            if not self.fallback_to_rules:
                logger.exception("子查询优化失败")
                raise
            logger.warning("DeepSeek 子查询优化失败，使用规则回退", exc_info=True)
            sub_queries = self.optimizer.optimize_with_rules(sub_queries)

        result = DecompositionResult(
            original_query=analysis.original_query,
            need_decompose=need_decompose,
            reason=reason or ("查询需要拆解" if need_decompose else "查询目标单一，可直接检索"),
            sub_queries=sub_queries,
        )
        logger.info("查询拆解 Pipeline 完成，子查询数=%d", len(result.sub_queries))
        return result
