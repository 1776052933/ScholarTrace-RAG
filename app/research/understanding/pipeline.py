"""科研查询理解的统一执行流水线。"""

import logging
from typing import Optional

from .llm_client import DeepSeekClient
from .query_decomposer import QueryDecomposer
from .query_parser import QueryParser
from .query_rewriter import QueryRewriter
from .schema import QueryAnalysis


logger = logging.getLogger(__name__)


class QueryUnderstandingPipeline:
    """依次执行查询解析、复杂问题拆解和论文检索式扩展。

    默认创建一个 ``DeepSeekClient`` 并共享给所有组件，保证同一套 API 配置被
    Parser、Decomposer 和 Rewriter 复用。测试或其他 Agent 也可以注入兼容客户端。
    """

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        use_llm: bool = True,
        fallback_to_rules: bool = True,
        max_subqueries: int = 6,
        max_keywords: int = 20,
    ) -> None:
        self.use_llm = use_llm
        self.fallback_to_rules = fallback_to_rules
        self.llm_client = (llm_client or DeepSeekClient()) if use_llm else None

        # 三个组件共享同一个 DeepSeekClient，避免重复初始化连接和配置。
        self.parser = QueryParser(self.llm_client)
        self.decomposer = QueryDecomposer(self.llm_client, max_subqueries=max_subqueries)
        self.rewriter = QueryRewriter(self.llm_client, max_keywords=max_keywords)

    def run(self, query: str) -> QueryAnalysis:
        """执行完整查询理解流程并返回可 JSON 序列化的 QueryAnalysis。"""
        query = " ".join(query.strip().split())
        if not query:
            raise ValueError("query must not be empty")

        logger.info("查询理解流水线开始，字符数=%d", len(query))

        # 1. Parser：识别研究意图、实体及各类检索约束。
        if self.use_llm:
            try:
                analysis = self.parser.parse(query)
            except Exception:
                if not self.fallback_to_rules:
                    logger.exception("查询理解流水线在 Parser 阶段失败")
                    raise
                logger.warning("Parser 调用 DeepSeek 失败，切换到规则解析", exc_info=True)
                analysis = self.parser.parse_with_rules(query)
        else:
            analysis = self.parser.parse_with_rules(query)

        # 2. Decomposer：判断复杂度，必要时生成独立子查询。
        if self.use_llm:
            try:
                analysis = self.decomposer.decompose(analysis)
            except Exception:
                if not self.fallback_to_rules:
                    logger.exception("查询理解流水线在 Decomposer 阶段失败")
                    raise
                logger.warning("Decomposer 调用 DeepSeek 失败，切换到规则拆解", exc_info=True)
                analysis = self.decomposer.decompose_with_rules(analysis)
        else:
            analysis = self.decomposer.decompose_with_rules(analysis)

        # 3. Rewriter：生成适用于学术数据库的中英文关键词和扩展检索式。
        if self.use_llm:
            try:
                analysis = self.rewriter.rewrite(analysis)
            except Exception:
                if not self.fallback_to_rules:
                    logger.exception("查询理解流水线在 Rewriter 阶段失败")
                    raise
                logger.warning("Rewriter 调用 DeepSeek 失败，切换到规则改写", exc_info=True)
                analysis = self.rewriter.rewrite_with_rules(analysis)
        else:
            analysis = self.rewriter.rewrite_with_rules(analysis)

        logger.info(
            "查询理解流水线完成，实体数=%d，子查询数=%d，扩展查询数=%d，关键词数=%d",
            len(analysis.entities),
            len(analysis.sub_queries),
            len(analysis.expanded_queries),
            len(analysis.keywords),
        )
        return analysis

    # 为其他 Agent 提供语义更明确的同义入口。
    understand = run

