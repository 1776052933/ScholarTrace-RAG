"""为 Semantic Scholar、arXiv 和 OpenAlex 生成论文检索表达。"""

import logging
import re
from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .llm_client import DeepSeekClient
from .prompts import QUERY_REWRITE_PROMPT
from .query_parser import _extract_json_object
from .schema import QueryAnalysis


logger = logging.getLogger(__name__)


class _RewriteResponse(BaseModel):
    """DeepSeek 查询改写响应的内部校验模型。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expanded_queries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    english_keywords: List[str] = Field(default_factory=list)


class QueryRewriter:
    """扩展科研概念、英文标准术语和适用于论文数据库的检索式。"""

    def __init__(self, llm_client: Optional[DeepSeekClient] = None, max_keywords: int = 20):
        if max_keywords < 1:
            raise ValueError("max_keywords must be greater than zero")
        self.llm_client = llm_client
        self.max_keywords = max_keywords

    def rewrite(self, analysis: QueryAnalysis) -> QueryAnalysis:
        """调用 DeepSeek，并更新 ``expanded_queries`` 与 ``keywords``。

        ``keywords`` 最终会同时包含中文关键词、标准英文术语、缩写和常用同义词，
        便于下游检索 Agent 根据目标数据源选择或组合。
        """
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("analysis must be a QueryAnalysis instance")

        logger.info("开始生成论文检索关键词，已有实体数=%d", len(analysis.entities))
        client = self.llm_client or DeepSeekClient()
        self.llm_client = client
        user_content = (
            f"关键词总数最多为 {self.max_keywords} 个。目标检索源为 Semantic Scholar、"
            f"arXiv 和 OpenAlex。\n查询分析如下：\n{analysis.to_json()}"
        )
        try:
            response_text = client.chat(
                [
                    {"role": "system", "content": QUERY_REWRITE_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            )
            payload, repaired = _extract_json_object(response_text)
            if repaired:
                logger.warning("DeepSeek 改写响应包含非 JSON 包装，已自动提取")
            response = _RewriteResponse.model_validate(payload)
        except Exception:
            logger.exception("论文检索查询改写失败")
            raise

        analysis.expanded_queries = _dedupe(response.expanded_queries)
        analysis.keywords = _dedupe([*response.keywords, *response.english_keywords])[: self.max_keywords]
        logger.info(
            "论文检索查询改写完成，扩展查询数=%d，关键词数=%d",
            len(analysis.expanded_queries),
            len(analysis.keywords),
        )
        return analysis

    def rewrite_with_rules(self, analysis: QueryAnalysis) -> QueryAnalysis:
        """使用内置术语表生成确定性扩展，供离线模式和 API 失败回退。"""
        if not isinstance(analysis, QueryAnalysis):
            raise TypeError("analysis must be a QueryAnalysis instance")

        concepts = [entity.name for entity in analysis.entities]
        concepts += analysis.method_constraints
        concepts += analysis.domain_constraints
        concepts += analysis.dataset_constraints
        concepts += _tokens(analysis.original_query)
        concepts = _dedupe(concepts)

        expansions: List[str] = []
        for chinese, synonyms in _TERM_EXPANSIONS.items():
            if chinese in analysis.original_query or chinese in concepts:
                if chinese not in concepts:
                    concepts.append(chinese)
                expansions.extend(synonyms)

        keywords = _dedupe([*concepts, *expansions])[: self.max_keywords]
        english_terms = [item for item in keywords if re.search(r"[A-Za-z]", item)]
        expanded_queries = [analysis.original_query]
        expanded_queries.extend(analysis.sub_queries)
        if english_terms:
            expanded_queries.append(" ".join(english_terms))

        analysis.keywords = keywords
        analysis.expanded_queries = _dedupe(expanded_queries)
        logger.info(
            "规则改写完成，扩展查询数=%d，关键词数=%d",
            len(analysis.expanded_queries),
            len(analysis.keywords),
        )
        return analysis


# 高频科研概念的离线扩展。DeepSeek 模式不受此表限制，可生成更丰富的领域术语。
_TERM_EXPANSIONS = {
    "大语言模型": ["Large Language Model", "LLM", "Foundation Model"],
    "图神经网络": ["Graph Neural Network", "GNN", "Graph Representation Learning"],
    "卷积神经网络": ["Convolutional Neural Network", "CNN"],
    "医学图像分割": ["Medical Image Segmentation", "Semantic Segmentation"],
    "目标检测": ["Object Detection"],
    "强化学习": ["Reinforcement Learning", "RL"],
    "检索增强生成": ["Retrieval-Augmented Generation", "RAG"],
}


def _tokens(query: str) -> List[str]:
    """提取可用于离线回退的英文词和连续中文概念。"""
    english = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", query)
    chinese = re.findall(r"[\u4e00-\u9fff]{2,12}", query)
    chinese_stop = {"请帮我", "查找", "搜索", "研究", "论文", "相关", "最新", "如何", "的应用"}
    english_stop = {"and", "for", "the", "with", "using", "compare", "papers", "since"}
    return [item for item in english if item.casefold() not in english_stop] + [item for item in chinese if item not in chinese_stop]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen, output = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            output.append(value)
    return output

