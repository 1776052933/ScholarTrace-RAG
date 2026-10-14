"""将科研子查询优化为适合论文数据库的搜索表达。"""

import logging
import re
from typing import Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .llm_client import DeepSeekClient
from .prompts import QUERY_OPTIMIZE_PROMPT
from .schema import SubQuery


logger = logging.getLogger(__name__)


class _OptimizedQuery(BaseModel):
    """DeepSeek 只需返回三个搜索相关字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1)
    english_query: str = Field(min_length=1)
    keywords: List[str] = Field(min_length=1)


class _OptimizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_queries: List[_OptimizedQuery] = Field(min_length=1)


class SubQueryOptimizer:
    """生成规范查询、英文查询和后续论文检索关键词。"""

    def __init__(self, llm_client: Optional[DeepSeekClient] = None):
        self.llm_client = llm_client

    def optimize(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """使用 DeepSeek 优化查询，并保留原子查询的类型、目的和优先级。"""
        if not sub_queries:
            return []
        if self.llm_client is None:
            return self.optimize_with_rules(sub_queries)

        payload = self.llm_client.complete_json(
            QUERY_OPTIMIZE_PROMPT,
            __import__("json").dumps(
                [{"query": item.query} for item in sub_queries], ensure_ascii=False
            ),
            temperature=0.1,
        )
        response = _OptimizationResponse.model_validate(payload)
        if len(response.sub_queries) != len(sub_queries):
            raise ValueError("DeepSeek optimizer changed the number of sub-queries")

        return [
            original.model_copy(update={
                "query": _clean_query_text(optimized.query),
                "english_query": _clean_english_query(optimized.english_query),
                "keywords": _dedupe(optimized.keywords),
            })
            for original, optimized in zip(sub_queries, response.sub_queries)
        ]

    def optimize_with_rules(self, sub_queries: List[SubQuery]) -> List[SubQuery]:
        """使用内置科研术语表完成确定性离线优化。"""
        result = []
        for item in sub_queries:
            query = _clean_query_text(item.query)
            english_query = _to_english_query(query)
            result.append(item.model_copy(update={
                "query": query,
                "english_query": english_query,
                "keywords": _search_keywords(query, english_query),
            }))
        logger.info("完成 %d 个子查询的搜索表达优化", len(result))
        return result


_TERM_MAP = {
    "医学图像分割": "medical image segmentation",
    "目标检测": "object detection",
    "图像分类": "image classification",
    "大语言模型": "large language model",
    "药物发现": "drug discovery",
    "图神经网络": "graph neural network",
    "卷积神经网络": "convolutional neural network",
    "经典方法": "classical methods",
    "性能表现": "performance",
    "性能比较": "performance comparison",
    "比较": "comparison",
    "方法": "methods",
    "模型": "models",
    "应用": "applications",
    "评价": "evaluation",
    "评估": "evaluation",
    "发展趋势": "research trends",
    "趋势": "trends",
    "综述": "survey",
    "研究": "research",
}


def _to_english_query(query: str) -> str:
    english = query
    # 长词优先替换，避免“趋势”先破坏“发展趋势”。
    for chinese in sorted(_TERM_MAP, key=len, reverse=True):
        english = english.replace(chinese, f" {_TERM_MAP[chinese]} ")
    english = re.sub(r"[\u4e00-\u9fff]+", " ", english)
    english = re.sub(r"\s+", " ", english).strip()

    # 将“基于 Transformer/Transformer 方法”改写为常见论文检索表达。
    if re.match(r"(?i)^Transformer\b", english) and not re.match(r"(?i)^Transformer based\b", english):
        english = re.sub(r"(?i)^Transformer\b", "Transformer based", english, count=1)
    if "methods" in english.casefold() and "survey" not in english.casefold():
        english += " survey"
    return english or query


def _clean_query_text(value: str) -> str:
    text = str(value or "").strip(" ，。;；")
    text = re.sub(r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_english_query(value: str) -> str:
    cleaned = _clean_query_text(value)
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return _to_english_query(cleaned)
    return cleaned


def _search_keywords(query: str, english_query: str) -> List[str]:
    values: List[str] = []
    for term in ("Transformer", "CNN", "DETR", "LLM", "GNN", "RAG"):
        if re.search(rf"\b{re.escape(term)}\b", english_query, re.I):
            values.append(term)
    for phrase in sorted(set(_TERM_MAP.values()), key=len, reverse=True):
        if phrase.casefold() in english_query.casefold():
            values.append(phrase)
    values.extend(re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", english_query))
    return _dedupe(values)


def _dedupe(values: Iterable[str]) -> List[str]:
    seen, result = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result
