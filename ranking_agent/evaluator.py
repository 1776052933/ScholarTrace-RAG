"""使用 DeepSeek 评价候选论文与科研查询的匹配程度。"""

import json
import logging
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cache import EvaluationCache
from .llm_client import DeepSeekClient
from .prompts import PAPER_BATCH_EVALUATION_PROMPT, PAPER_EVALUATION_PROMPT
from .schema import Paper, QueryContext


logger = logging.getLogger(__name__)


class PaperEvaluationJSONError(ValueError):
    """DeepSeek 输出无法解析或不符合评分结构。"""


class _EvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    relevance_score: float = Field(ge=0.0, le=1.0)
    method_match: float = Field(ge=0.0, le=1.0)
    task_match: float = Field(ge=0.0, le=1.0)
    reason: str


class _BatchEvaluationItem(_EvaluationResponse):
    item_id: str = Field(min_length=1)


class _BatchEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluations: List[Dict[str, Any]]


class PaperEvaluator:
    """调用 DeepSeek 对单篇论文进行相关性、方法和任务匹配评价。"""

    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        *,
        cache: Optional[EvaluationCache] = None,
    ) -> None:
        self.llm_client = llm_client
        # 依赖注入客户端多用于测试，此时仅在显式传入 cache 时启用缓存。
        self.cache = cache if cache is not None else (
            EvaluationCache() if llm_client is None else None
        )
        self._client_lock = Lock()

    def evaluate(self, query_context: QueryContext, paper: Paper) -> Dict[str, Any]:
        """返回经过 Pydantic 校验的 LLM 评分字典。"""
        if not isinstance(query_context, QueryContext):
            raise TypeError("query_context must be a QueryContext")
        if not isinstance(paper, Paper):
            raise TypeError("paper must be a Paper")

        model = (
            str(getattr(self.llm_client, "model", "") or "").strip()
            or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        )
        if self.cache is not None:
            cached = self.cache.get(query_context, paper, model=model)
            if cached is not None:
                return cached

        client = self._get_client()
        payload = {
            "query_context": query_context.model_dump(mode="json"),
            "paper": paper.model_dump(mode="json"),
        }
        logger.info("调用 DeepSeek 评价论文：%s", paper.paper_id)
        content = client.chat(
            [
                {"role": "system", "content": PAPER_EVALUATION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        try:
            data = _extract_json(content)
            evaluation = _EvaluationResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            logger.warning("论文 %s 的 LLM 评分 JSON 无效", paper.paper_id)
            raise PaperEvaluationJSONError(
                f"invalid DeepSeek evaluation JSON for paper {paper.paper_id}"
            ) from exc
        result = evaluation.model_dump(mode="json")
        if self.cache is not None:
            self.cache.put(query_context, paper, result, model=client.model)
        return result

    def evaluate_batch(
        self,
        query_context: QueryContext,
        papers: List[Paper],
    ) -> List[Dict[str, Any]]:
        """在一次 DeepSeek 请求中独立评价最多五篇论文。"""
        if not isinstance(query_context, QueryContext):
            raise TypeError("query_context must be a QueryContext")
        if not isinstance(papers, list) or not papers:
            return []
        if len(papers) > 5:
            raise ValueError("a DeepSeek evaluation batch may contain at most five papers")
        if any(not isinstance(paper, Paper) for paper in papers):
            raise TypeError("papers must contain only Paper instances")

        model = (
            str(getattr(self.llm_client, "model", "") or "").strip()
            or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        )
        results: List[Optional[Dict[str, Any]]] = [None] * len(papers)
        missing: List[tuple[int, Paper, str]] = []
        for index, paper in enumerate(papers):
            cached = self.cache.get(query_context, paper, model=model) if self.cache else None
            if cached is not None:
                results[index] = cached
            else:
                missing.append((index, paper, f"P{index + 1}"))

        if missing:
            client = self._get_client()
            payload = {
                "query_context": query_context.model_dump(mode="json"),
                "papers": [
                    {
                        "item_id": item_id,
                        "paper": paper.model_dump(mode="json"),
                    }
                    for _, paper, item_id in missing
                ],
            }
            logger.info("调用 DeepSeek 批量评价论文：%d 篇", len(missing))
            try:
                content = client.chat(
                    [
                        {"role": "system", "content": PAPER_BATCH_EVALUATION_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                    max_tokens=max(900, len(missing) * 500),
                )
                data = _BatchEvaluationResponse.model_validate(_extract_json(content))
                by_item_id: Dict[str, Dict[str, Any]] = {}
                for raw_evaluation in data.evaluations:
                    try:
                        evaluation = _BatchEvaluationItem.model_validate(raw_evaluation)
                    except ValidationError:
                        logger.warning("忽略 DeepSeek 批量评分中的无效条目")
                        continue
                    if evaluation.item_id in by_item_id:
                        logger.warning("忽略 DeepSeek 批量评分中的重复条目：%s", evaluation.item_id)
                        continue
                    value = evaluation.model_dump(mode="json")
                    value.pop("item_id", None)
                    by_item_id[evaluation.item_id] = value

                for index, paper, item_id in missing:
                    evaluation = by_item_id.get(item_id)
                    if evaluation is None:
                        continue
                    results[index] = evaluation
                    if self.cache is not None:
                        self.cache.put(query_context, paper, evaluation, model=client.model)
            except (PaperEvaluationJSONError, ValidationError, TypeError, ValueError):
                logger.warning("DeepSeek 批量评分无效，将仅重试未完成论文", exc_info=True)

        # 批量返回漏项或格式异常时，只对对应论文执行单篇补偿请求。
        for index, paper in enumerate(papers):
            if results[index] is None:
                results[index] = self.evaluate(query_context, paper)
        return [dict(value) for value in results if value is not None]

    def _get_client(self) -> DeepSeekClient:
        client = self.llm_client
        if client is not None:
            return client
        with self._client_lock:
            if self.llm_client is None:
                self.llm_client = DeepSeekClient()
            return self.llm_client


def _extract_json(content: str) -> Dict[str, Any]:
    """从纯 JSON、Markdown 代码块或说明文字中提取首个 JSON 对象。"""
    if not isinstance(content, str) or not content.strip():
        raise PaperEvaluationJSONError("DeepSeek returned empty evaluation content")
    text = content.strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PaperEvaluationJSONError("no valid JSON object found in DeepSeek response")
