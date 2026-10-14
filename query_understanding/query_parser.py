"""使用 DeepSeek 将自然语言科研问题解析为 QueryAnalysis。"""

from datetime import datetime
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError

from .llm_client import DeepSeekClient
from .prompts import QUERY_ANALYSIS_PROMPT
from .schema import Entity, QueryAnalysis, QueryConstraints, QueryIntent, ResearchEntity, TimeConstraint


logger = logging.getLogger(__name__)


class QueryParserError(RuntimeError):
    """查询解析失败的基础异常。"""


class QueryJSONDecodeError(QueryParserError):
    """DeepSeek 返回内容中无法提取有效 JSON 对象。"""


class QueryParser:
    """调用 DeepSeek 并使用 Pydantic 验证科研查询分析结果。"""

    def __init__(self, llm_client: Optional[DeepSeekClient] = None):
        # 延迟创建客户端，使模块导入和显式规则回退不要求 API Key。
        self.llm_client = llm_client

    def parse(self, query: str) -> QueryAnalysis:
        """解析用户查询并返回经过 Pydantic 校验的 QueryAnalysis。

        DeepSeek 偶尔会用 `````json`` 代码块包裹结果，或在 JSON 前后添加少量
        说明文字。本方法会自动提取其中第一个完整 JSON 对象，再执行严格校验。
        """
        query = " ".join(query.strip().split())
        if not query:
            raise ValueError("query must not be empty")

        logger.info("开始解析科研查询，字符数=%d", len(query))
        client = self.llm_client or DeepSeekClient()
        self.llm_client = client
        messages = [
            {"role": "system", "content": QUERY_ANALYSIS_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            response_text = client.chat(messages)
            payload, repaired = _extract_json_object(response_text)
            if repaired:
                logger.warning("DeepSeek 响应包含非 JSON 包装，已自动提取 JSON 对象")
            # original_query 由本地可信输入写入，不要求模型重复返回。
            payload["original_query"] = query
            analysis = QueryAnalysis.model_validate(payload)
        except QueryJSONDecodeError:
            logger.exception("科研查询解析失败：DeepSeek 响应不是有效 JSON")
            raise
        except ValidationError:
            logger.exception("科研查询解析失败：JSON 未通过 QueryAnalysis 校验")
            raise
        except Exception:
            logger.exception("科研查询解析失败：DeepSeek 调用异常")
            raise

        logger.info("科研查询解析完成，实体数=%d，关键词数=%d", len(analysis.entities), len(analysis.keywords))
        return analysis

    def parse_with_rules(self, query: str) -> QueryAnalysis:
        """显式使用本地规则解析，供无网络场景和 pipeline 失败回退使用。"""
        query = " ".join(query.strip().split())
        if not query:
            raise ValueError("query must not be empty")
        raw = self._rule_parse(query)
        constraints: QueryConstraints = raw["constraints"]
        allowed_types = {"method", "task", "domain", "dataset", "model", "application"}
        entities = [
            Entity(
                name=item.text,
                type=item.entity_type if item.entity_type in allowed_types else "task",
                importance="high" if index == 0 else "medium",
            )
            for index, item in enumerate(raw["entities"])
        ]
        return QueryAnalysis(
            original_query=query,
            research_intent=raw["intent"].value,
            entities=entities,
            method_constraints=constraints.methods,
            domain_constraints=constraints.domains,
            dataset_constraints=constraints.datasets,
            task_type=None,
            time_constraints=constraints.time,
            keywords=_dedupe(item.name for item in entities),
        )

    def _rule_parse(self, query: str) -> Dict[str, Any]:
        intent = _detect_intent(query)
        methods = _extract_after_markers(query, ["使用", "采用", "基于", "with", "using", "based on"])
        methods += _extract_compared_methods(query)
        methods = _dedupe(methods)
        domains = _extract_after_markers(query, ["领域", "场景", "domain", "in the field of"])
        datasets = _extract_datasets(query)
        time = _extract_time(query)
        entities = [ResearchEntity(text=x, entity_type="method") for x in methods]
        entities += [ResearchEntity(text=x, entity_type="domain") for x in domains]
        entities += [ResearchEntity(text=x, entity_type="dataset") for x in datasets]
        topic = _extract_topic(query)
        if topic:
            entities.insert(0, ResearchEntity(text=topic, entity_type="topic"))
        return {
            "normalized_query": query, "language": _detect_language(query), "intent": intent,
            "entities": _dedupe_entities(entities),
            "constraints": QueryConstraints(methods=methods, domains=domains, datasets=datasets, time=time),
            "complexity": _complexity(query), "confidence": 0.62,
        }


def _extract_json_object(response_text: str) -> tuple[Dict[str, Any], bool]:
    """从纯 JSON、Markdown 代码块或夹杂说明文字的响应中提取对象。"""
    if not isinstance(response_text, str) or not response_text.strip():
        raise QueryJSONDecodeError("DeepSeek returned an empty response")

    original = response_text.strip().lstrip("\ufeff")
    candidates = re.findall(r"```(?:json)?\s*([\s\S]*?)```", original, flags=re.IGNORECASE)
    candidates.append(original)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        candidate = candidate.strip()
        # 先尝试纯 JSON；失败后从每个左花括号开始使用 raw_decode，允许前后有说明文字。
        starts = [0] + [match.start() for match in re.finditer(r"\{", candidate) if match.start() != 0]
        for start in starts:
            try:
                value, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                trailing = candidate[start + end:].strip()
                repaired = start != 0 or bool(trailing) or candidate != original
                return value, repaired
    raise QueryJSONDecodeError("No valid JSON object found in DeepSeek response")


def _detect_intent(query: str) -> QueryIntent:
    q = query.lower()
    rules = [
        (QueryIntent.METHOD_COMPARISON, ["比较", "对比", "versus", " vs ", "compare"]),
        (QueryIntent.DATASET_DISCOVERY, ["数据集", "dataset", "benchmark"]),
        (QueryIntent.STATE_OF_THE_ART, ["最先进", "sota", "state of the art", "最佳性能"]),
        (QueryIntent.TREND_ANALYSIS, ["趋势", "演进", "trend", "evolution"]),
        (QueryIntent.REPLICATION, ["复现", "重现", "replicate", "reproduce"]),
        (QueryIntent.LITERATURE_REVIEW, ["综述", "文献回顾", "review", "survey"]),
        (QueryIntent.PROBLEM_SOLUTION, ["如何", "解决", "how to", "solution"]),
    ]
    return next((intent for intent, words in rules if any(w in q for w in words)), QueryIntent.GENERAL_SEARCH)


def _extract_after_markers(query: str, markers: Iterable[str]) -> List[str]:
    values = []
    stop = r"(?=，|。|；|;|、|\band\b|\bfor\b|\bin\b|在|用于|进行|的综述|近\d|20\d{2}|$)"
    for marker in markers:
        pattern = re.escape(marker) + r"\s*([\w.+#\-/\u4e00-\u9fff ]{2,50}?)" + stop
        values.extend(m.group(1).strip() for m in re.finditer(pattern, query, re.I))
    return _dedupe(values)


def _extract_datasets(query: str) -> List[str]:
    patterns = [
        r"(?:数据集|dataset)\s*[:：]?\s*([A-Za-z0-9_.+-]+)",
        r"([A-Za-z][A-Za-z0-9_.+-]{2,})\s*(?:数据集|dataset)",
        r"\b([A-Z][A-Za-z]*-?\d{1,4}[A-Za-z]*)\b",
    ]
    values = [m.group(1) for p in patterns for m in re.finditer(p, query, re.I if "dataset" in p else 0)]
    return _dedupe(values)


def _extract_compared_methods(query: str) -> List[str]:
    values: List[str] = []
    english = re.search(r"(?:compare|comparison of)\s+(.+?)\s+(?:methods?|models?|approaches?)\b", query, re.I)
    if english:
        values.extend(re.split(r"\s*(?:,|\band\b|\bvs\.?\b|\bversus\b)\s*", english.group(1), flags=re.I))
    chinese = re.search(r"(?:比较|对比)\s*([^，。；]+?)(?:方法|模型|算法)(?=，|。|；|在|用于|$)", query)
    if chinese and re.search(r"、|和|与|及|vs\.?", chinese.group(1), re.I):
        values.extend(re.split(r"\s*(?:、|和|与|及|vs\.?)\s*", chinese.group(1), flags=re.I))
    return _dedupe(values)


def _extract_time(query: str) -> Optional[TimeConstraint]:
    current = datetime.now().year
    recent = re.search(r"(?:近|最近|过去)\s*(\d{1,2})\s*年|(?:last|past)\s+(\d{1,2})\s+years?", query, re.I)
    if recent:
        n = int(recent.group(1) or recent.group(2))
        return TimeConstraint(start_year=current - n + 1, end_year=current, description=recent.group(0))
    span = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|–|—|至|到)\s*(19\d{2}|20\d{2})\b", query)
    if span:
        return TimeConstraint(start_year=int(span.group(1)), end_year=int(span.group(2)), description=span.group(0))
    after = re.search(r"(19\d{2}|20\d{2})\s*(?:年)?(?:以后|以来|之后)|(?:since|after)\s+(19\d{2}|20\d{2})", query, re.I)
    if after:
        year = int(after.group(1) or after.group(2))
        return TimeConstraint(start_year=year, end_year=current, description=after.group(0))
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", query)]
    return TimeConstraint(start_year=min(years), end_year=max(years), description=str(years[0])) if years else None


def _extract_topic(query: str) -> str:
    cleaned = re.sub(r"(?:请|帮我|查找|寻找|搜索|find|search for|papers? (?:on|about))\s*", "", query, flags=re.I)
    return cleaned[:120].strip(" ，。?？")


def _detect_language(query: str) -> str:
    zh = len(re.findall(r"[\u4e00-\u9fff]", query))
    en = len(re.findall(r"[A-Za-z]", query))
    return "mixed" if zh and en else "zh" if zh else "en" if en else "unknown"


def _complexity(query: str) -> str:
    connectors = len(re.findall(r"以及|并且|同时|分别|比较|且|\band\b|\bversus\b|\bvs\.?\b", query, re.I))
    return "complex" if connectors >= 2 or len(query) > 180 else "compound" if connectors == 1 or len(query) > 90 else "simple"


def _dedupe(values: Iterable[str]) -> List[str]:
    seen, output = set(), []
    for value in values:
        value = value.strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key); output.append(value)
    return output


def _dedupe_entities(values: Iterable[ResearchEntity]) -> List[ResearchEntity]:
    seen, output = set(), []
    for item in values:
        key = (item.text.casefold(), item.entity_type)
        if key not in seen:
            seen.add(key); output.append(item)
    return output


def _strings(value: Any) -> List[str]:
    return _dedupe(str(x) for x in value) if isinstance(value, list) else []


def _year(value: Any) -> Optional[int]:
    try:
        result = int(value)
        return result if 1800 <= result <= 2200 else None
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> Optional[int]:
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None
