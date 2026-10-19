"""Pure workflow data conversion and stage-report helpers, shared by services and graphs."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import logging
import re
from time import perf_counter
from typing import Any, Dict, Optional

from query_decomposition.schema import SubQuery as DecompositionSubQuery

logger = logging.getLogger("auto_paper")


def _timed(name: str, function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = perf_counter()
    logger.info("开始阶段：%s", name)
    result = function(*args, **kwargs)
    elapsed = _elapsed(started)
    logger.info("完成阶段：%s，耗时 %.3f 秒", name, elapsed)
    return result, elapsed


def _stage(result: Any, elapsed: float) -> Dict[str, Any]:
    return {
        "elapsed_seconds": elapsed,
        "result": _json_value(result),
    }


def _model_data(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    raise TypeError("pipeline result must be a mapping or Pydantic model")


def _replace_papers(value: Any, papers: list[Any]) -> Any:
    if hasattr(value, "model_copy"):
        return value.model_copy(update={"papers": papers})
    if isinstance(value, Mapping):
        updated = dict(value)
        updated["papers"] = papers
        if "total" in updated:
            updated["total"] = len(papers)
        return updated
    raise TypeError("search result must be a mapping or Pydantic model")


def _understanding_for_decomposition(value: Any) -> Dict[str, Any]:
    """适配查询理解与独立拆解模块间的可空字段差异。"""
    data = _model_data(value)
    data["task_type"] = data.get("task_type") or ""
    return data


def _merge_upstream_subqueries(decomposition: Any, understanding: Any) -> Any:
    """保留查询理解阶段已经生成、但独立拆解阶段未覆盖的检索方向。"""
    upstream = _model_data(understanding).get("sub_queries") or []
    if not upstream or not hasattr(decomposition, "model_copy"):
        return decomposition

    existing = list(_value(decomposition, "sub_queries") or [])
    seen = {_normalized_query(_value(item, "query")) for item in existing}
    keywords = [
        str(item).strip()
        for item in _model_data(understanding).get("keywords") or []
        if str(item).strip()
    ][:12]
    for raw_query in upstream:
        query = re.sub(
            r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])",
            " ",
            str(raw_query).strip(),
        )
        query = re.sub(r"\s+", " ", query).strip()
        normalized = _normalized_query(query)
        if not query or normalized in seen:
            continue
        existing.append(DecompositionSubQuery(
            query=query,
            purpose="补充查询理解阶段识别出的独立检索方向",
            query_type=_subquery_type(query),
            priority="high",
            keywords=keywords,
        ))
        seen.add(normalized)
    return decomposition.model_copy(update={"sub_queries": existing})


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = " ".join(value.strip().split())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _optional_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) == 8:
        raw = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise ValueError("publication_date_to must use YYYY-MM-DD format") from exc


def _subquery_type(query: str) -> str:
    if re.search(r"比较|对比|区别|compare|versus|\bvs\b", query, re.I):
        return "comparison"
    if re.search(r"趋势|发展|演进|trend|evolution", query, re.I):
        return "trend_analysis"
    if re.search(r"数据集|dataset|benchmark", query, re.I):
        return "dataset_search"
    if re.search(r"应用|场景|application|use case", query, re.I):
        return "application"
    if re.search(r"性能|评价|评估|指标|performance|evaluation|metric", query, re.I):
        return "evaluation"
    return "method_search"


def _normalized_query(value: Any) -> str:
    return re.sub(r"\W+", "", str(value or "").casefold())


def _elapsed(started: float) -> float:
    return round(perf_counter() - started, 6)
