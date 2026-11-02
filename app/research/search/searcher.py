"""统一管理 Google Scholar、arXiv 和 OpenAlex 搜索。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import inspect
import logging
from typing import Dict, Iterable, List, Optional, Protocol

from .identifiers import (
    canonical_paper_key,
    enrich_paper_identifiers,
    normalize_doi,
    normalize_title,
    paper_identity_keys,
    unique_strings,
)
from .schema import Paper, PaperMetadata, SearchPlan, SearchResult
from .tools import ArxivTool, GoogleScholarTool, OpenAlexTool


logger = logging.getLogger(__name__)


class SearchTool(Protocol):
    source: str
    def search(
        self,
        query: str,
        limit: int = 10,
        publication_date_to: Optional[str] = None,
    ) -> List[PaperMetadata]: ...


class PaperSearcher:
    """对计划中的每条查询调用全部论文源，合并并去重结果。"""

    def __init__(
        self,
        tools: Optional[Dict[str, SearchTool]] = None,
        *,
        results_per_query: int = 10,
        max_workers: int = 9,
    ) -> None:
        self.tools = tools or {
            "google_scholar": GoogleScholarTool(),
            "arxiv": ArxivTool(),
            "openalex": OpenAlexTool(),
        }
        self.results_per_query = results_per_query
        self.max_workers = max_workers

    def search(
        self,
        plan: SearchPlan,
        *,
        publication_date_to: Optional[str] = None,
    ) -> SearchResult:
        """执行 SearchPlan，返回带查询来源信息的统一 SearchResult。"""
        jobs = [
            (query.query, source_name, tool)
            for query in plan.queries
            for source_name, tool in self.tools.items()
        ]
        query_sources = {
            query.query: list(self.tools.keys())
            for query in plan.queries
        }
        papers: List[Paper] = []
        if jobs:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as executor:
                futures = {
                    executor.submit(
                        _search_tool,
                        tool,
                        query_text,
                        self.results_per_query,
                        publication_date_to,
                    ): (query_text, source_name)
                    for query_text, source_name, tool in jobs
                }
                for future in as_completed(futures):
                    query_text, source_name = futures[future]
                    try:
                        for item in future.result():
                            if not _within_publication_cutoff(item, publication_date_to):
                                continue
                            papers.append(to_internal_paper(item, query_text, source_name))
                    except Exception as exc:
                        # 工具本身已经降级为空列表；此处是防止第三方自定义工具中断 Agent。
                        logger.warning("论文源 %s 搜索失败：%s", source_name, exc)

        merged = deduplicate_papers(papers)
        used = "；".join(dict.fromkeys(query.query for query in plan.queries))
        return SearchResult(
            papers=merged,
            total=len(merged),
            query_used=used,
            query_sources=query_sources,
        )

def to_internal_paper(value: PaperMetadata, query: str = "", source: str = "") -> Paper:
    """把统一元数据补充为 Agent 内部可评分、可去重对象。"""
    value = enrich_paper_identifiers(value)
    actual_source = source or value.source
    data = value.model_dump()
    data["source"] = value.source or actual_source
    data["sources"] = _unique([*value.sources, actual_source])
    data["matched_queries"] = _unique([*value.matched_queries, query])
    if isinstance(value, Paper):
        data["paper_id"] = canonical_paper_key(value) or value.paper_id
        data["relevance_score"] = value.relevance_score
    else:
        data["paper_id"] = canonical_paper_key(value) or (
            value.doi
            or (f"https://arxiv.org/abs/{value.arxiv_id}" if value.arxiv_id else "")
            or (f"https://openalex.org/{value.openalex_id}" if value.openalex_id else "")
            or f"{actual_source}:{normalize_title(value.title)}"
        )
    return Paper(**data)


def deduplicate_papers(papers: Iterable[PaperMetadata]) -> List[Paper]:
    """Use all known identifiers plus normalized title and merge relation metadata."""
    merged: List[Paper | None] = []
    key_to_index: Dict[str, int] = {}
    for value in papers:
        enriched = enrich_paper_identifiers(value)
        paper = enriched if isinstance(enriched, Paper) else to_internal_paper(enriched)
        keys = paper_identity_keys(paper)
        indices = sorted({key_to_index[key] for key in keys if key in key_to_index})
        if not indices:
            index = len(merged)
            merged.append(paper)
            for key in keys:
                key_to_index[key] = index
            continue
        index = indices[0]
        existing = merged[index]
        assert existing is not None
        for duplicate_index in indices[1:]:
            duplicate = merged[duplicate_index]
            if duplicate is not None:
                existing = _merge_paper(existing, duplicate)
                merged[duplicate_index] = None
                for known_key, known_index in list(key_to_index.items()):
                    if known_index == duplicate_index:
                        key_to_index[known_key] = index
        existing = _merge_paper(existing, paper)
        merged[index] = existing
        for key in paper_identity_keys(existing):
            key_to_index[key] = index
    result = [paper for paper in merged if paper is not None]
    for paper in result:
        paper.paper_id = canonical_paper_key(paper) or paper.paper_id
    return result


def _merge_paper(existing: Paper, paper: Paper) -> Paper:
    existing.matched_queries = unique_strings([*existing.matched_queries, *paper.matched_queries])
    existing.sources = unique_strings([*existing.sources, *paper.sources, paper.source])
    existing.authors = unique_strings([*existing.authors, *paper.authors])
    if (paper.citation_count or 0) > (existing.citation_count or 0):
        existing.citation_count = paper.citation_count
    if not existing.abstract and paper.abstract:
        existing.abstract = paper.abstract
    if not existing.doi and paper.doi:
        existing.doi = paper.doi
    if not existing.arxiv_id and paper.arxiv_id:
        existing.arxiv_id = paper.arxiv_id
    if not existing.openalex_id and paper.openalex_id:
        existing.openalex_id = paper.openalex_id
    existing.citation_seed_ids = unique_strings([
        *existing.citation_seed_ids,
        *paper.citation_seed_ids,
    ])
    existing.citation_directions = unique_strings([
        *existing.citation_directions,
        *paper.citation_directions,
    ])
    existing.citation_parent_ids = unique_strings([
        *existing.citation_parent_ids,
        *paper.citation_parent_ids,
    ])
    existing.citation_relations = _unique_relations([
        *existing.citation_relations,
        *paper.citation_relations,
    ])
    existing.citation_depth = min(existing.citation_depth, paper.citation_depth)
    if not existing.publication_date and paper.publication_date:
        existing.publication_date = paper.publication_date
    if not existing.pdf_url and paper.pdf_url:
        existing.pdf_url = paper.pdf_url
    if not existing.url and paper.url:
        existing.url = paper.url
    return existing


def _unique_relations(values):
    result, seen = [], set()
    for value in values:
        if not isinstance(value, dict):
            continue
        key = (
            str(value.get("parent_id") or "").casefold(),
            str(value.get("direction") or "").casefold(),
            int(value.get("depth") or 0),
        )
        if key[0] and key not in seen:
            seen.add(key)
            result.append({
                "parent_id": str(value.get("parent_id") or ""),
                "direction": str(value.get("direction") or ""),
                "depth": int(value.get("depth") or 0),
            })
    return result


def _normalize_doi(value: Optional[str]) -> str:
    return normalize_doi(value) or ""


def _normal_title(value: str) -> str:
    return normalize_title(value)


def _unique(values):
    seen, result = set(), []
    for value in values:
        value = str(value).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            result.append(value)
    return result


def _search_tool(
    tool: SearchTool,
    query: str,
    limit: int,
    publication_date_to: Optional[str],
) -> List[PaperMetadata]:
    if publication_date_to and _accepts_publication_cutoff(tool.search):
        return tool.search(query, limit, publication_date_to=publication_date_to)
    return tool.search(query, limit)


def _accepts_publication_cutoff(method) -> bool:
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "publication_date_to"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _within_publication_cutoff(
    paper: PaperMetadata,
    publication_date_to: Optional[str],
) -> bool:
    if not publication_date_to:
        return True
    try:
        cutoff = date.fromisoformat(publication_date_to)
    except ValueError:
        return True
    if paper.publication_date:
        try:
            return date.fromisoformat(paper.publication_date[:10]) <= cutoff
        except ValueError:
            pass
    return paper.year is None or paper.year <= cutoff.year
