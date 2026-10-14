"""论文精排 Agent 的统一执行流水线。"""

import logging
import math
import random
import re
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Dict, List, Optional

from .embedding import EmbeddingModel
from .evaluator import PaperEvaluator
from .feature import PaperFeatureExtractor
from .reranker import PaperReranker
from .schema import (
    Paper,
    PaperGraphEdge,
    PaperGraphPoint,
    PaperScore,
    QueryContext,
    RankingResult,
)
from .scorer import RankingScorer
from .similarity import SimilarityCalculator


logger = logging.getLogger(__name__)


class RankingAgent:
    """组合向量、LLM 评价、综合评分和重排序组件。"""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        evaluator: Optional[PaperEvaluator] = None,
        reranker: Optional[PaperReranker] = None,
        scorer: Optional[RankingScorer] = None,
        *,
        similarity_calculator: Optional[SimilarityCalculator] = None,
        feature_extractor: Optional[PaperFeatureExtractor] = None,
        evaluation_workers: int = 3,
        evaluation_batch_size: int = 5,
        llm_candidate_min: int = 30,
        llm_candidate_max: int = 50,
        llm_candidate_ratio: float = 0.30,
        highly_relevant_ratio: float = 0.30,
    ) -> None:
        if (
            isinstance(evaluation_workers, bool)
            or not isinstance(evaluation_workers, int)
            or evaluation_workers <= 0
        ):
            raise ValueError("evaluation_workers must be a positive integer")
        if (
            isinstance(evaluation_batch_size, bool)
            or not isinstance(evaluation_batch_size, int)
            or not 1 <= evaluation_batch_size <= 5
        ):
            raise ValueError("evaluation_batch_size must be between 1 and 5")
        if llm_candidate_min <= 0 or llm_candidate_max < llm_candidate_min:
            raise ValueError("LLM candidate limits must satisfy 0 < min <= max")
        if not 0.0 < llm_candidate_ratio <= 1.0:
            raise ValueError("llm_candidate_ratio must be between 0 and 1")
        if not 0.0 < highly_relevant_ratio <= 1.0:
            raise ValueError("highly_relevant_ratio must be between 0 and 1")
        self.embedding_model = embedding_model or EmbeddingModel()
        self.evaluator = evaluator or PaperEvaluator()
        self.reranker = reranker or PaperReranker()
        self.scorer = scorer or RankingScorer()
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        self.feature_extractor = feature_extractor or PaperFeatureExtractor()
        self.evaluation_workers = evaluation_workers
        self.evaluation_batch_size = evaluation_batch_size
        self.llm_candidate_min = llm_candidate_min
        self.llm_candidate_max = llm_candidate_max
        self.llm_candidate_ratio = llm_candidate_ratio
        self.highly_relevant_ratio = highly_relevant_ratio
        # RankingResult 按既定 Schema 只保存论文；分数可从此属性读取。
        self.last_scores: Dict[str, PaperScore] = {}
        self.last_llm_evaluated_ids: set[str] = set()

    def run(self, query_context: Any, papers: Any) -> RankingResult:
        """精排候选论文并返回 RankingResult。"""
        started_at = perf_counter()
        context = _query_context(query_context)
        candidates = self.reranker.deduplicate(_papers(papers))
        self.last_scores = {}
        self.last_llm_evaluated_ids = set()
        if not candidates:
            return RankingResult(papers=[], total=0, ranking_time=round(perf_counter() - started_at, 6))

        query_vector = self.embedding_model.encode_query(_query_text(context))
        paper_vectors = _paper_vectors(self.embedding_model, candidates)
        prepared = []
        for paper, paper_vector in zip(candidates, paper_vectors):
            semantic_score = self.similarity_calculator.calculate(query_vector, paper_vector)
            metadata_features = self.feature_extractor.extract(paper)
            quality_score = (
                metadata_features["citation_score"] * 0.40
                + metadata_features["venue_score"] * 0.35
                + metadata_features["abstract_score"] * 0.25
            )
            document = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}"
            method_score = _constraint_score(context.method_constraints, document, semantic_score)
            task_terms = [context.task_type, *context.domain_constraints]
            task_score = _constraint_score(task_terms, document, semantic_score)
            coarse_score = _coarse_score(
                semantic_score,
                method_score,
                task_score,
                quality_score,
                metadata_features["year_score"],
            )
            prepared.append({
                "paper": paper,
                "vector": paper_vector,
                "semantic_score": semantic_score,
                "metadata_features": metadata_features,
                "quality_score": quality_score,
                "method_score": method_score,
                "task_score": task_score,
                "coarse_score": coarse_score,
            })

        prepared.sort(
            key=lambda item: (
                item["coarse_score"],
                item["paper"].citation_count,
                item["paper"].year or 0,
            ),
            reverse=True,
        )
        evaluation_count = _evaluation_candidate_count(
            len(prepared),
            minimum=self.llm_candidate_min,
            maximum=self.llm_candidate_max,
            ratio=self.llm_candidate_ratio,
        )
        llm_candidates = [item["paper"] for item in prepared[:evaluation_count]]
        self.last_llm_evaluated_ids = {paper.paper_id for paper in llm_candidates}
        logger.info(
            "两阶段精排：去重候选=%d，DeepSeek 评价=%d",
            len(prepared),
            evaluation_count,
        )
        evaluations = _evaluate_candidates(
            self.evaluator,
            context,
            llm_candidates,
            max_workers=self.evaluation_workers,
            batch_size=self.evaluation_batch_size,
        )
        evaluation_by_id = {
            paper.paper_id: evaluation
            for paper, evaluation in zip(llm_candidates, evaluations)
        }
        for item in prepared:
            paper = item["paper"]
            semantic_score = item["semantic_score"]
            metadata_features = item["metadata_features"]
            quality_score = item["quality_score"]
            llm = evaluation_by_id.get(paper.paper_id)
            if llm is not None:
                method_score = float(llm["method_match"])
                task_score = float(llm["task_match"])
                llm_score = float(llm["relevance_score"])
            else:
                method_score = item["method_score"]
                task_score = item["task_score"]
                llm_score = semantic_score

            self.last_scores[paper.paper_id] = self.scorer.calculate_score({
                "semantic_score": semantic_score,
                "method_score": method_score,
                "task_score": task_score,
                "quality_score": quality_score,
                "time_score": metadata_features["year_score"],
                "llm_score": llm_score,
            })

        ranked = self.reranker.rerank(candidates, self.last_scores)
        threshold, highly_relevant_count = _apply_adaptive_relevance(
            ranked,
            self.last_scores,
            self.highly_relevant_ratio,
        )
        vector_by_id = {
            item["paper"].paper_id: item["vector"]
            for item in prepared
        }
        graph_points = _build_graph_points(
            ranked,
            self.last_scores,
            vector_by_id,
            query_vector,
            self.last_llm_evaluated_ids,
        )
        graph_edges = _build_graph_edges(ranked)
        return RankingResult(
            papers=ranked,
            total=len(ranked),
            ranking_time=round(perf_counter() - started_at, 6),
            scores=self.last_scores,
            graph_points=graph_points,
            graph_edges=graph_edges,
            llm_evaluated_count=evaluation_count,
            highly_relevant_count=highly_relevant_count,
            adaptive_threshold=threshold,
        )


def _coarse_score(
    semantic_score: float,
    method_score: float,
    task_score: float,
    quality_score: float,
    time_score: float,
) -> float:
    """BGE-M3 与确定性规则构成的一阶段粗排分数。"""
    return round(
        semantic_score * 0.55
        + method_score * 0.20
        + task_score * 0.15
        + quality_score * 0.07
        + time_score * 0.03,
        6,
    )


def _evaluation_candidate_count(
    total: int,
    *,
    minimum: int,
    maximum: int,
    ratio: float,
) -> int:
    if total <= 0:
        return 0
    desired = max(minimum, math.ceil(total * ratio))
    return min(total, maximum, desired)


def _apply_adaptive_relevance(
    ranked: List[Paper],
    scores: Dict[str, PaperScore],
    ratio: float,
) -> tuple[float, int]:
    """按本次候选集分布将前一定比例标为高度相关。"""
    if not ranked:
        return 0.0, 0
    high_count = max(1, math.ceil(len(ranked) * ratio))
    threshold = scores[ranked[high_count - 1].paper_id].final_score
    high_ids = {paper.paper_id for paper in ranked[:high_count]}
    for paper in ranked:
        score = scores[paper.paper_id]
        level = "highly_relevant" if paper.paper_id in high_ids else "partially_relevant"
        scores[paper.paper_id] = score.model_copy(update={"relevance_level": level})
    return round(threshold, 6), high_count


def _build_graph_points(
    ranked: List[Paper],
    scores: Dict[str, PaperScore],
    vector_by_id: Dict[str, List[float]],
    query_vector: List[float],
    llm_evaluated_ids: set[str],
) -> List[PaperGraphPoint]:
    directions = _project_directions(ranked, vector_by_id, query_vector)
    points = []
    for rank, (paper, direction) in enumerate(zip(ranked, directions), start=1):
        score = scores[paper.paper_id]
        distance = round(0.12 + (1.0 - score.final_score) * 1.88, 6)
        points.append(PaperGraphPoint(
            paper_id=paper.paper_id,
            title=paper.title,
            rank=rank,
            x=round(direction[0] * distance, 6),
            y=round(direction[1] * distance, 6),
            z=round(direction[2] * distance, 6),
            distance=distance,
            final_score=score.final_score,
            relevance_level=score.relevance_level,
            llm_evaluated=paper.paper_id in llm_evaluated_ids,
            citation_depth=paper.citation_depth,
            relation_keys=paper.relation_keys,
        ))
    return points


def _build_graph_edges(ranked: List[Paper]) -> List[PaperGraphEdge]:
    target_id = "__research_target__"
    key_to_paper_id: Dict[str, str] = {}
    for paper in ranked:
        for key in paper.relation_keys:
            key_to_paper_id.setdefault(key.casefold(), paper.paper_id)

    edges: List[PaperGraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for paper in ranked:
        resolved: list[tuple[str, str, int]] = []
        if paper.citation_depth > 0:
            relations = paper.citation_relations or [
                {"parent_id": parent_id, "direction": "citation_tracking"}
                for parent_id in paper.citation_parent_ids
            ]
            for relation in relations:
                parent_key = str(relation.get("parent_id") or "").casefold()
                parent_id = key_to_paper_id.get(parent_key)
                if parent_id and parent_id != paper.paper_id:
                    resolved.append((
                        parent_id,
                        str(relation.get("direction") or "citation_tracking"),
                        min(2, max(0, int(relation.get("depth") or paper.citation_depth))),
                    ))
        if not resolved:
            resolved = [(target_id, "search", paper.citation_depth)]
        for source_id, direction, depth in resolved:
            key = (source_id, paper.paper_id, direction)
            if key in seen:
                continue
            seen.add(key)
            edges.append(PaperGraphEdge(
                source_id=source_id,
                target_id=paper.paper_id,
                direction=direction,
                depth=depth,
            ))
    return edges


def _project_directions(
    papers: List[Paper],
    vector_by_id: Dict[str, List[float]],
    query_vector: List[float],
) -> List[tuple[float, float, float]]:
    """用轻量确定性随机投影保留向量方向差异，避免额外矩阵依赖。"""
    query = [float(value) for value in query_vector]
    directions = []
    for paper in papers:
        vector = [float(value) for value in vector_by_id[paper.paper_id]]
        if len(vector) != len(query):
            directions.append(_deterministic_direction(paper.paper_id))
            continue
        projected = [0.0, 0.0, 0.0]
        for index, (paper_value, query_value) in enumerate(zip(vector, query)):
            delta = paper_value - query_value
            code = (index * 2654435761 + 1013904223) & 0xFFFFFFFF
            projected[0] += delta if code & 1 else -delta
            projected[1] += delta if code & 2 else -delta
            projected[2] += delta if code & 4 else -delta
        norm = math.sqrt(sum(value * value for value in projected))
        if norm <= 1e-12:
            directions.append(_deterministic_direction(paper.paper_id))
        else:
            directions.append(tuple(value / norm for value in projected))
    return directions


def _deterministic_direction(identifier: str) -> tuple[float, float, float]:
    generator = random.Random(identifier)
    z = generator.uniform(-1.0, 1.0)
    angle = generator.uniform(0.0, math.tau)
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return radius * math.cos(angle), radius * math.sin(angle), z


def _paper_vectors(embedding_model: Any, papers: List[Paper]) -> List[List[float]]:
    batch_method = getattr(embedding_model, "encode_papers", None)
    if callable(batch_method):
        vectors = list(batch_method((paper.title, paper.abstract) for paper in papers))
    else:
        vectors = [
            embedding_model.encode_paper(paper.title, paper.abstract)
            for paper in papers
        ]
    if len(vectors) != len(papers):
        raise ValueError("embedding model returned an unexpected number of paper vectors")
    return vectors


def _evaluate_candidates(
    evaluator: Any,
    context: QueryContext,
    papers: List[Paper],
    *,
    max_workers: int,
    batch_size: int,
) -> List[Optional[Dict[str, Any]]]:
    batch_method = getattr(evaluator, "evaluate_batch", None)
    if callable(batch_method):
        batches = [
            papers[index:index + batch_size]
            for index in range(0, len(papers), batch_size)
        ]
        worker_count = min(max_workers, len(batches))
        if worker_count == 1:
            values = [_evaluate_batch(evaluator, context, batch) for batch in batches]
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="paper-batch-evaluator",
            ) as executor:
                futures = [
                    executor.submit(_evaluate_batch, evaluator, context, batch)
                    for batch in batches
                ]
                values = [future.result() for future in futures]
        return [evaluation for batch in values for evaluation in batch]

    worker_count = min(max_workers, len(papers))
    if worker_count == 1:
        return [_evaluate_one(evaluator, context, paper) for paper in papers]
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="paper-evaluator",
    ) as executor:
        futures = [
            executor.submit(_evaluate_one, evaluator, context, paper)
            for paper in papers
        ]
        return [future.result() for future in futures]


def _evaluate_batch(
    evaluator: Any,
    context: QueryContext,
    papers: List[Paper],
) -> List[Optional[Dict[str, Any]]]:
    try:
        values = list(evaluator.evaluate_batch(context, papers))
        if len(values) != len(papers):
            raise ValueError("batch evaluator returned an unexpected result count")
        return values
    except Exception:
        logger.warning(
            "批量 LLM 评价失败，改为逐篇补偿评价",
            exc_info=True,
        )
        return [_evaluate_one(evaluator, context, paper) for paper in papers]


def _evaluate_one(
    evaluator: Any,
    context: QueryContext,
    paper: Paper,
) -> Optional[Dict[str, Any]]:
    try:
        return evaluator.evaluate(context, paper)
    except Exception:
        # 单篇论文的 LLM 失败不应中断整个候选集精排。
        logger.warning(
            "论文 %s 的 LLM 评价失败，使用确定性评分回退",
            paper.paper_id,
            exc_info=True,
        )
        return None


def _query_context(value: Any) -> QueryContext:
    if isinstance(value, QueryContext):
        return value
    data = _mapping(value)
    selected = {
        "original_query": data.get("original_query") or data.get("query") or "",
        "research_intent": data.get("research_intent") or data.get("original_query") or data.get("query") or "",
        "entities": [_mapping(item) for item in data.get("entities") or []],
        "method_constraints": data.get("method_constraints") or [],
        "domain_constraints": data.get("domain_constraints") or [],
        "task_type": data.get("task_type") or "",
        "keywords": data.get("keywords") or [],
    }
    return QueryContext.model_validate(selected)


def _papers(value: Any) -> List[Paper]:
    if isinstance(value, Mapping) or hasattr(value, "model_dump"):
        container = _mapping(value)
        values = container.get("papers")
        if values is None:
            values = [container]
    else:
        values = value
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("papers must be an iterable or an object containing papers")
    return [_paper(item) for item in values]


def _paper(value: Any) -> Paper:
    if isinstance(value, Paper):
        return value
    data = _mapping(value)
    title = str(data.get("title") or "").strip()
    url = str(data.get("url") or "")
    pdf_url = str(data.get("pdf_url") or "")
    doi = _normalize_doi(data.get("doi")) or _extract_doi(url) or _extract_doi(pdf_url)
    arxiv_id = (
        str(data.get("arxiv_id") or "").strip()
        or _extract_arxiv_id(url)
        or _extract_arxiv_id(pdf_url)
        or None
    )
    openalex_id = (
        str(data.get("openalex_id") or "").strip()
        or _extract_openalex_id(url)
        or _extract_openalex_id(pdf_url)
        or None
    )
    raw_identifier = str(data.get("paper_id") or "").strip()
    exact_identifier = raw_identifier if (
        raw_identifier.startswith(("http://", "https://", "10."))
    ) else ""
    identifier = (
        pdf_url
        or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
        or exact_identifier
        or doi
        or url
        or raw_identifier
        or _stable_title_id(title)
    )
    authors = data.get("authors") or []
    authors = [
        str(author.get("name") or "").strip() if isinstance(author, Mapping) else str(author).strip()
        for author in authors
    ]
    relation_keys = _relation_keys(title, doi, arxiv_id, openalex_id)
    return Paper(
        paper_id=str(identifier),
        title=title,
        abstract=str(data.get("abstract") or ""),
        authors=[author for author in authors if author],
        year=data.get("year"),
        venue=str(data.get("venue") or ""),
        citation_count=int(data.get("citation_count") or 0),
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        pdf_url=pdf_url,
        source=str(data.get("source") or ""),
        sources=[str(item) for item in data.get("sources") or [] if str(item).strip()],
        keywords=[str(item) for item in data.get("keywords") or [] if str(item).strip()],
        citation_parent_ids=[
            str(item) for item in data.get("citation_parent_ids") or [] if str(item).strip()
        ],
        citation_directions=[
            str(item) for item in data.get("citation_directions") or [] if str(item).strip()
        ],
        citation_relations=[
            dict(item) for item in data.get("citation_relations") or []
            if isinstance(item, Mapping)
        ],
        citation_depth=min(2, max(0, int(data.get("citation_depth") or 0))),
        relation_keys=relation_keys,
    )


def _mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    raise TypeError("value must be a mapping or Pydantic model")


def _query_text(context: QueryContext) -> str:
    entity_names = [str(entity.get("name") or "") for entity in context.entities]
    return " ".join(filter(None, [
        context.original_query,
        context.research_intent,
        *entity_names,
        *context.method_constraints,
        *context.domain_constraints,
        context.task_type,
        *context.keywords,
    ]))


def _constraint_score(terms: Iterable[str], document: str, fallback: float) -> float:
    terms = [str(term).strip().casefold() for term in terms if str(term).strip()]
    if not terms:
        return fallback
    text = document.casefold()
    return sum(term in text for term in terms) / len(terms)


def _stable_title_id(title: str) -> str:
    normalized = "".join(character.casefold() for character in title if character.isalnum())
    return f"title:{normalized}"


def _normalize_doi(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I)


def _extract_doi(value: str) -> Optional[str]:
    match = re.search(r"(?i)(10\.\d{4,9}/[^\s\"'<>?#]+)", value or "")
    return match.group(1).rstrip(".,;:)]}").casefold() if match else None


def _extract_arxiv_id(value: str) -> Optional[str]:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None


def _extract_openalex_id(value: str) -> Optional[str]:
    match = re.search(r"(?i)(?:openalex\.org/)?(W\d+)", value or "")
    return match.group(1).upper() if match else None


def _relation_keys(
    title: str,
    doi: Optional[str],
    arxiv_id: Optional[str],
    openalex_id: Optional[str],
) -> List[str]:
    keys = []
    if doi:
        keys.append(f"doi:{_normalize_doi(doi).casefold()}")
    if arxiv_id:
        clean_arxiv = re.sub(r"(?i)v\d+$", "", arxiv_id).casefold()
        keys.append(f"arxiv:{clean_arxiv}")
    if openalex_id:
        keys.append(f"openalex:{openalex_id.casefold()}")
    keys.append(_stable_title_id(title))
    return list(dict.fromkeys(keys))
