from __future__ import annotations

from threading import Lock
from time import sleep

from ranking_agent.pipeline import RankingAgent


class BatchEmbedding:
    def __init__(self) -> None:
        self.batch_calls = 0

    def encode_query(self, query: str):
        return [1.0, 0.0]

    def encode_papers(self, papers):
        self.batch_calls += 1
        values = list(papers)
        return [
            [1.0, index / 10.0]
            for index, _ in enumerate(values)
        ]

    def encode_paper(self, title: str, abstract: str):
        raise AssertionError("the ranking pipeline should use the batch API")


class ConcurrentEvaluator:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def evaluate(self, context, paper):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep(0.04)
            index = int(paper.paper_id.rsplit("-", 1)[-1])
            score = 0.65 + index * 0.05
            return {
                "relevance_score": score,
                "method_match": score,
                "task_match": score,
                "reason": "deterministic test score",
            }
        finally:
            with self.lock:
                self.active -= 1


class CountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = Lock()

    def evaluate(self, context, paper):
        with self.lock:
            self.calls += 1
        return {
            "relevance_score": 0.8,
            "method_match": 0.8,
            "task_match": 0.8,
            "reason": "candidate cutoff test",
        }


class BatchEvaluator:
    def __init__(self) -> None:
        self.batch_sizes = []
        self.lock = Lock()

    def evaluate_batch(self, context, papers):
        with self.lock:
            self.batch_sizes.append(len(papers))
        return [
            {
                "relevance_score": 0.8,
                "method_match": 0.8,
                "task_match": 0.8,
                "reason": "batch score",
            }
            for _ in papers
        ]


def _context():
    return {
        "original_query": "medical image segmentation",
        "research_intent": "find segmentation methods",
        "entities": [],
        "method_constraints": ["transformer"],
        "domain_constraints": ["medical image"],
        "task_type": "segmentation",
        "keywords": ["transformer", "segmentation"],
    }


def _papers():
    return [
        {
            "paper_id": f"paper-{index}",
            "title": f"Transformer paper {index}",
            "abstract": "Transformer method for medical image segmentation.",
            "year": 2025,
            "citation_count": index,
        }
        for index in range(4)
    ]


def test_ranking_batches_embeddings_and_evaluates_concurrently():
    embedding = BatchEmbedding()
    evaluator = ConcurrentEvaluator()
    result = RankingAgent(
        embedding_model=embedding,
        evaluator=evaluator,
        evaluation_workers=3,
    ).run(_context(), _papers())

    assert result.total == 4
    assert embedding.batch_calls == 1
    assert evaluator.max_active >= 2


def test_parallel_and_serial_evaluation_produce_the_same_ranking():
    serial = RankingAgent(
        embedding_model=BatchEmbedding(),
        evaluator=ConcurrentEvaluator(),
        evaluation_workers=1,
    )
    parallel = RankingAgent(
        embedding_model=BatchEmbedding(),
        evaluator=ConcurrentEvaluator(),
        evaluation_workers=3,
    )

    serial_result = serial.run(_context(), _papers())
    parallel_result = parallel.run(_context(), _papers())

    assert [paper.paper_id for paper in serial_result.papers] == [
        paper.paper_id for paper in parallel_result.papers
    ]
    assert {
        key: value.model_dump(mode="json")
        for key, value in serial.last_scores.items()
    } == {
        key: value.model_dump(mode="json")
        for key, value in parallel.last_scores.items()
    }


def test_two_stage_ranking_evaluates_only_high_potential_candidates():
    papers = [
        {
            "paper_id": f"paper-{index}",
            "title": f"Transformer paper {index}",
            "abstract": "Transformer method for medical image segmentation.",
            "year": 2025,
            "citation_count": index,
        }
        for index in range(100)
    ]
    evaluator = CountingEvaluator()
    agent = RankingAgent(
        embedding_model=BatchEmbedding(),
        evaluator=evaluator,
        evaluation_workers=4,
    )

    result = agent.run(_context(), papers)

    assert result.total == 100
    assert evaluator.calls == 30
    assert result.llm_evaluated_count == 30
    assert len(result.scores) == 100
    assert len(result.graph_points) == 100
    assert result.highly_relevant_count == 30
    assert sum(
        score.relevance_level == "highly_relevant"
        for score in result.scores.values()
    ) == 30


def test_production_batch_interface_groups_five_papers_per_request():
    papers = [
        {
            "paper_id": f"paper-{index}",
            "title": f"Transformer paper {index}",
            "abstract": "Transformer method for medical image segmentation.",
            "year": 2025,
            "citation_count": index,
        }
        for index in range(100)
    ]
    evaluator = BatchEvaluator()

    result = RankingAgent(
        embedding_model=BatchEmbedding(),
        evaluator=evaluator,
        evaluation_workers=3,
        evaluation_batch_size=5,
    ).run(_context(), papers)

    assert result.llm_evaluated_count == 30
    assert sorted(evaluator.batch_sizes) == [5, 5, 5, 5, 5, 5]


def test_graph_edges_preserve_two_hop_discovery_tree():
    papers = [
        {
            "paper_id": "10.test/root",
            "title": "Initial paper",
            "doi": "10.test/root",
            "abstract": "Transformer method for medical image segmentation.",
        },
        {
            "paper_id": "https://openalex.org/W2",
            "title": "First hop",
            "openalex_id": "W2",
            "abstract": "Transformer method for medical image segmentation.",
            "citation_depth": 1,
            "citation_parent_ids": ["doi:10.test/root"],
            "citation_relations": [{
                "parent_id": "doi:10.test/root",
                "direction": "reference",
                "depth": 1,
            }],
        },
        {
            "paper_id": "https://openalex.org/W3",
            "title": "Second hop",
            "openalex_id": "W3",
            "abstract": "Transformer method for medical image segmentation.",
            "citation_depth": 2,
            "citation_parent_ids": ["openalex:w2"],
            "citation_relations": [{
                "parent_id": "openalex:w2",
                "direction": "citation",
                "depth": 2,
            }],
        },
    ]
    result = RankingAgent(
        embedding_model=BatchEmbedding(),
        evaluator=CountingEvaluator(),
        evaluation_workers=1,
    ).run(_context(), papers)

    edges = {(edge.source_id, edge.target_id, edge.depth) for edge in result.graph_edges}
    assert ("__research_target__", "10.test/root", 0) in edges
    assert ("10.test/root", "https://openalex.org/W2", 1) in edges
    assert ("https://openalex.org/W2", "https://openalex.org/W3", 2) in edges
