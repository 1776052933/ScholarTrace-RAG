"""RankingAgent 的可重复运行示例。

运行方式：
    python -m ranking_agent.examples
或：
    python ranking_agent/examples.py
"""

import json
import math
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ranking_agent.pipeline import RankingAgent
    from ranking_agent.schema import Paper, QueryContext
else:
    from .pipeline import RankingAgent
    from .schema import Paper, QueryContext


QUERY_CONTEXT = QueryContext(
    original_query="近五年Transformer医学图像分割方法",
    research_intent="研究近五年基于Transformer的医学图像分割方法",
    entities=[
        {"name": "Transformer", "type": "method"},
        {"name": "Medical Image Segmentation", "type": "task"},
    ],
    method_constraints=["Transformer"],
    domain_constraints=["医学图像", "medical image"],
    task_type="segmentation",
    keywords=["Transformer", "medical image segmentation", "Swin UNETR"],
)


PAPERS = [
    Paper(
        paper_id="paper-1",
        title="Swin UNETR: Transformers for Semantic Segmentation of Brain Tumors in MRI Images",
        abstract=(
            "This paper presents Swin UNETR, a Transformer-based architecture for "
            "three-dimensional medical image segmentation. Experiments on MRI brain "
            "tumor datasets evaluate Dice score, robustness, and segmentation accuracy. "
        ) * 4,
        authors=["Author A", "Author B"],
        year=2025,
        venue="MICCAI",
        citation_count=120,
        url="https://example.org/swin-unetr",
        keywords=["Swin UNETR", "Transformer", "MRI", "medical segmentation"],
    ),
    Paper(
        paper_id="paper-2",
        title="CNN Methods for Medical Image Segmentation: A Review",
        abstract=(
            "This review analyzes convolutional neural network methods for medical "
            "image segmentation, including U-Net variants, public datasets, Dice score, "
            "evaluation protocols, limitations, and clinical applications. "
        ) * 4,
        authors=["Author C"],
        year=2022,
        venue="Medical Image Analysis",
        citation_count=150,
        url="https://example.org/cnn-review",
        keywords=["CNN", "medical image segmentation", "review"],
    ),
    Paper(
        paper_id="paper-3",
        title="Transformer Models for Object Detection",
        abstract=(
            "This paper studies Transformer architectures for general object detection. "
            "Experiments compare detection accuracy and inference speed on natural image "
            "benchmarks, without medical image segmentation experiments. "
        ) * 4,
        authors=["Author D"],
        year=2024,
        venue="CVPR",
        citation_count=30,
        url="https://example.org/transformer-detection",
        keywords=["Transformer", "object detection", "DETR"],
    ),
]


class ExampleEmbeddingModel:
    """固定余弦相似度，保证示例不需要下载模型且结果可复现。"""

    _similarities = {
        "paper-1": 0.95,
        "paper-2": 0.70,
        "paper-3": 0.45,
    }

    def encode_query(self, query: str):
        return [1.0, 0.0]

    def encode_paper(self, title: str, abstract: str):
        paper_id = (
            "paper-1" if "Swin UNETR" in title
            else "paper-2" if "CNN Methods" in title
            else "paper-3"
        )
        similarity = self._similarities[paper_id]
        return [similarity, math.sqrt(1.0 - similarity * similarity)]


class ExamplePaperEvaluator:
    """固定 DeepSeek 评价结果，仅用于可重复的 Pipeline 集成示例。"""

    _scores = {
        "paper-1": (0.95, 0.95, 0.95),
        "paper-2": (0.68, 0.35, 0.90),
        "paper-3": (0.35, 0.85, 0.15),
    }

    def evaluate(self, query_context: QueryContext, paper: Paper):
        relevance, method, task = self._scores[paper.paper_id]
        return {
            "relevance_score": relevance,
            "method_match": method,
            "task_match": task,
            "reason": "用于验证精排流程的确定性评分",
        }


def run_example() -> None:
    agent = RankingAgent(
        embedding_model=ExampleEmbeddingModel(),
        evaluator=ExamplePaperEvaluator(),
    )
    result = agent.run(QUERY_CONTEXT, PAPERS)

    assert agent.last_scores["paper-1"].relevance_level == "highly_relevant"
    assert agent.last_scores["paper-2"].relevance_level == "partially_relevant"
    assert agent.last_scores["paper-3"].relevance_level == "partially_relevant"
    assert [paper.paper_id for paper in result.papers] == ["paper-1", "paper-2", "paper-3"]

    output = {
        "query": QUERY_CONTEXT.original_query,
        "total": result.total,
        "ranking_time": result.ranking_time,
        "ranked_papers": [
            {
                "rank": rank,
                "paper": paper.model_dump(mode="json"),
                "score": agent.last_scores[paper.paper_id].model_dump(mode="json"),
            }
            for rank, paper in enumerate(result.papers, start=1)
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_example()
