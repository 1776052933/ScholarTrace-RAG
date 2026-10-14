"""query_decomposition 的三个端到端测试案例。"""

from pathlib import Path
import re
import sys
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from query_decomposition.pipeline import QueryDecompositionPipeline
    from query_decomposition.schema import DecompositionResult
else:
    from .pipeline import QueryDecompositionPipeline
    from .schema import DecompositionResult


CASE_1 = {
    "original_query": "近五年基于Transformer的医学图像分割方法",
    "research_intent": "系统分析Transformer医学图像分割的模型、数据集、评价指标、挑战和发展趋势",
    "entities": [
        {"name": "Transformer", "type": "model", "importance": "high"},
        {"name": "医学图像分割", "type": "task", "importance": "high"},
        {"name": "医学图像", "type": "domain", "importance": "high"},
    ],
    "method_constraints": ["Transformer"],
    "domain_constraints": ["医学图像"],
    "dataset_constraints": [],
    "task_type": "semantic_segmentation",
    "keywords": ["Transformer", "Medical Image Segmentation", "datasets", "evaluation metrics", "challenges", "trends"],
    "expanded_queries": ["Transformer medical image segmentation methods 2022 2026"],
}

CASE_2 = {
    "original_query": "比较CNN和Transformer在目标检测中的性能",
    "research_intent": "比较CNN和Transformer目标检测方法的性能",
    "entities": [
        {"name": "CNN", "type": "model", "importance": "high"},
        {"name": "Transformer", "type": "model", "importance": "high"},
        {"name": "目标检测", "type": "task", "importance": "high"},
    ],
    "method_constraints": ["CNN", "Transformer"],
    "domain_constraints": ["计算机视觉"],
    "dataset_constraints": [],
    "task_type": "object_detection",
    "keywords": ["CNN", "Transformer", "Object Detection", "performance comparison"],
    "expanded_queries": ["CNN versus Transformer object detection performance"],
}

CASE_3 = {
    "original_query": "LLM在药物发现中的应用",
    "research_intent": "研究大语言模型在药物发现中的应用",
    "entities": [
        {"name": "Large Language Model", "type": "model", "importance": "high"},
        {"name": "Drug Discovery", "type": "application", "importance": "high"},
    ],
    "method_constraints": ["LLM"],
    "domain_constraints": ["Drug Discovery"],
    "dataset_constraints": [],
    "task_type": "drug_discovery",
    "keywords": ["LLM", "Large Language Model", "Drug Discovery"],
    "expanded_queries": ["large language model drug discovery applications"],
}


def _print_result(name: str, result: DecompositionResult) -> None:
    print(f"\n===== {name} =====")
    print(result.to_json(indent=2))


def _all_text(result: DecompositionResult) -> str:
    return " ".join(
        f"{item.query} {item.english_query} {item.purpose} {' '.join(item.keywords)}"
        for item in result.sub_queries
    )


def run_examples() -> List[DecompositionResult]:
    """运行三个案例、执行要求检查并输出完整 JSON。"""
    pipeline = QueryDecompositionPipeline(use_llm=False)

    case1 = pipeline.run(CASE_1)
    case1_text = _all_text(case1)
    assert "Transformer" in case1_text
    assert re.search(r"dataset|benchmark|数据集", case1_text, re.I)
    assert re.search(r"evaluation|metric|评价|指标", case1_text, re.I)
    assert re.search(r"challenge|limitation|挑战|局限", case1_text, re.I)
    assert re.search(r"trend|evolution|趋势|发展", case1_text, re.I)
    _print_result("CASE 1", case1)

    case2 = pipeline.run(CASE_2)
    case2_text = _all_text(case2)
    assert len(case2.sub_queries) >= 3
    assert "CNN" in case2_text
    assert "Transformer" in case2_text
    assert any(item.query_type == "comparison" for item in case2.sub_queries)
    _print_result("CASE 2", case2)

    case3 = pipeline.run(CASE_3)
    assert case3.need_decompose is False or len(case3.sub_queries) <= 2
    _print_result("CASE 3", case3)

    print("\n三个查询拆解案例全部通过。")
    return [case1, case2, case3]


if __name__ == "__main__":
    run_examples()

