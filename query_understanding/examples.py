"""使用真实 DeepSeek API 运行 query_understanding 示例。

运行前请配置 ``DEEPSEEK_API_KEY``，然后使用以下任一方式执行：

    python query_understanding/examples.py
    python -m query_understanding.examples

本文件不使用模拟响应，所有分析、拆解和改写结果均由 DeepSeek 实时生成。
"""

from pathlib import Path
import sys
from typing import List

if __package__ in {None, ""}:
    # 支持直接执行：python query_understanding/examples.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from query_understanding.pipeline import QueryUnderstandingPipeline
    from query_understanding.schema import QueryAnalysis
else:
    # 支持模块执行：python -m query_understanding.examples
    from .pipeline import QueryUnderstandingPipeline
    from .schema import QueryAnalysis


CASE_1 = "近五年基于Transformer的医学图像分割方法"
CASE_2 = "比较CNN和Transformer在目标检测中的性能"
CASE_3 = "LLM在药物发现中的应用"


def _print_result(case_name: str, result: QueryAnalysis) -> None:
    """输出单个案例的完整 QueryAnalysis JSON。"""
    print(f"\n===== {case_name} =====")
    print(result.to_json(indent=2))


def _contains_text(values: List[str], expected: str) -> bool:
    """忽略大小写检查列表中是否存在目标科研概念。"""
    expected = expected.casefold()
    return any(expected in value.casefold() for value in values)


def run_examples() -> List[QueryAnalysis]:
    """调用真实 DeepSeek API 运行三个验收案例。"""
    pipeline = QueryUnderstandingPipeline(fallback_to_rules=False)

    # Case 1：检查研究意图、Transformer 方法和医学图像领域。
    case1 = pipeline.run(CASE_1)
    case1_entity_names = [entity.name for entity in case1.entities]
    assert case1.research_intent, "Case 1 未生成 research_intent"
    assert _contains_text(
        [*case1.method_constraints, *case1_entity_names], "Transformer"
    ), "Case 1 未识别 Transformer"
    assert _contains_text(
        [*case1.domain_constraints, *case1_entity_names], "医学图像"
    ) or _contains_text(
        [*case1.domain_constraints, *case1_entity_names], "Medical Image"
    ), "Case 1 未识别医学图像领域"
    _print_result("CASE 1", case1)

    # Case 2：比较型问题应被拆解为至少两个可独立检索的子查询。
    case2 = pipeline.run(CASE_2)
    _print_result("CASE 2", case2)
    assert len(case2.sub_queries) >= 2, "Case 2 的 sub_queries 数量少于 2"

    # Case 3：检查大语言模型与药物发现两个核心实体。
    case3 = pipeline.run(CASE_3)
    case3_entity_names = [entity.name for entity in case3.entities]
    has_llm = any(
        _contains_text(case3_entity_names, name)
        for name in ("Large Language Model", "LLM", "大语言模型")
    )
    has_drug_discovery = any(
        _contains_text(case3_entity_names, name)
        for name in ("Drug Discovery", "药物发现")
    )
    assert has_llm, "Case 3 未识别 Large Language Model/LLM 实体"
    assert has_drug_discovery, "Case 3 未识别 Drug Discovery 实体"
    _print_result("CASE 3", case3)

    print("\n全部 DeepSeek 示例测试通过。")
    return [case1, case2, case3]


if __name__ == "__main__":
    run_examples()
