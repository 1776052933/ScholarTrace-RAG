"""SearchAgent 完整运行示例。

直接运行：
    python -m search_agent.examples
或：
    python search_agent/examples.py
"""

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from search_agent.config import SearchAgentConfig
    from search_agent.pipeline import SearchAgent
else:
    from .config import SearchAgentConfig
    from .pipeline import SearchAgent


# 模拟 query_decomposition 的输出。sub_queries 保留查询、目的、类型、
# 优先级及英文检索表达，可直接交给 SearchAgent。
DECOMPOSITION_RESULT = {
    "original_query": "比较CNN和Transformer在目标检测中的性能",
    "research_intent": "比较CNN和Transformer目标检测性能",
    "need_decompose": True,
    "reason": "包含CNN、Transformer两个研究对象及性能比较关系",
    "keywords": [
        "CNN",
        "Transformer",
        "目标检测",
        "object detection",
        "performance comparison",
    ],
    "expanded_queries": [
        "CNN object detection performance",
        "Transformer object detection performance",
        "CNN vs Transformer object detection benchmark",
    ],
    "sub_queries": [
        {
            "query": "CNN目标检测",
            "purpose": "检索CNN目标检测方法及实验结果",
            "query_type": "method_search",
            "priority": "high",
            "english_query": "CNN object detection methods performance",
            "keywords": ["CNN", "object detection", "performance"],
        },
        {
            "query": "Transformer目标检测",
            "purpose": "检索Transformer目标检测方法及实验结果",
            "query_type": "method_search",
            "priority": "high",
            "english_query": "Transformer object detection methods performance",
            "keywords": ["Transformer", "DETR", "object detection", "performance"],
        },
        {
            "query": "CNN和Transformer目标检测性能比较",
            "purpose": "比较两类方法的准确率、速度及benchmark表现",
            "query_type": "comparison",
            "priority": "high",
            "english_query": "CNN versus Transformer object detection benchmark comparison",
            "keywords": ["CNN", "Transformer", "object detection", "benchmark"],
        },
    ],
}


def run_example() -> None:
    """运行真实论文搜索接口并输出 Agent 每轮状态。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = replace(
        SearchAgentConfig.from_env(),
        max_rounds=3,
        results_per_task=5,
        # 配置 DeepSeek Key 后，Planner、Evaluator 和 Reflector 自动使用 LLM；
        # 未配置时使用内置规则，但论文数据库仍然调用真实 API。
        use_llm=bool(os.getenv("DEEPSEEK_API_KEY")),
    )
    result = SearchAgent(config=config).run(DECOMPOSITION_RESULT)

    plans = [iteration.plan.model_dump(mode="json") for iteration in result.iterations]
    called_queries = [
        {
            "round": iteration.round_number,
            "query": query.query,
            "source": query.source,
            "purpose": query.purpose,
        }
        for iteration in result.iterations
        for query in iteration.plan.queries
    ]
    evaluations = [
        {
            "round": iteration.round_number,
            **iteration.evaluation.model_dump(mode="json"),
        }
        for iteration in result.iterations
    ]
    final_status = {
        "original_query": result.original_query,
        "rounds_completed": result.rounds_completed,
        "stopped_reason": result.stopped_reason,
        "paper_count": len(result.papers),
        "need_refine": result.final_evaluation.need_refine,
    }

    output = {
        "1_search_plans": plans,
        "2_called_queries": called_queries,
        "3_paper_count": len(result.papers),
        "4_evaluations": evaluations,
        "5_final_status": final_status,
        "papers": [paper.model_dump(mode="json") for paper in result.papers],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_example()
