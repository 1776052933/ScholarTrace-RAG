"""QueryAnalyzer 单元测试。运行：python -m unittest query_decomposition.test_analyzer"""

import unittest

from .analyzer import QueryAnalyzer
from .prompts import QUERY_COMPLEXITY_PROMPT


def _input(**updates):
    value = {
        "original_query": "单一主题论文检索",
        "research_intent": "检索单一主题研究",
        "entities": [],
        "method_constraints": [],
        "domain_constraints": [],
        "dataset_constraints": [],
        "task_type": "",
        "keywords": [],
        "expanded_queries": [],
    }
    value.update(updates)
    return value


class _FakeDeepSeekClient:
    def __init__(self, need_decompose=False):
        self.need_decompose = need_decompose
        self.called = False

    def complete_json(self, system_prompt, user_prompt, **kwargs):
        self.called = True
        assert system_prompt == QUERY_COMPLEXITY_PROMPT
        assert kwargs["temperature"] == 0.0
        return {"need_decompose": self.need_decompose, "reason": "DeepSeek复杂度判断"}


class QueryAnalyzerTests(unittest.TestCase):
    def test_three_entities_require_decomposition(self):
        client = _FakeDeepSeekClient()
        analyzer = QueryAnalyzer(client)
        entities = [{"name": str(i), "type": "method"} for i in range(3)]
        self.assertTrue(analyzer.analyze(_input(entities=entities)))
        self.assertFalse(client.called)

    def test_action_marker_requires_decomposition(self):
        analyzer = QueryAnalyzer(_FakeDeepSeekClient())
        self.assertTrue(analyzer.analyze(_input(original_query="比较两种方法的发展趋势")))

    def test_five_expanded_queries_require_decomposition(self):
        analyzer = QueryAnalyzer(_FakeDeepSeekClient())
        self.assertTrue(analyzer.analyze(_input(expanded_queries=[f"q{i}" for i in range(5)])))

    def test_broad_task_requires_decomposition(self):
        analyzer = QueryAnalyzer(_FakeDeepSeekClient())
        self.assertTrue(analyzer.analyze(_input(
            original_query="跨领域目标检测方法",
            task_type="object_detection",
        )))

    def test_ambiguous_query_uses_deepseek(self):
        client = _FakeDeepSeekClient(need_decompose=False)
        analyzer = QueryAnalyzer(client)
        self.assertFalse(analyzer.analyze(_input()))
        self.assertTrue(client.called)
        self.assertEqual(analyzer.last_reason, "DeepSeek复杂度判断")


if __name__ == "__main__":
    unittest.main()

