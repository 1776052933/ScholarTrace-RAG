from query_decomposition.optimizer import SubQueryOptimizer
from query_decomposition.schema import SubQuery
from query_understanding.query_decomposer import QueryDecomposer
from query_understanding.schema import Entity, QueryAnalysis


def test_optimizer_separates_english_and_chinese_query_parts():
    item = SubQuery(
        query="transformer医学图像分割",
        purpose="test",
        query_type="method_search",
        priority="high",
    )

    result = SubQueryOptimizer().optimize_with_rules([item])

    assert result[0].query == "transformer 医学图像分割"
    assert result[0].english_query == "Transformer based medical image segmentation"


def test_rule_decomposer_inserts_spaces_between_methods_and_target():
    analysis = QueryAnalysis(
        original_query="比较 reinforcement learning 和 diffusion models 用于 video generation",
        research_intent="compare methods",
        entities=[
            Entity(name="reinforcement learning", type="method", importance="high"),
            Entity(name="diffusion models", type="model", importance="high"),
            Entity(name="video generation", type="task", importance="high"),
        ],
    )

    result = QueryDecomposer().decompose_with_rules(analysis)

    assert "reinforcement learning video generation 研究" in result.sub_queries
    assert "diffusion models video generation 研究" in result.sub_queries
    assert "reinforcement learning 和 diffusion models 比较" in result.sub_queries
