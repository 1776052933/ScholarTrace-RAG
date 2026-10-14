from ranking_agent.cache import EvaluationCache
from ranking_agent.evaluator import PaperEvaluator
from ranking_agent.schema import Paper, QueryContext


class FakeClient:
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return (
            '{"relevance_score": 0.9, "method_match": 0.8, '
            '"task_match": 0.7, "reason": "matched"}'
        )


class FakeBatchClient:
    model = "deepseek-chat"

    def __init__(self, *, omit_last=False) -> None:
        self.calls = 0
        self.omit_last = omit_last

    def chat(self, messages, **kwargs):
        import json

        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        papers = payload.get("papers")
        if papers is None:
            return (
                '{"relevance_score": 0.7, "method_match": 0.6, '
                '"task_match": 0.5, "reason": "single retry"}'
            )
        values = papers[:-1] if self.omit_last else papers
        return json.dumps({
            "evaluations": [
                {
                    "item_id": item["item_id"],
                    "relevance_score": 0.9,
                    "method_match": 0.8,
                    "task_match": 0.7,
                    "reason": "batch matched",
                }
                for item in values
            ]
        })


def _context(query: str = "medical segmentation") -> QueryContext:
    return QueryContext(
        original_query=query,
        research_intent=query,
    )


def _paper(identifier: str = "paper-1") -> Paper:
    return Paper(
        paper_id=identifier,
        title="Transformer segmentation",
        abstract="A medical image segmentation method.",
    )


def test_valid_evaluation_is_reused_without_calling_the_api(tmp_path):
    cache = EvaluationCache(str(tmp_path / "scores.sqlite3"))
    first_client = FakeClient()
    second_client = FakeClient()

    first = PaperEvaluator(first_client, cache=cache).evaluate(_context(), _paper())
    second = PaperEvaluator(second_client, cache=cache).evaluate(_context(), _paper())

    assert first == second
    assert first_client.calls == 1
    assert second_client.calls == 0


def test_changed_query_does_not_reuse_an_old_evaluation(tmp_path):
    cache = EvaluationCache(str(tmp_path / "scores.sqlite3"))
    client = FakeClient()
    evaluator = PaperEvaluator(client, cache=cache)

    evaluator.evaluate(_context("query one"), _paper())
    evaluator.evaluate(_context("query two"), _paper())

    assert client.calls == 2


def test_five_papers_are_evaluated_in_one_api_request(tmp_path):
    cache = EvaluationCache(str(tmp_path / "scores.sqlite3"))
    client = FakeBatchClient()
    papers = [_paper(f"paper-{index}") for index in range(5)]

    values = PaperEvaluator(client, cache=cache).evaluate_batch(_context(), papers)

    assert len(values) == 5
    assert client.calls == 1


def test_batch_omission_retries_only_the_missing_paper(tmp_path):
    cache = EvaluationCache(str(tmp_path / "scores.sqlite3"))
    client = FakeBatchClient(omit_last=True)
    papers = [_paper(f"paper-{index}") for index in range(3)]

    values = PaperEvaluator(client, cache=cache).evaluate_batch(_context(), papers)

    assert len(values) == 3
    assert client.calls == 2
    assert values[-1]["reason"] == "single retry"
