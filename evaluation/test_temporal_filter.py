from search_agent.schema import PaperMetadata, SearchPlan, SearchQuery
from search_agent.searcher import PaperSearcher
from evaluation.scholargym import scholar_gym_cutoff
from evaluation.run_pasa import _evaluate_one as evaluate_pasa_one
from evaluation.run_scholargym import _evaluate_one as evaluate_scholargym_one


def _plan() -> SearchPlan:
    return SearchPlan(
        original_query="test query",
        search_round=1,
        queries=[SearchQuery(
            query="test query",
            source="all",
            purpose="test",
            priority="high",
        )],
        strategy="test",
    )


def _paper(title: str, year: int, publication_date: str | None = None) -> PaperMetadata:
    return PaperMetadata(
        title=title,
        year=year,
        publication_date=publication_date,
        source="fake",
    )


def test_searcher_passes_cutoff_and_filters_exact_dates():
    class DateAwareTool:
        source = "fake"

        def __init__(self):
            self.cutoff = None

        def search(self, query, limit=10, publication_date_to=None):
            self.cutoff = publication_date_to
            return [
                _paper("eligible", 2024, "2024-05-01"),
                _paper("future", 2024, "2024-08-01"),
            ]

    tool = DateAwareTool()
    result = PaperSearcher({"fake": tool}).search(
        _plan(),
        publication_date_to="2024-06-30",
    )

    assert tool.cutoff == "2024-06-30"
    assert [paper.title for paper in result.papers] == ["eligible"]


def test_searcher_keeps_legacy_tools_and_filters_by_year():
    class LegacyTool:
        source = "legacy"

        def search(self, query, limit=10):
            return [_paper("eligible", 2023), _paper("future", 2025)]

    result = PaperSearcher({"legacy": LegacyTool()}).search(
        _plan(),
        publication_date_to="2024-06-30",
    )

    assert [paper.title for paper in result.papers] == ["eligible"]


def test_scholargym_month_constraint_uses_month_end():
    assert scholar_gym_cutoff("2024-02") == "2024-02-29"
    assert scholar_gym_cutoff("2023") == "2023-12-31"
    assert scholar_gym_cutoff("invalid") is None


class _FakeWorkflow:
    def __init__(self):
        self.publication_date_to = None

    def run(self, query, **kwargs):
        self.publication_date_to = kwargs.get("publication_date_to")
        return {
            "elapsed_seconds": 0,
            "stages": {
                "search_agent": {"result": {"papers": []}},
                "ranking_agent": {"result": {"papers": []}},
            },
        }


def test_scholargym_evaluator_receives_cutoff_without_global_args(tmp_path):
    workflow = _FakeWorkflow()
    record = evaluate_scholargym_one(
        workflow,
        {
            "query_id": "sg-1",
            "query": "test",
            "gold_papers": [],
            "gold_arxiv_ids": [],
        },
        raw_dir=tmp_path,
        top_k=20,
        cutoffs=[20],
        publication_date_to="2024-06-30",
    )

    assert record["status"] == "completed"
    assert workflow.publication_date_to == "2024-06-30"


def test_pasa_evaluator_receives_cutoff_without_global_args(tmp_path):
    workflow = _FakeWorkflow()
    record = evaluate_pasa_one(
        workflow,
        {"query_id": "pasa-1", "query": "test", "gold_titles": []},
        raw_dir=tmp_path,
        top_k=20,
        cutoffs=[20],
        publication_date_to="2024-06-30",
    )

    assert record["status"] == "completed"
    assert workflow.publication_date_to == "2024-06-30"
