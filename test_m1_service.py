"""CLI compatibility after moving the workflow into the service package."""

from app.services.workflow import AutoPaperWorkflow
from main import AutoPaperWorkflow as CliWorkflow


class Stage:
    def __init__(self, result):
        self.result = result

    def run(self, *_args, **_kwargs):
        return self.result


class Literature:
    last_download_results = []

    def register_ranked_papers(self, ranking_result, *, scores=None):
        return [{"paper_id": p["paper_id"], "status": "METADATA_ONLY"}
                for p in ranking_result["papers"]]

    def get_missing_core_papers(self):
        return []


def test_service_and_cli_share_one_workflow_implementation():
    workflow = AutoPaperWorkflow(
        query_understanding=Stage({"query": "kiwi research"}),
        query_decomposition=Stage({"sub_queries": []}),
        search_agent=Stage({"papers": [{"paper_id": "arxiv:2401.12345", "title": "Kiwi paper"}]}),
        ranking_agent=Stage({"papers": [{"paper_id": "arxiv:2401.12345", "title": "Kiwi paper"}]}),
        literature_service=Literature(),
        enable_rag=False,
        enable_citation_tracking=False,
    )
    result = workflow.run("kiwi research", download_papers=False, ingest_rag=False)
    assert CliWorkflow is AutoPaperWorkflow
    assert result["status"] == "completed"
    assert result["stages"]["literature_database"]["selected_paper_ids"] == ["arxiv:2401.12345"]
