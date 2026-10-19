"""M2 model configuration reaches each existing agent without changing algorithms."""
from app.config import ModelSettings
from app.services.workflow import AutoPaperWorkflow


def test_model_settings_are_shared_across_agents(tmp_path):
    settings = ModelSettings(
        api_key="test-key",
        model="test-model",
        base_url="https://example.invalid/v1",
        timeout_seconds=17,
    )
    workflow = AutoPaperWorkflow(
        database_url=f"sqlite:///{(tmp_path / 'models.sqlite3').as_posix()}",
        chroma_directory=str(tmp_path / "chroma"),
        enable_rag=False,
        enable_citation_tracking=False,
        model_settings=settings,
    )
    clients = [
        workflow.query_understanding.llm_client,
        workflow.query_decomposition.llm_client,
        workflow.search_agent.agent.planner.llm_client,
        workflow.search_agent.agent.evaluator.llm_client,
        workflow.ranking_agent.evaluator.llm_client,
    ]
    assert all(client.model == "test-model" for client in clients)
    assert all(client.base_url == "https://example.invalid/v1" for client in clients)
    assert all((getattr(client, "timeout_seconds", None) or getattr(client, "timeout", None)) == 17 for client in clients)
    assert "test-key" not in repr(settings)
