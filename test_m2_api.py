"""M2 API acceptance: durable jobs, restart recovery, contracts and paper evidence."""
from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.main import create_app
from app.services.workflow import WorkflowCancelled
from app.storage.models import ResearchJobModel
from literature_database.models import PaperModel


class ImmediateWorkflow:
    def run(self, query, *, should_cancel, **kwargs):
        assert kwargs["max_papers"] == 1
        assert not should_cancel()
        logging.getLogger("auto_paper").info("开始阶段：search_agent")
        return {"status": "completed", "query": query, "stages": {}}


class CancellableWorkflow:
    def run(self, query, *, should_cancel, **kwargs):
        import time
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if should_cancel():
                raise WorkflowCancelled("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel signal was not delivered")


class FakeQuestions:
    def ask(self, paper_id, question):
        return "answer-1", {"paper_id": paper_id, "question": question, "answer": "6 layers", "citations": [{"paper_id": paper_id, "page": 3, "excerpt": "N = 6"}]}

    def evidence(self, answer_id):
        if answer_id != "answer-1":
            raise KeyError(answer_id)
        return [{"paper_id": "arxiv:123", "page": 3, "excerpt": "N = 6"}]


def migrated_database(tmp_path: Path, monkeypatch) -> str:
    url = f"sqlite:///{(tmp_path / 'api.sqlite3').as_posix()}"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    return url


def test_api_job_survives_restart_and_reconciles_orphan(tmp_path, monkeypatch):
    url = migrated_database(tmp_path, monkeypatch)
    app = create_app(database_url=url, data_root=tmp_path / "assets", workflow_factory=lambda _: ImmediateWorkflow())
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["database"] is True
        assert client.get("/openapi.json").status_code == 200
        response = client.post("/api/v1/jobs", json={
            "query": "Example research", "max_papers": 1,
            "download_papers": False, "enable_rag": False,
            "summarize": False, "extract_knowledge": False,
        })
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        app.state.runner.join(job_id, timeout=5)
        done = client.get(f"/api/v1/jobs/{job_id}").json()
        assert done["status"] == "completed"
        assert done["result"]["query"] == "Example research"
        assert any("search_agent" in log for log in done["logs"])
        listed = client.get("/api/v1/jobs").json()
        assert listed[0]["job_id"] == job_id
        assert "result" not in listed[0] and "logs" not in listed[0]
        with app.state.database.get_session() as session:
            session.add(ResearchJobModel(job_id="orphan", query="old", options={}, status="running", progress=30, logs=[], retryable=False))
            session.commit()

    restarted = create_app(database_url=url, data_root=tmp_path / "assets", workflow_factory=lambda _: ImmediateWorkflow())
    with TestClient(restarted) as client:
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
        orphan = client.get("/api/v1/jobs/orphan").json()
        assert orphan["status"] == "interrupted"
        assert orphan["retryable"] is True
        assert client.get("/api/v1/jobs/missing").status_code == 404
        assert len(client.get("/api/v1/jobs").json()) == 2


def test_api_cooperative_cancel_and_question_contract(tmp_path, monkeypatch):
    url = migrated_database(tmp_path, monkeypatch)
    app = create_app(database_url=url, data_root=tmp_path / "assets", workflow_factory=lambda _: CancellableWorkflow(), question_service=FakeQuestions())
    with TestClient(app) as client:
        invalid = client.post("/api/v1/jobs", json={"query": "  "})
        assert invalid.status_code == 422
        created = client.post("/api/v1/jobs", json={
            "query": "cancel me", "max_papers": 1,
            "download_papers": False, "enable_rag": False,
            "summarize": False, "extract_knowledge": False,
        })
        job_id = created.json()["job_id"]
        assert client.post("/api/v1/jobs", json={
            "query": "second job", "max_papers": 1,
            "download_papers": False, "enable_rag": False,
            "summarize": False, "extract_knowledge": False,
        }).status_code == 409
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200
        app.state.runner.join(job_id, timeout=5)
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "cancelled"

        with app.state.database.get_session() as session:
            session.add(PaperModel(paper_id="arxiv:123", title="Example paper", authors=[]))
            session.commit()
        papers = client.get("/api/v1/papers").json()
        assert papers[0]["paper_id"] == "arxiv:123"
        answer = client.post("/api/v1/questions", json={"paper_id": "arxiv:123", "question": "Layers?"})
        assert answer.status_code == 200
        assert answer.json()["answer"]["answer"] == "6 layers"
        evidence = client.get("/api/v1/answers/answer-1/evidence")
        assert evidence.json()[0]["page"] == 3
