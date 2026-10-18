"""M4 acceptance: selected papers, durable retry, restart and multi-paper API."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.storage.models import ResearchJobModel, utc_now
from literature_database.models import PaperModel
from rag_reader.qa import PaperQA
from rag_reader.schema import DocumentChunk, RAGAnswer


def migrated_database(tmp_path: Path) -> str:
    url = f"sqlite:///{(tmp_path / 'm4.sqlite3').as_posix()}"
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    return url


def seed_discovery(app) -> None:
    with app.state.database.get_session() as session:
        session.add(ResearchJobModel(
            job_id="a" * 32, job_type="research", query="example topic",
            options={"query": "example topic", "download_papers": False},
            status="completed", progress=100, logs=[], retryable=False,
            result={"stages": {"ranking_agent": {"result": {"papers": [
                {"paper_id": "paper-1", "title": "First paper"},
                {"paper_id": "paper-2", "title": "Second paper"},
            ]}}}},
        ))
        session.commit()


class FakeLiterature:
    def __init__(self) -> None:
        self.fail_download = {"paper-2"}
        self.fail_ingest = set()
        self.processed = []
        self.last_download_results = []

    def prepare_for_reader(self, ranking, **_kwargs):
        paper_id = ranking["papers"][0]["paper_id"]
        self.processed.append(paper_id)
        if paper_id in self.fail_download:
            self.last_download_results = [{"paper_id": paper_id, "success": False, "error": "temporary network error"}]
            return []
        self.last_download_results = [{"paper_id": paper_id, "success": True}]
        return [paper_id]

    def ingest_into_reader(self, resources, _reader, **_kwargs):
        if resources[0] in self.fail_ingest:
            return [{"paper_id": resources[0], "success": False, "error": "parser failed"}]
        return [{"paper_id": resources[0], "success": True, "chunk_count": 7}]


def test_selected_ingestion_is_idempotent_and_retries_only_failed_papers(tmp_path):
    url = migrated_database(tmp_path)
    dispatched = []
    app = create_app(database_url=url, data_root=tmp_path / "assets", ingestion_dispatcher=dispatched.append)
    literature = FakeLiterature()
    app.state.ingestions.literature_factory = lambda: literature
    app.state.ingestions.reader_factory = lambda: object()
    with TestClient(app) as client:
        seed_discovery(app)
        payload = {"discovery_job_id": "a" * 32, "paper_ids": ["paper-2", "paper-1"]}
        assert client.post("/api/v1/ingestions", json={**payload, "paper_ids": ["paper-3"]}).status_code == 409
        first = client.post("/api/v1/ingestions", json=payload)
        assert first.status_code == 202, first.text
        job_id = first.json()["job_id"]
        repeat = client.post("/api/v1/ingestions", json={**payload, "paper_ids": ["paper-1", "paper-2"]})
        assert repeat.json()["job_id"] == job_id
        assert dispatched == [job_id, job_id]
        partial = app.state.ingestions.execute(job_id)
        assert partial.status == "partial_success"
        assert partial.progress == 100 and partial.retryable
        assert [item["status"] for item in partial.result["papers"]] == ["completed", "failed"]
        assert [item["attempts"] for item in partial.result["papers"]] == [1, 1]
        assert app.state.ingestions.execute(job_id).status == "partial_success"  # duplicate delivery
        literature.fail_download.clear()
        retried = client.post(f"/api/v1/ingestions/{job_id}/retry")
        assert retried.status_code == 202
        done = app.state.ingestions.execute(job_id)
        assert done.status == "completed"
        assert literature.processed == ["paper-1", "paper-2", "paper-2"]
        assert [item["attempts"] for item in done.result["papers"]] == [1, 2]
        assert client.get(f"/api/v1/jobs/{job_id}").json()["job_type"] == "ingestion"


def test_lost_worker_is_recovered_without_api_marking_ingestion_interrupted(tmp_path):
    url = migrated_database(tmp_path)
    dispatched = []
    app = create_app(database_url=url, data_root=tmp_path / "assets", ingestion_dispatcher=dispatched.append)
    with TestClient(app) as client:
        seed_discovery(app)
        created = client.post("/api/v1/ingestions", json={
            "discovery_job_id": "a" * 32, "paper_ids": ["paper-1"],
        }).json()
        job_id = created["job_id"]
        with app.state.database.get_session() as session:
            row = session.get(ResearchJobModel, job_id)
            row.status = "running"
            row.updated_at = utc_now() - timedelta(minutes=10)
            result = dict(row.result)
            result["papers"][0]["status"] = "running"
            row.result = result
            session.commit()
    restarted = create_app(database_url=url, data_root=tmp_path / "assets", ingestion_dispatcher=dispatched.append)
    literature = FakeLiterature()
    literature.fail_download.clear()
    restarted.state.ingestions.literature_factory = lambda: literature
    restarted.state.ingestions.reader_factory = lambda: object()
    with TestClient(restarted) as client:
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
        recovered = restarted.state.ingestions.reconcile(stale_after_seconds=120)
        assert recovered == [job_id]
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "queued"
        assert dispatched[-1] == job_id
        assert restarted.state.ingestions.execute(job_id).status == "completed"


def test_parser_failure_is_visible_and_retryable(tmp_path):
    url = migrated_database(tmp_path)
    app = create_app(database_url=url, data_root=tmp_path / "assets", ingestion_dispatcher=lambda _id: None)
    literature = FakeLiterature()
    literature.fail_download.clear()
    literature.fail_ingest.add("paper-1")
    app.state.ingestions.literature_factory = lambda: literature
    app.state.ingestions.reader_factory = lambda: object()
    with TestClient(app) as client:
        seed_discovery(app)
        job_id = client.post("/api/v1/ingestions", json={
            "discovery_job_id": "a" * 32, "paper_ids": ["paper-1"],
        }).json()["job_id"]
        failed = app.state.ingestions.execute(job_id)
        assert failed.status == "failed"
        assert "parser failed" in failed.result["papers"][0]["error"]
        literature.fail_ingest.clear()
        assert client.post(f"/api/v1/ingestions/{job_id}/retry").status_code == 202
        assert app.state.ingestions.execute(job_id).status == "completed"


class FakeQuestions:
    def ask_many(self, ids, question):
        return "answer-1", {"answer": "Both use attention.", "question": question,
                            "citations": [{"paper_id": ids[0], "page": 3, "excerpt": "attention"}]}

    def evidence(self, answer_id):
        if answer_id != "answer-1":
            raise KeyError(answer_id)
        return [{"paper_id": "paper-1", "page": 3, "excerpt": "attention"}]

    def export(self, answer_id):
        if answer_id != "answer-1":
            raise KeyError(answer_id)
        return "# Answer\n\nBoth use attention.\n"


def test_multi_paper_api_contract_without_external_model(tmp_path):
    url = migrated_database(tmp_path)
    app = create_app(database_url=url, data_root=tmp_path / "assets", question_service=FakeQuestions())
    with TestClient(app) as client:
        bad = client.post("/api/v1/questions/multi", json={"paper_ids": ["paper-1"], "question": "Compare"})
        assert bad.status_code == 422
        response = client.post("/api/v1/questions/multi", json={
            "paper_ids": ["paper-1", "paper-2"], "question": "Compare",
        })
        assert response.status_code == 200
        assert response.json()["answer"]["answer"] == "Both use attention."
        assert client.get("/api/v1/answers/answer-1/evidence").json()[0]["page"] == 3
        assert "Both use attention" in client.get("/api/v1/answers/answer-1/export").text


def test_multi_paper_qa_keeps_sources_from_both_papers():
    chunks = [DocumentChunk(chunk_id=f"p{index}-chunk-1", paper_id=f"p{index}",
                            text=f"Evidence in paper {index}", page=index)
              for index in (1, 2)]

    class Retriever:
        def retrieve_multiple(self, paper_ids, question, *, top_k):
            assert paper_ids == ["p1", "p2"] and top_k == 5
            return chunks

        def retrieve(self, paper_id, _question, *, top_k):
            assert top_k == 7
            return [chunk for chunk in chunks if chunk.paper_id == paper_id]

    class Client:
        def chat(self, _messages, **_kwargs):
            return json.dumps({"answer": "Both papers provide evidence.",
                               "sources": ["p1-chunk-1", "p2-chunk-1"], "confidence": 0.8})

    answer = PaperQA(Retriever(), llm_client=Client()).ask_many(["p1", "p2"], "Compare")
    assert {item.paper_id for item in answer.citations} == {"p1", "p2"}
    assert {item.page for item in answer.citations} == {1, 2}
    assert answer.confidence == 0.5  # only one passage per paper after capped expansion


def test_multi_paper_qa_adds_evidence_for_paper_missing_from_global_top_k():
    first = DocumentChunk(chunk_id="p1-chunk", paper_id="p1", text="First evidence", page=1)
    second = DocumentChunk(chunk_id="p2-chunk", paper_id="p2", text="Second evidence", page=2)

    class Retriever:
        def retrieve_multiple(self, _ids, _question, *, top_k):
            assert top_k == 5
            return [first]

        def retrieve(self, paper_id, _question, *, top_k):
            assert paper_id in {"p1", "p2"} and top_k == 7
            return [first] if paper_id == "p1" else [second]

    class Client:
        def chat(self, messages, **_kwargs):
            assert "Second evidence" in str(messages)
            return json.dumps({"answer": "Comparison", "sources": ["p1-chunk", "p2-chunk"],
                               "confidence": 0.8})

    answer = PaperQA(Retriever(), llm_client=Client()).ask_many(["p1", "p2"], "Compare")
    assert {item.paper_id for item in answer.citations} == {"p1", "p2"}


def test_multi_paper_answer_is_persisted_with_export(tmp_path, monkeypatch):
    from app.services import questions as module
    from literature_database.database import DatabaseManager

    monkeypatch.setenv("RAG_VECTOR_BACKEND", "chroma")
    url = migrated_database(tmp_path)
    database = DatabaseManager(url)
    with database.get_session() as session:
        for paper_id in ("p1", "p2"):
            session.add(PaperModel(paper_id=paper_id, title=paper_id, authors=[], embedding_status="COMPLETED"))
        session.commit()

    class Reader:
        def __init__(self, **_kwargs):
            self.qa = self

        def ask_many(self, paper_ids, question):
            assert paper_ids == ["p1", "p2"]
            return RAGAnswer(question=question, answer="Shared conclusion", sources=["p1-chunk-1"],
                             confidence=0.8, citations=[{
                                 "paper_id": "p1", "chunk_id": "p1-chunk-1", "title": "First",
                                 "page": 3, "excerpt": "Shared conclusion",
                             }])

    monkeypatch.setattr(module, "RAGReader", Reader)
    monkeypatch.setattr(module, "DeepSeekClient", lambda **_kwargs: object())
    monkeypatch.setattr(module, "create_vector_store", lambda *_args, **_kwargs: object())
    service = module.QuestionService(database, tmp_path)
    answer_id, answer = service.ask_many(["p1", "p2"], "Compare")
    assert answer["answer"] == "Shared conclusion"
    database.dispose()

    reopened = DatabaseManager(url)
    saved = module.QuestionService(reopened, tmp_path)
    assert saved.evidence(answer_id)[0]["page"] == 3
    report = saved.export(answer_id)
    assert "p1, p2" in report and "Shared conclusion" in report
    reopened.dispose()
