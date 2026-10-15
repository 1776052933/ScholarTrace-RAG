"""M3 refuses to treat a legacy Chroma completion as Milvus-ready."""
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.api.main import create_app
from literature_database.models import PaperModel


def test_api_requires_reindex_for_legacy_completed_paper(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "milvus")
    monkeypatch.setenv("RAG_COLLECTION_NAME", "paper_chunks_m3")
    url = f"sqlite:///{(tmp_path / 'm3.sqlite3').as_posix()}"
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    app = create_app(database_url=url, data_root=tmp_path)
    with TestClient(app) as client:
        with app.state.database.get_session() as session:
            session.add(PaperModel(
                paper_id="legacy", title="Previously in Chroma", authors=[],
                embedding_status="COMPLETED", index_signature=None,
            ))
            session.commit()
        papers = client.get("/api/v1/papers").json()
        assert papers[0]["embedding_status"] == "REINDEX_REQUIRED"
        response = client.post("/api/v1/questions", json={
            "paper_id": "legacy", "question": "What is the method?"
        })
        assert response.status_code == 409
        assert "not been rebuilt" in response.json()["detail"]



def test_milvus_ingest_failure_clears_old_ready_signature(tmp_path: Path):
    from literature_database.database import DatabaseManager
    from literature_database.schema import PaperMetadata, PaperResource
    from literature_database.service import LiteratureService
    url = f"sqlite:///{(tmp_path / 'fail.sqlite3').as_posix()}"
    database = DatabaseManager(url)
    database.create_tables()
    with database.get_session() as session:
        session.add(PaperModel(
            paper_id="paper-1", title="Paper", authors=[],
            embedding_status="COMPLETED", index_signature="milvus:old:signature",
        ))
        session.commit()

    class FailingReader:
        vector_store = type("Store", (), {"index_scope": "milvus:new"})()
        def ingest_paper(self, _resource):
            raise RuntimeError("index unavailable")
        def get_document(self, _paper_id):
            raise AssertionError("must not be called after ingestion failure")

    service = LiteratureService(database, create_tables=False)
    result = service.ingest_into_reader(
        [PaperResource(paper=PaperMetadata(paper_id="paper-1", title="Paper"))],
        FailingReader(),
        text_output_dir=str(tmp_path / "text"),
    )
    assert result[0]["success"] is False
    with database.get_session() as session:
        paper = session.get(PaperModel, "paper-1")
        assert paper.embedding_status == "FAILED"
        assert paper.index_signature is None
    database.dispose()
