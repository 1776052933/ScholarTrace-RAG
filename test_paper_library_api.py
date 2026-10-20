"""Local PDF, collection persistence, and document lifecycle acceptance."""
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select
import fitz

from app.api.main import create_app
from app.storage.models import ResearchJobModel
from literature_database.models import PaperModel
from literature_database.service import LiteratureService


def _pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The actual PDF title and some useful research text")
    output = document.tobytes()
    document.close()
    return output


def _app(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = f"sqlite:///{(tmp_path / 'app.sqlite3').as_posix()}"
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")
    app = create_app(database_url=url, data_root=tmp_path / "data", ingestion_dispatcher=lambda _: None)
    return app, url


def test_upload_collection_index_and_delete(tmp_path, monkeypatch):
    app, url = _app(tmp_path, monkeypatch)
    data = _pdf()
    with TestClient(app) as client:
        bad = client.post("/api/v1/papers/upload", files={"file": ("bad.pdf", b"not a pdf")})
        assert bad.status_code == 422
        collection = client.post("/api/v1/collections", json={"name": "Reading list"})
        assert collection.status_code == 201, collection.text
        collection_id = collection.json()["collection_id"]
        uploaded = client.post("/api/v1/papers/upload", files={"file": ("custom.pdf", data)},
                               data={"title": "A custom label", "collection_id": collection_id})
        assert uploaded.status_code == 202, uploaded.text
        paper_id = uploaded.json()["paper_id"]
        job_id = uploaded.json()["ingestion_job_id"]
        assert paper_id.startswith("upload:") and job_id
        assert client.get("/api/v1/papers").json()[0]["has_pdf"] is True
        assert client.get(f"/api/v1/papers/{paper_id}/pdf").content == data
        assert client.get(f"/api/v1/collections/{collection_id}").json()["paper_ids"] == [paper_id]
        same = client.post("/api/v1/papers/upload", files={"file": ("again.pdf", data)},
                           data={"title": "Different label"})
        assert same.status_code == 202 and same.json()["ingestion_job_id"] == job_id
        assert client.delete(f"/api/v1/papers/{paper_id}").status_code == 409
        # The uploaded PDF must be accepted even when its human label differs from the PDF title.
        resources = LiteratureService(app.state.database, create_tables=False).prepare_for_reader(
            {"papers": [{"paper_id": paper_id, "title": "A custom label"}]}, target_count=1,
            output_dir=str(tmp_path / "data" / "papers"),
        )
        assert len(resources) == 1 and Path(resources[0].file_path).read_bytes() == data
        assert client.patch(f"/api/v1/collections/{collection_id}", json={"name": "Revised"}).json()["name"] == "Revised"

    restarted = create_app(database_url=url, data_root=tmp_path / "data", ingestion_dispatcher=lambda _: None)
    with TestClient(restarted) as client:
        assert client.get(f"/api/v1/collections/{collection_id}").json()["paper_ids"] == [paper_id]
        assert client.delete(f"/api/v1/collections/{collection_id}/papers/{paper_id}").json()["paper_ids"] == []
        assert client.post(f"/api/v1/collections/{collection_id}/papers", json={"paper_id": paper_id}).json()["paper_ids"] == [paper_id]
        with restarted.state.database.get_session() as session:
            session.get(ResearchJobModel, job_id).status = "completed"
            session.commit()
        deleted = []
        class Store:
            def delete(self, *, paper_id):
                deleted.append(paper_id)
        monkeypatch.setattr("rag_reader.store_factory.create_vector_store", lambda *args, **kwargs: Store())
        assert client.delete(f"/api/v1/papers/{paper_id}").status_code == 204
        assert deleted == [paper_id]
        assert client.get(f"/api/v1/collections/{collection_id}").json()["paper_ids"] == []
        assert client.get(f"/api/v1/papers/{paper_id}/pdf").status_code == 404
        with restarted.state.database.get_session() as session:
            assert session.scalar(select(PaperModel).where(PaperModel.paper_id == paper_id)) is None
        assert client.delete(f"/api/v1/collections/{collection_id}").status_code == 204


def test_upload_rejects_unknown_collection_without_leaking_pdf(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post("/api/v1/papers/upload", files={"file": ("test.pdf", _pdf())},
                               data={"collection_id": "missing"})
        assert response.status_code == 404
        assert client.get("/api/v1/papers").json() == []
    assert list((tmp_path / "data" / "uploads").glob("*.pdf")) == []


def test_upload_can_attach_pdf_to_discovered_paper(tmp_path, monkeypatch):
    app, _ = _app(tmp_path, monkeypatch)
    paper_id = "arxiv:local-missing-pdf"
    with app.state.database.get_session() as session:
        session.add(PaperModel(paper_id=paper_id, title="Previously discovered study",
                               authors=["Researcher"], abstract="", venue="",
                               citation_count=0, rank_score=0.0,
                               availability="METADATA_ONLY", embedding_status="PENDING"))
        session.commit()
    with TestClient(app) as client:
        upload = client.post("/api/v1/papers/upload", files={"file": ("other-title.pdf", _pdf())},
                             data={"paper_id": paper_id, "title": "Ignored upload title"})
        assert upload.status_code == 202, upload.text
        assert upload.json()["paper_id"] == paper_id
        library = client.get("/api/v1/papers").json()
        assert len(library) == 1 and library[0]["has_pdf"] is True
        assert library[0]["source"] == "discovered"
        resources = LiteratureService(app.state.database, create_tables=False).prepare_for_reader(
            {"papers": [{"paper_id": paper_id, "title": "Previously discovered study"}]}, target_count=1,
            output_dir=str(tmp_path / "data" / "papers"),
        )
        assert len(resources) == 1
        assert client.post("/api/v1/papers/upload", files={"file": ("repeat.pdf", _pdf())},
                           data={"paper_id": paper_id}).status_code == 409
        assert client.delete(f"/api/v1/papers/{paper_id}").status_code == 409
