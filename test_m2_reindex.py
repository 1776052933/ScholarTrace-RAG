"""Reindex only PDFs that the legacy database judged usable."""
from pathlib import Path

from literature_database.database import DatabaseManager
from literature_database.models import PaperFileModel, PaperModel
from scripts import reindex_migrated_papers


def test_reindex_skips_mismatched_paper_files(tmp_path: Path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'papers.sqlite3').as_posix()}"
    db = DatabaseManager(url)
    db.create_tables()
    files = {}
    for paper_id in ("valid", "mismatch"):
        pdf = tmp_path / f"{paper_id}.pdf"
        parsed = tmp_path / f"{paper_id}.txt"
        pdf.write_bytes(b"%PDF-1.4\nexample")
        parsed.write_text("example")
        files[paper_id] = (pdf, parsed)
    with db.get_session() as session:
        session.add_all([
            PaperModel(paper_id="valid", title="Valid paper"),
            PaperModel(paper_id="mismatch", title="Wrong paper"),
        ])
        session.flush()
        for paper_id, status in (("valid", "AVAILABLE"), ("mismatch", "MISMATCH")):
            pdf, parsed = files[paper_id]
            session.add(PaperFileModel(paper_id=paper_id, file_path=str(pdf), file_type="pdf", status=status))
            session.add(PaperFileModel(paper_id=paper_id, file_path=str(parsed), file_type="text", status="PARSED" if status == "AVAILABLE" else "MISMATCH"))
        session.commit()
    db.dispose()

    ingested = []
    vector_paths = []

    class FakeReader:
        def __init__(self, *, vector_store):
            pass

        def ingest_paper(self, resource):
            ingested.append(resource.paper.paper_id)
            return ["chunk"]

    def fake_vector_store(**kwargs):
        vector_paths.append(kwargs["persist_directory"])
        return object()

    monkeypatch.setattr(reindex_migrated_papers, "VectorStore", fake_vector_store)
    monkeypatch.setattr(reindex_migrated_papers, "RAGReader", FakeReader)
    assert reindex_migrated_papers.rebuild(url, tmp_path / "index") == {"indexed": 1, "failed": 0}
    assert ingested == ["valid"]
    assert vector_paths == [str((tmp_path / "index" / "chroma").resolve())]
    db = DatabaseManager(url)
    with db.get_session() as session:
        assert session.get(PaperModel, "valid").embedding_status == "COMPLETED"
        assert session.get(PaperModel, "mismatch").embedding_status == "PENDING"
    db.dispose()


def test_windows_unicode_absolute_chroma_path_uses_ascii_relative_path(tmp_path: Path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    import pytest
    from rag_reader.vector_store import VectorStore

    if os.name != "nt":
        pytest.skip("Windows Chroma path regression")
    workdir = tmp_path / "论文目录"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    actual = workdir / "data" / "chroma"
    paths = []

    class FakeClient:
        def get_or_create_collection(self, **kwargs):
            return object()

    def fake_client(*, path):
        paths.append(path)
        return FakeClient()

    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=fake_client))
    VectorStore(persist_directory=str(actual))
    assert paths == [str(Path("data") / "chroma")]
    assert Path(paths[0]).resolve() == actual
