"""Top-K download coordination keeps DB writes serial and network work parallel."""

from pathlib import Path
import threading
import time

import paper_downloader

from literature_database.database import DatabaseManager
from literature_database.downloader import PaperDownloadCoordinator
from literature_database.repository import FileRepository, PaperRepository
from literature_database.schema import PaperMetadata
from literature_database.service import LiteratureService


def test_download_many_passes_metadata_and_runs_concurrently(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    database.create_tables()
    active = 0
    max_active = 0
    lock = threading.Lock()
    received = []

    def fake_download(value, *, output_dir, **kwargs):
        nonlocal active, max_active
        assert isinstance(value, dict)
        received.append(value)
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.08)
        path = Path(output_dir) / f"{value['arxiv_id']}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        with lock:
            active -= 1
        return {
            "success": True,
            "file_path": str(path),
            "error": None,
            "engine_used": "arxiv",
        }

    monkeypatch.setattr(paper_downloader, "download_pdf", fake_download)
    ids = [
        "http://arxiv.org/abs/2201.01266v1",
        "http://arxiv.org/abs/1706.03762v7",
    ]
    with database.get_session() as session:
        repository = PaperRepository(session)
        repository.add_paper(PaperMetadata(
            paper_id=ids[0],
            title="Swin UNETR",
            arxiv_id="2201.01266v1",
        ))
        repository.add_paper(PaperMetadata(
            paper_id=ids[1],
            title="Attention Is All You Need",
            arxiv_id="1706.03762v7",
        ))
        session.commit()

        coordinator = PaperDownloadCoordinator(
            session,
            output_dir=str(tmp_path / "pdfs"),
            max_workers=2,
        )
        results = coordinator.download_many(ids)
        session.commit()

    assert [item["paper_id"] for item in results] == ids
    assert all(item["success"] for item in results)
    assert max_active == 2
    assert {item["arxiv_id"] for item in received} == {
        "2201.01266v1",
        "1706.03762v7",
    }
    database.dispose()


def test_mismatched_existing_pdf_is_not_reused(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    database.create_tables()
    paper_id = "paper-1"
    wrong_path = tmp_path / "wrong.pdf"
    correct_path = tmp_path / "correct.pdf"
    wrong_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    def fake_download(value, *, output_dir, **kwargs):
        correct_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        return {
            "success": True,
            "file_path": str(correct_path),
            "error": None,
            "engine_used": "direct",
        }

    monkeypatch.setattr(
        "literature_database.downloader.pdf_matches_title",
        lambda path, title: False if Path(path) == wrong_path else None,
    )

    with database.get_session() as session:
        PaperRepository(session).add_paper(PaperMetadata(
            paper_id=paper_id,
            title="Requested Paper",
        ))
        old_record = FileRepository(session).add_file(paper_id, str(wrong_path))
        session.commit()

        result = PaperDownloadCoordinator(
            session,
            download_function=fake_download,
            output_dir=str(tmp_path),
        ).download(paper_id)
        session.commit()

        assert result["success"] is True
        assert result["skipped"] is False
        assert result["file_path"] == str(correct_path.resolve())
        assert old_record.status == "MISMATCH"

    database.dispose()


def test_mismatched_pdf_invalidates_parsed_text(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    database.create_tables()
    paper_id = "paper-1"
    wrong_pdf = tmp_path / "wrong.pdf"
    wrong_text = tmp_path / "wrong.txt"
    wrong_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    wrong_text.write_text("wrong paper", encoding="utf-8")

    monkeypatch.setattr(
        "literature_database.downloader.pdf_matches_title",
        lambda path, title: False,
    )
    with database.get_session() as session:
        paper = PaperRepository(session).add_paper(PaperMetadata(
            paper_id=paper_id,
            title="Requested Paper",
        ))
        files = FileRepository(session)
        files.add_file(paper_id, str(wrong_pdf), status="AVAILABLE", file_type="pdf")
        text_record = files.add_file(paper_id, str(wrong_text), status="PARSED", file_type="text")
        paper.embedding_status = "COMPLETED"
        session.commit()

        coordinator = PaperDownloadCoordinator(
            session,
            download_function=lambda *args, **kwargs: {
                "success": False,
                "file_path": None,
                "error": "not found",
            },
            output_dir=str(tmp_path),
        )
        coordinator.download(paper_id)
        session.commit()

        assert text_record.status == "MISMATCH"
        assert paper.embedding_status == "PENDING"
    database.dispose()


def test_same_pdf_path_cannot_be_bound_to_different_titles(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    database.create_tables()
    shared_path = tmp_path / "shared.pdf"
    shared_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    def fake_download(value, *, output_dir, **kwargs):
        return {
            "success": True,
            "file_path": str(shared_path),
            "error": None,
            "engine_used": "arxiv",
        }

    monkeypatch.setattr(
        "literature_database.downloader.pdf_matches_title",
        lambda path, title: None,
    )
    with database.get_session() as session:
        repository = PaperRepository(session)
        repository.add_paper(PaperMetadata(paper_id="paper-1", title="First Paper"))
        repository.add_paper(PaperMetadata(paper_id="paper-2", title="Entirely Different Study"))
        session.commit()

        coordinator = PaperDownloadCoordinator(
            session,
            download_function=fake_download,
            output_dir=str(tmp_path),
        )
        results = coordinator.download_many(["paper-1", "paper-2"])
        session.commit()

    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert "already bound to a different paper" in results[1]["error"]
    database.dispose()


def test_same_pdf_path_is_rejected_even_for_similar_titles(tmp_path, monkeypatch):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    database.create_tables()
    shared_path = tmp_path / "shared.pdf"
    shared_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        "literature_database.downloader.pdf_matches_title",
        lambda path, title: True,
    )

    def fake_download(value, *, output_dir, **kwargs):
        return {"success": True, "file_path": str(shared_path), "engine_used": "arxiv"}

    with database.get_session() as session:
        repository = PaperRepository(session)
        repository.add_paper(PaperMetadata(
            paper_id="paper-1",
            title="Anchor-guided global view reconstruction for multi-view multi-label feature selection",
        ))
        repository.add_paper(PaperMetadata(
            paper_id="paper-2",
            title="Uncertainty-Aware Global-View Reconstruction for Multi-View Multi-Label Feature Selection",
        ))
        session.commit()
        results = PaperDownloadCoordinator(
            session,
            download_function=fake_download,
            output_dir=str(tmp_path),
        ).download_many(["paper-1", "paper-2"])
        session.commit()

    assert [item["success"] for item in results] == [True, False]
    database.dispose()


def test_reader_preparation_backfills_failed_top_papers(tmp_path):
    database = DatabaseManager(f"sqlite:///{(tmp_path / 'papers.db').as_posix()}")
    attempted = []

    def fake_download(title, *, output_dir, **kwargs):
        attempted.append(title)
        if title in {"Paper 1", "Paper 2"}:
            return {
                "success": False,
                "file_path": None,
                "error": "publisher did not provide a PDF",
                "engine_used": None,
            }
        path = Path(output_dir) / f"{title.replace(' ', '_')}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        return {
            "success": True,
            "file_path": str(path),
            "error": None,
            "engine_used": "direct",
        }

    service = LiteratureService(database=database, download_function=fake_download)
    ranking_result = {
        "papers": [
            {"paper_id": f"paper-{index}", "title": f"Paper {index}"}
            for index in range(1, 5)
        ]
    }

    resources = service.prepare_for_reader(
        ranking_result,
        target_count=2,
        output_dir=str(tmp_path / "pdfs"),
    )

    assert {resource.paper.paper_id for resource in resources} == {"paper-3", "paper-4"}
    assert set(attempted[:2]) == {"Paper 1", "Paper 2"}
    assert set(attempted[2:]) == {"Paper 3", "Paper 4"}
    assert len(service.last_download_results) == 4
    assert sum(item["success"] for item in service.last_download_results) == 2
    database.dispose()
