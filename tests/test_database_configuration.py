"""All local entry points must resolve to the same database target."""

from __future__ import annotations

import pytest

from app.api.main import create_app
from app.papers.database import DatabaseManager
from app.storage.database_config import database_url
from app.tasks import _service


def test_api_worker_and_paper_service_share_local_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SCHOLARTRACE_DATABASE_BACKEND", raising=False)
    monkeypatch.setenv("AUTO_PAPER_DATA_ROOT", str(tmp_path / "data"))
    expected = database_url()
    standalone = DatabaseManager()
    app = create_app()
    worker = _service()
    try:
        assert expected.endswith("/data/app.sqlite3")
        assert standalone.database_url == app.state.database.database_url == worker.database.database_url == expected
    finally:
        standalone.dispose()
        app.state.database.dispose()
        worker.database.dispose()


def test_compose_backend_guard_rejects_accidental_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHOLARTRACE_DATABASE_BACKEND", "mysql")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="expected mysql database"):
        database_url(root=tmp_path)
    with pytest.raises(ValueError, match="expected mysql database"):
        create_app(data_root=tmp_path)
    # An explicit database URL remains available for an isolated test environment.
    assert database_url(f"sqlite:///{(tmp_path / 'fixture.sqlite3').as_posix()}").startswith("sqlite:///")
