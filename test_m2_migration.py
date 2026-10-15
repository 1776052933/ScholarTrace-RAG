"""Exercise the real SQLite backup and idempotent data-import path."""
from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from literature_database.models import Base, PaperFileModel, PaperModel
from scripts.migrate_sqlite_to_mysql import backup_sqlite, import_tables


def test_backup_and_import_can_be_repeated(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    source_path = tmp_path / "legacy.sqlite3"
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "attention.pdf").write_bytes(b"%PDF-1.4\nexample")
    source_engine = create_engine(f"sqlite:///{source_path.as_posix()}")
    Base.metadata.create_all(source_engine)
    with Session(source_engine) as session:
        session.add(PaperModel(paper_id="arxiv:1706.03762", title="Attention Is All You Need", authors=["Vaswani"], year=2017))
        session.add(PaperFileModel(paper_id="arxiv:1706.03762", file_path=r"C:\old\papers\attention.pdf", status="AVAILABLE", file_type="pdf"))
        session.commit()
    source_engine.dispose()

    backup = backup_sqlite(source_path, tmp_path / "backups")
    assert backup.is_file() and backup != source_path
    target_path = tmp_path / "m2.sqlite3"
    target_url = f"sqlite:///{target_path.as_posix()}"
    config = Config(str(Path(__file__).with_name("alembic.ini")))
    config.set_main_option("sqlalchemy.url", target_url)
    command.upgrade(config, "head")

    assets_root = tmp_path / "m2_assets" / "papers"
    first = import_tables(backup, target_url, source_assets_root=tmp_path, assets_root=assets_root)
    second = import_tables(backup, target_url, source_assets_root=tmp_path, assets_root=assets_root)
    assert first["papers"] == {"copied": 1, "already_present": 0}
    assert first["paper_files"] == {"copied": 1, "already_present": 0}
    assert second["papers"] == {"copied": 0, "already_present": 1}
    assert second["paper_files"] == {"copied": 0, "already_present": 1}
    target_engine = create_engine(target_url)
    with Session(target_engine) as session:
        paper = session.get(PaperModel, "arxiv:1706.03762")
        assert paper is not None and paper.year == 2017
        copied_path = Path(session.scalars(select(PaperFileModel)).one().file_path)
        assert copied_path.is_file()
        assert copied_path.read_bytes() == b"%PDF-1.4\nexample"
    target_engine.dispose()
