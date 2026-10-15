"""Back up an M1 SQLite database and import its paper records into migrated MySQL.

Usage: python -m scripts.migrate_sqlite_to_mysql --source Auto_Paper_UI/runtime/literature.db --target-url mysql+pymysql://...
The destination must first run `python -m alembic upgrade head`.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import shutil
import sqlite3
from uuid import uuid4

from sqlalchemy import MetaData, Table, create_engine, inspect, select


TABLES = ("papers", "paper_files", "download_tasks", "upload_queue")


def backup_sqlite(source: Path, backup_dir: Path) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"{source.stem}_{stamp}_{uuid4().hex[:6]}.sqlite3"
    with sqlite3.connect(str(source)) as src, sqlite3.connect(str(destination)) as dst:
        src.backup(dst)
    return destination


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_asset(raw_path: str, paper_id: str, file_id: int, source_root: Path, assets_root: Path) -> str:
    candidate = Path(raw_path)
    if not candidate.is_file():
        parts = raw_path.replace("\\", "/").split("/")
        paper_positions = [index for index, part in enumerate(parts) if part.casefold() == "papers"]
        if not paper_positions:
            raise FileNotFoundError(f"cannot resolve migrated paper asset: {raw_path}")
        candidate = source_root.joinpath(*parts[paper_positions[-1]:])
    if not candidate.is_file():
        raise FileNotFoundError(f"paper asset is missing: {raw_path}")
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in paper_id)
    digest = _file_hash(candidate)[:12]
    destination = assets_root / safe_id / f"{file_id}_{digest}{candidate.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _file_hash(destination) != _file_hash(candidate):
            raise RuntimeError(f"asset collision at {destination}")
    else:
        shutil.copy2(candidate, destination)
    return str(destination.resolve())


def import_tables(
    backup: Path,
    target_url: str,
    *,
    source_assets_root: Path | None = None,
    assets_root: Path | None = None,
) -> dict[str, dict[str, int]]:
    source_engine = create_engine(f"sqlite:///{backup.resolve().as_posix()}")
    target_engine = create_engine(target_url, pool_pre_ping=True)
    counts: dict[str, dict[str, int]] = {}
    try:
        source_names = set(inspect(source_engine).get_table_names())
        target_names = set(inspect(target_engine).get_table_names())
        for name in TABLES:
            if name not in source_names or name not in target_names:
                raise RuntimeError(f"table {name} missing; migrate target schema first")
        source_meta, target_meta = MetaData(), MetaData()
        with source_engine.connect() as source, target_engine.begin() as target:
            for name in TABLES:
                source_table = Table(name, source_meta, autoload_with=source_engine)
                target_table = Table(name, target_meta, autoload_with=target_engine)
                primary_keys = [column.name for column in source_table.primary_key.columns]
                copied = skipped = 0
                for source_row in source.execute(select(source_table)).mappings():
                    row = dict(source_row)
                    if name == "paper_files" and assets_root is not None:
                        if source_assets_root is None:
                            raise ValueError("source_assets_root is required when copying assets")
                        row["file_path"] = _copy_asset(
                            row["file_path"], row["paper_id"], row["id"],
                            source_assets_root, assets_root,
                        )
                    predicate = [target_table.c[key] == row[key] for key in primary_keys]
                    existing = target.execute(select(target_table).where(*predicate)).mappings().first()
                    if existing is not None:
                        if any(existing[key] != value for key, value in row.items()):
                            raise RuntimeError(f"conflicting {name} primary key {tuple(row[key] for key in primary_keys)}")
                        skipped += 1
                        continue
                    target.execute(target_table.insert().values(**row))
                    copied += 1
                counts[name] = {"copied": copied, "already_present": skipped}
    finally:
        source_engine.dispose()
        target_engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target-url", default=os.getenv("DATABASE_URL"), help="Destination DATABASE_URL; defaults to environment variable")
    parser.add_argument("--backup-dir", default="data/m2/backups", type=Path)
    parser.add_argument("--assets-root", type=Path, help="Copy PDF/text assets here and rewrite stored paths")
    args = parser.parse_args()
    if not args.target_url:
        parser.error("set DATABASE_URL or pass --target-url")
    backup = backup_sqlite(args.source, args.backup_dir)
    print(f"SQLite backup: {backup}")
    counts = import_tables(backup, args.target_url, source_assets_root=args.source.parent, assets_root=args.assets_root)
    for table, result in counts.items():
        print(f"{table}: copied={result['copied']}, already_present={result['already_present']}")


if __name__ == "__main__":
    main()
