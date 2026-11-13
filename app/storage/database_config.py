"""Resolve one database target for API, workers and standalone paper services."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.engine import make_url


def data_root(value: str | Path | None = None) -> Path:
    return Path(value or os.getenv("AUTO_PAPER_DATA_ROOT") or "data/m2").expanduser().resolve()


def database_url(value: str | None = None, *, root: str | Path | None = None) -> str:
    """Prefer an explicit URL, then DATABASE_URL, then the shared local SQLite file.

    Compose sets SCHOLARTRACE_DATABASE_BACKEND=mysql. A mismatched URL must fail
    during startup instead of silently opening a second SQLite database.
    """
    supplied = value if value is not None else os.getenv("DATABASE_URL")
    if supplied is None or not supplied.strip():
        target = data_root(root)
        target.mkdir(parents=True, exist_ok=True)
        supplied = f"sqlite:///{(target / 'app.sqlite3').as_posix()}"
    resolved = supplied.strip()
    actual = make_url(resolved).get_backend_name()
    # Explicit URLs are deliberate overrides (also used by isolated SQLite tests).
    expected = os.getenv("SCHOLARTRACE_DATABASE_BACKEND", "").strip().lower() if value is None else ""
    if expected and actual != expected:
        raise ValueError(f"expected {expected} database, configured {actual}; check DATABASE_URL")
    return resolved
