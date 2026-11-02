"""Entry point for the ScholarTrace command-line interface."""

from __future__ import annotations

import sys

from app.cli import main


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
