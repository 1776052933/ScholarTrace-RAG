"""Offline translations, matched to exact originals without modifying review data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CATALOG = Path(__file__).parent / "reviews" / "m6_zh_translations.json"


def load_translations(path: Path = CATALOG) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != "m6-zh-translations-v1":
        raise ValueError("Unsupported translation catalog")
    return payload["translations"]


def translate(text: str, catalog: dict[str, dict[str, str]]) -> str | None:
    """Never reuse a translation if even one character of its source changed."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    item = catalog.get(digest)
    return item["zh"] if item else None


def reference_translation(text: str, catalog: dict[str, dict[str, str]]) -> str:
    return " | ".join(translate(part.strip(), catalog) or part.strip()
                      for part in text.split("|"))
