"""Download and load the official LitSearch query split."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import pyarrow.parquet as parquet
import requests


QUERY_DATA_URL = (
    "https://huggingface.co/datasets/princeton-nlp/LitSearch/resolve/main/"
    "query/full-00000-of-00001.parquet"
)


def ensure_query_dataset(path: Path, timeout: int = 60) -> Path:
    """Download the small official query file when it is not present."""
    path = path.resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(QUERY_DATA_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        output.write(chunk)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_queries(
    path: Path,
    *,
    query_set: str | None = None,
    minimum_quality: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load LitSearch rows and attach stable local query identifiers."""
    rows = parquet.read_table(path).to_pylist()
    result = []
    for index, row in enumerate(rows):
        if query_set and str(row.get("query_set") or "") != query_set:
            continue
        quality = int(row.get("quality") or 0)
        if minimum_quality is not None and quality < minimum_quality:
            continue
        result.append({
            "query_id": f"litsearch_{index:04d}",
            "row_index": index,
            "query_set": str(row.get("query_set") or ""),
            "query": str(row.get("query") or "").strip(),
            "specificity": int(row.get("specificity") or 0),
            "quality": quality,
            "gold_corpus_ids": [int(value) for value in row.get("corpusids") or []],
        })
    if shuffle:
        random.Random(seed).shuffle(result)
    return result

