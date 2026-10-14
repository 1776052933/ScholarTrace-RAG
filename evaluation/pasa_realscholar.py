"""Load the official PaSa RealScholarQuery test set."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any


PASA_DATASET_REPO = "CarlanLark/pasa-dataset"
PASA_DATASET_FILE = "RealScholarQuery/test.jsonl"


class DatasetAccessError(RuntimeError):
    """Raised when the gated PaSa dataset cannot be downloaded."""


def ensure_real_scholar_dataset(path: Path) -> Path:
    """Download the small gated test file after the user grants HF access."""
    path = path.expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    token = os.getenv("HF_TOKEN", "").strip() or None
    try:
        from huggingface_hub import hf_hub_download

        cached_path = hf_hub_download(
            repo_id=PASA_DATASET_REPO,
            filename=PASA_DATASET_FILE,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        raise DatasetAccessError(
            "无法下载 PaSa RealScholarQuery。请先在 "
            "https://huggingface.co/datasets/CarlanLark/pasa-dataset "
            "登录并同意数据集条款，然后设置 HF_TOKEN；也可以使用 "
            "--dataset-path 指向手动下载的 RealScholarQuery/test.jsonl。"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    shutil.copyfile(cached_path, temporary)
    temporary.replace(path)
    return path


def load_real_scholar_queries(
    path: Path,
    *,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Read PaSa JSONL rows while preserving stable source row IDs."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            value = json.loads(line)
            question = str(value.get("question") or "").strip()
            answers = _answer_titles(value.get("answer"))
            if not question or not answers:
                continue
            source_meta = value.get("source_meta")
            if not isinstance(source_meta, dict):
                source_meta = {}
            published_time = str(source_meta.get("published_time") or "").strip()
            rows.append({
                "query_id": f"pasa_real_{index:03d}",
                "row_index": index,
                "query": question,
                "gold_titles": answers,
                "source_meta": source_meta,
                "published_time": published_time or None,
                "pasa_search_cutoff": _pasa_cutoff(published_time),
            })
    if shuffle:
        random.Random(seed).shuffle(rows)
    return rows


def _answer_titles(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    titles = []
    for item in value:
        if isinstance(item, str):
            title = item.strip()
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("paper_title") or "").strip()
        else:
            title = ""
        if title:
            titles.append(title)
    return titles


def _pasa_cutoff(value: str) -> str | None:
    try:
        return (datetime.strptime(value, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
    except (TypeError, ValueError):
        return None
