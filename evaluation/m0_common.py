"""Shared reproducibility helpers for the M0 baseline runners."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
import json
from pathlib import Path
import subprocess
from typing import Any

from rag_reader.embedding import EmbeddingModel, _HashingEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks"
RUNS = ROOT / "evaluation" / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def code_hash() -> str:
    files = [
        *sorted((ROOT / "rag_reader").glob("*.py")),
        *sorted((ROOT / "evaluation").glob("m0*.py")),
        *sorted((ROOT / "evaluation").glob("run_m0*.py")),
        ROOT / "evaluation" / "metrics.py",
        ROOT / "evaluation" / "litsearch.py",
        ROOT / "literature_database" / "schema.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def require_real_embedding(model: EmbeddingModel) -> str:
    backend = model._get_model()
    if isinstance(backend, _HashingEncoder):
        raise RuntimeError(
            "BGE-M3 failed to load and the application fell back to hashing vectors; "
            "M0 baseline would be invalid. Check model availability."
        )
    return type(backend).__name__


def runtime_info() -> dict[str, Any]:
    packages = {}
    for name in ("chromadb", "sentence-transformers", "torch", "pyarrow", "langchain-text-splitters", "openai"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages}


def embedding_revision(model: EmbeddingModel) -> str | None:
    try:
        config = model._get_model()[0].auto_model.config
        return getattr(config, "_commit_hash", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def new_run_dir(task: str, requested: str | None) -> Path:
    path = Path(requested).resolve() if requested else RUNS / f"m0_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize(records: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    metrics = {}
    for name in names:
        values = [float(record["metrics"][name]) for record in completed if record["metrics"].get(name) is not None]
        metrics[name] = sum(values) / len(values) if values else None
    return {
        "question_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "macro_metrics": metrics,
        "total_seconds": sum(record.get("total_seconds", 0.0) for record in records),
    }


