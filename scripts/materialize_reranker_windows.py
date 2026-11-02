"""Expose the pinned Linux Hugging Face reranker cache as Windows hard links.

Docker-created snapshot symlinks appear as zero-byte reparse points to native
Windows Python. Hard links point to the same blobs without duplicating weights.
This script only creates missing files and refuses to replace existing ones.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/m3/reranker_hf_cache/hub/models--BAAI--bge-reranker-v2-m3"
SNAPSHOT = CACHE / "snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
DEST = ROOT / "data/m3/reranker_windows_model"
BLOBS = {
    "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "b1879d702821e753ffe4245048eee415d54a9385",
    "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    "tokenizer_config.json": "328a00a9a560aadcf2a3064f917517359eb3cc26",
}


def main():
    if os.name != "nt":
        raise SystemExit("Windows-only materialization; Linux can read snapshot symlinks")
    sources = {name: CACHE / "blobs" / blob for name, blob in BLOBS.items()}
    sources["config.json"] = SNAPSHOT / "config.json"
    for name, source in sources.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty source for {name}: {source}")
    DEST.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        output = DEST / name
        if output.exists():
            if output.stat().st_size != source.stat().st_size:
                raise ValueError(f"Existing {output} differs in size; refusing to replace it")
            continue
        if name == "config.json":
            shutil.copy2(source, output)
        else:
            os.link(source, output)
        print(f"ready {name}: {output.stat().st_size} bytes")
    print(f"Windows reranker ready: {DEST}")


if __name__ == "__main__":
    main()
