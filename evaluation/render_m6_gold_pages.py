"""Render frozen cross-paper gold pages for independent visual review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf

from .m6_metrics import load_multi_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/runs/m6_pdf_review"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    args = parser.parse_args()
    cases, _ = load_multi_cases()
    manifest = json.loads(Path("evaluation/gold/m3_pdf_layout/manifest.json").read_text(encoding="utf-8"))
    paths = {row["id"]: Path(row["pdf"]) for row in manifest["documents"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = set()
    for case in cases:
        if args.split != "all" and case["split"] != args.split:
            continue
        for ref in case["references"]:
            key = (ref["document_id"], ref["page"])
            if key in saved:
                continue
            saved.add(key)
            with pymupdf.open(paths[key[0]]) as document:
                page_number = key[1] - 1
                if page_number < 0 or page_number >= len(document):
                    raise ValueError(f"Gold page out of range: {key}")
                output = args.output_dir / f"{key[0]}_page_{key[1]:03d}.png"
                document[page_number].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False).save(output)
                print(output, flush=True)


if __name__ == "__main__":
    main()
