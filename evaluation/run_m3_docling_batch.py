"""Run the existing Docling adapter on a frozen local PDF manifest.

Outputs are intentionally gitignored; gold labels should be made from the PDF
pages without reading these parser outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from rag_reader.docling_parser import DoclingPDFParser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/gold/m3_pdf_layout/manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/m3/layout_batch"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    adapter = DoclingPDFParser()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in manifest["documents"]:
        pdf = Path(entry["pdf"])
        dest = args.out_dir / f"{entry['id']}_docling.json"
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if args.skip_existing and dest.is_file():
            print(json.dumps({"id": entry["id"], "status": "skipped_existing"}), flush=True)
            continue
        started = perf_counter()
        try:
            records = adapter.parse(pdf)
            elapsed = perf_counter() - started
            dest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            row = {"id": entry["id"], "status": "completed", "pdf_sha256": digest,
                   "seconds": round(elapsed, 3), "records": len(records),
                   "table_candidates": sum(record["metadata"]["item_type"] == "table" for record in records)}
        except Exception as exc:
            row = {"id": entry["id"], "status": "failed", "pdf_sha256": digest,
                   "seconds": round(perf_counter() - started, 3), "error": repr(exc)}
        results.append(row)
        (args.out_dir / "run_summary.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
