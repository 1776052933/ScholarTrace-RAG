"""Compare plain Top-5 retrieval with the bounded evidence graph on frozen PDFs.

This reads an existing M3 Chroma index and performs no generation or indexing.
The four cross-paper cases are derived from first-pass labels, so scores are
diagnostic until a person independently reviews their source PDF pages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from rag_reader.embedding import EmbeddingModel
from rag_reader.evidence_graph import EvidenceGraph
from rag_reader.retriever import PaperRetriever
from rag_reader.vector_store import VectorStore

from .m6_metrics import load_multi_cases, score_evidence, summarize_variants


DEFAULT_INDEX = Path("evaluation/runs/m3_parser_retrieval_10pdf_20261017")
DEFAULT_RUN = Path("evaluation/runs/m6_multi_evidence_20261018")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--embedding-model", default="/app/data/m3/bge_m3_model")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k must be positive")
    cases, hashes = load_multi_cases()
    source = args.index_dir / "manifest.json"
    if not source.is_file():
        parser.error(f"missing source index manifest: {source}")
    source_manifest = json.loads(source.read_text(encoding="utf-8"))
    for source_key, gold_key in (("pdf_manifest_sha256", "pdf_manifest"),
                                 ("batch_labels_sha256", "batch_labels"),
                                 ("attention_label_sha256", "attention_label")):
        if source_manifest.get(source_key) != hashes[gold_key]:
            parser.error(f"source index differs from frozen labels: {source_key}")
    if not Path(args.embedding_model).is_dir():
        parser.error(f"missing local embedding model: {args.embedding_model}")
    if source_manifest.get("embedding_model") != args.embedding_model:
        parser.error("source index used a different embedding model path")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error(f"run directory must be empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": "m6_cross_paper_evidence", "case_hashes": hashes,
                "source_index_manifest": source_manifest,
                "top_k": args.top_k,
                "variants": ["plain_dense", "bounded_evidence_graph"],
                "notes": ["No answer generation or judge API calls.",
                          "Gold is first-pass and awaits independent human review.",
                          "Additional graph chunks increase the evidence budget."]}
    _write(args.run_dir / "manifest.json", manifest)

    model = EmbeddingModel(model_name=args.embedding_model, device="cpu")
    model.embed("M6 offline evaluation warmup")
    records = []
    for parser_name in ("pymupdf", "docling"):
        store = VectorStore(collection_name=f"m3_layout_{parser_name}",
                            persist_directory=str(args.index_dir / "chroma"))
        if not store.collection.count():
            raise RuntimeError(f"empty source index: {parser_name}")
        retriever = PaperRetriever(model, store, default_top_k=args.top_k)
        graph = EvidenceGraph(retriever, top_k=args.top_k)
        for case_number, case in enumerate(cases):
            ids = [ref["document_id"] for ref in case["references"]]
            record = {"parser": parser_name, "case_id": case["id"],
                      "split": case["split"], "question": case["question"],
                      "references": case["references"], "variants": {}}
            # Alternate order so a warm Chroma query does not always favor the graph.
            order = manifest["variants"] if case_number % 2 == 0 else list(reversed(manifest["variants"]))
            for name in order:
                started = perf_counter()
                try:
                    if name == "plain_dense":
                        chunks = retriever.retrieve_multiple(ids, case["question"], top_k=args.top_k)
                        graph_info = None
                    else:
                        chunks, graph_info = graph.run(ids, case["question"])
                    record["variants"][name] = {
                        "status": "completed", "seconds": round(perf_counter() - started, 4),
                        "metrics": score_evidence([chunk.model_dump(mode="json") for chunk in chunks],
                                                  case["references"]),
                        "graph": graph_info,
                        "chunks": [{"chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
                                    "page": chunk.page, "text": chunk.text}
                                   for chunk in chunks],
                    }
                except Exception as exc:
                    record["variants"][name] = {
                        "status": "failed", "seconds": round(perf_counter() - started, 4),
                        "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            print(f"{parser_name} {case['id']}", flush=True)
    summaries = {name: {split: summarize_variants(
        [row for row in records if row["parser"] == name and row["split"] == split],
        manifest["variants"])
        for split in ("dev", "test")}
        for name in ("pymupdf", "docling")}
    _write(args.run_dir / "report.json", {"manifest": manifest,
                                          "summary": summaries, "records": records})
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
