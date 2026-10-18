"""Compare no-key cross-paper retrieval candidates on frozen dev or holdout cases.

Do not inspect holdout output until a single candidate and its settings have
been chosen from dev and documented. This program never uses gold in retrieval.
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


VARIANTS = ("plain_dense", "existing_graph", "balanced_graph",
            "per_paper_2", "per_paper_3", "per_paper_5")
INDEX_DIR = Path("evaluation/runs/m3_parser_retrieval_10pdf_20261017")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate(retriever: PaperRetriever, graph: EvidenceGraph, name: str,
               paper_ids: list[str], question: str):
    if name == "plain_dense":
        return retriever.retrieve_multiple(paper_ids, question, top_k=5), None
    if name == "existing_graph":
        return graph.run(paper_ids, question)
    if name == "balanced_graph":
        return EvidenceGraph(retriever, top_k=5, strategy="balanced", max_chunks=10).run(
            paper_ids, question)
    per_paper = int(name.rsplit("_", 1)[1])
    chunks = []
    seen = set()
    for paper_id in paper_ids:
        for chunk in retriever.retrieve(paper_id, question, top_k=per_paper):
            if chunk.chunk_id not in seen:
                chunks.append(chunk)
                seen.add(chunk.chunk_id)
    return chunks, {"strategy": name, "max_chunks": per_paper * len(paper_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="m6_dev_pairs.json")
    parser.add_argument("--include-original-dev", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embedding-model", default="/app/data/m3/bge_m3_model")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    names = [name.strip() for name in args.variants.split(",")]
    if not names or len(set(names)) != len(names) or any(name not in VARIANTS for name in names):
        parser.error(f"variants must be distinct choices from {VARIANTS}")
    cases, hashes = load_multi_cases(cases_name=args.cases, require_both_splits=False)
    splits = {case["split"] for case in cases}
    if splits not in ({"dev"}, {"test"}):
        parser.error("candidate run must use only one split")
    if args.include_original_dev:
        if splits != {"dev"}:
            parser.error("original dev questions cannot be added to holdout")
        original, original_hashes = load_multi_cases()
        cases = [case for case in original if case["split"] == "dev"] + cases
        hashes["original_multi_cases"] = original_hashes["multi_cases"]
    source_manifest = json.loads((args.index_dir / "manifest.json").read_text(encoding="utf-8"))
    for key, gold_key in (("pdf_manifest_sha256", "pdf_manifest"),
                          ("batch_labels_sha256", "batch_labels"),
                          ("attention_label_sha256", "attention_label")):
        if source_manifest.get(key) != hashes[gold_key]:
            parser.error(f"source index differs from frozen labels: {key}")
    if source_manifest.get("embedding_model") != args.embedding_model:
        parser.error("source index used a different embedding model")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error(f"run directory must be empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": "m6_candidate_retrieval", "split": next(iter(splits)),
                "cases_file": args.cases, "cases_hashes": hashes,
                "case_ids": [case["id"] for case in cases],
                "source_index_manifest": source_manifest, "variants": names,
                "note": "Evidence labels are used only for scoring after retrieval; answer generation is not evaluated."}
    _write(args.run_dir / "manifest.json", manifest)

    model = EmbeddingModel(model_name=args.embedding_model, device="cpu")
    model.embed("candidate retrieval warmup")
    records = []
    for parser_name in ("pymupdf", "docling"):
        store = VectorStore(collection_name=f"m3_layout_{parser_name}",
                            persist_directory=str(args.index_dir / "chroma"))
        if not store.collection.count():
            raise RuntimeError(f"empty source index: {parser_name}")
        retriever = PaperRetriever(model, store, default_top_k=5)
        graph = EvidenceGraph(retriever, top_k=5)
        for number, case in enumerate(cases):
            ids = [ref["document_id"] for ref in case["references"]]
            record = {"parser": parser_name, "case_id": case["id"],
                      "split": case["split"], "question": case["question"],
                      "references": case["references"], "variants": {}}
            order = names if number % 2 == 0 else list(reversed(names))
            for name in order:
                started = perf_counter()
                try:
                    chunks, info = _candidate(retriever, graph, name, ids, case["question"])
                    record["variants"][name] = {
                        "status": "completed", "seconds": round(perf_counter() - started, 4),
                        "metrics": score_evidence([chunk.model_dump(mode="json") for chunk in chunks],
                                                  case["references"]),
                        "info": info,
                        "chunks": [{"chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id,
                                    "page": chunk.page, "text": chunk.text} for chunk in chunks],
                    }
                except Exception as exc:
                    record["variants"][name] = {
                        "status": "failed", "seconds": round(perf_counter() - started, 4),
                        "error": f"{type(exc).__name__}: {exc}"}
            records.append(record)
            print(f"{parser_name} {case['id']}", flush=True)
    summary = {parser_name: summarize_variants(
        [row for row in records if row["parser"] == parser_name], names)
        for parser_name in ("pymupdf", "docling")}
    _write(args.run_dir / "report.json", {"manifest": manifest,
                                          "summary": summary, "records": records})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
