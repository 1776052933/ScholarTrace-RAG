"""Fixed-question parser ablation with the project's cleaner, splitter, BGE-M3 and Chroma dense search.

Only the PDF parser output differs between variants. No gold answers enter the
index; this script does not call a generation API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from rag_reader.cleaner import TextCleaner
from rag_reader.embedding import EmbeddingModel
from rag_reader.parser import PDFParser
from rag_reader.splitter import DocumentSplitter
from rag_reader.vector_store import VectorStore


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def hit(result: dict[str, Any], question: dict[str, Any], *, require_row: bool) -> bool:
    page = result.get("metadata", {}).get("page")
    if page != question["page"]:
        return False
    body = norm(result.get("text") or "")
    if not all(norm(part) in body for part in question["answer_fragments"]):
        return False
    return not require_row or not question.get("row_contains") or norm(question["row_contains"]) in body


def summary(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    rows = [item["variants"][variant] for item in records]
    times = sorted(item["query_seconds"] for item in rows)
    return {
        "questions": len(rows),
        "correct_page_at_5": sum(item["correct_page_at_5"] for item in rows),
        "answer_fragment_at_5": sum(item["answer_fragment_at_5"] for item in rows),
        "row_anchor_and_answer_at_5": sum(item["row_anchor_and_answer_at_5"] for item in rows),
        "answer_fragment_at_10": sum(item["answer_fragment_at_10"] for item in rows),
        "mean_reciprocal_rank_answer_at_10": sum(item["reciprocal_rank_answer_at_10"] for item in rows) / len(rows),
        "query_p50_seconds": median(times),
        "query_p95_seconds": times[max(0, int(0.95 * (len(times) - 1)))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/gold/m3_pdf_layout/manifest.json"))
    parser.add_argument("--batch-labels", type=Path, default=Path("evaluation/gold/m3_pdf_layout/batch_labels.json"))
    parser.add_argument("--attention-label", type=Path, default=Path("evaluation/gold/m3_pdf_layout/attention_is_all_you_need.json"))
    parser.add_argument("--docling-dir", type=Path, default=Path("data/m3/layout_batch"))
    parser.add_argument("--embedding-model", default="/app/data/m3/bge_m3_model")
    parser.add_argument("--run-dir", type=Path, default=Path("evaluation/runs/m3_parser_retrieval_10pdf_20261017"))
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    label_set = json.loads(args.batch_labels.read_text(encoding="utf-8"))
    labels = {item["id"]: item for item in label_set["documents"]}
    labels["attention"] = json.loads(args.attention_label.read_text(encoding="utf-8"))
    if set(labels) != {item["id"] for item in manifest["documents"]}:
        parser.error("Label IDs do not match the frozen PDF manifest")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.run_dir / "manifest.json"
    input_fingerprint = {
        "task": "m3_parser_retrieval_ten_pdf",
        "pdf_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "batch_labels_sha256": hashlib.sha256(args.batch_labels.read_bytes()).hexdigest(),
        "attention_label_sha256": hashlib.sha256(args.attention_label.read_bytes()).hexdigest(),
        "embedding_model": args.embedding_model,
        "embedding_weights_sha256": hashlib.sha256((Path(args.embedding_model) / "model.safetensors").read_bytes()).hexdigest(),
        "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
        "search_backend": "Chroma cosine dense", "top_k": 5, "candidate_k": 10,
        "questions": [(item["id"], question["id"])
                      for item in manifest["documents"] for question in labels[item["id"]]["evidence_questions"]],
    }
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != input_fingerprint:
        parser.error("Existing run directory has a different input fingerprint")
    manifest_path.write_text(json.dumps(input_fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = EmbeddingModel(model_name=args.embedding_model, device="cpu", batch_size=16)
    fingerprint = model.fingerprint()
    cleaner = TextCleaner()
    splitter = DocumentSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    stores = {name: VectorStore(collection_name=f"m3_layout_{name}", persist_directory=str(args.run_dir / "chroma"))
              for name in ("pymupdf", "docling")}
    progress_path = args.run_dir / "index_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    for entry in manifest["documents"]:
        pdf = Path(entry["pdf"])
        doc_id = entry["id"]
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != labels[doc_id]["pdf_sha256"]:
            parser.error(f"PDF differs from gold: {doc_id}")
        for name, store in stores.items():
            key = f"{name}:{doc_id}"
            if key in progress:
                continue
            started = perf_counter()
            raw = (PDFParser().parse(pdf) if name == "pymupdf" else
                   json.loads((args.docling_dir / f"{doc_id}_docling.json").read_text(encoding="utf-8")))
            cleaned = cleaner.clean_page_records(raw)
            chunks = splitter.split_pages(doc_id, cleaned, title=doc_id)
            if not chunks:
                raise RuntimeError(f"No chunks for {key}")
            vectors = model.embed_chunks(chunks)
            store.ensure_embedding_compatible(fingerprint, len(vectors[0]))
            store.delete(paper_id=doc_id)
            store.add_documents(chunks, vectors)
            progress[key] = {"chunks": len(chunks), "seconds": round(perf_counter() - started, 3)}
            progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"indexed": key, **progress[key]}), flush=True)

    records = []
    for entry in manifest["documents"]:
        doc_id = entry["id"]
        for question in labels[doc_id]["evidence_questions"]:
            vector = model.embed(question["question"])
            variants = {}
            for name, store in stores.items():
                started = perf_counter()
                results = store.search(vector, top_k=10, paper_id=doc_id)
                elapsed = perf_counter() - started
                answer_ranks = [i + 1 for i, result in enumerate(results) if hit(result, question, require_row=False)]
                variants[name] = {
                    "query_seconds": round(elapsed, 5),
                    "correct_page_at_5": any(item.get("metadata", {}).get("page") == question["page"] for item in results[:5]),
                    "answer_fragment_at_5": any(hit(item, question, require_row=False) for item in results[:5]),
                    "row_anchor_and_answer_at_5": any(hit(item, question, require_row=True) for item in results[:5]),
                    "answer_fragment_at_10": bool(answer_ranks),
                    "reciprocal_rank_answer_at_10": 1 / answer_ranks[0] if answer_ranks else 0.0,
                    "top_10": [{"chunk_id": item["chunk_id"], "page": item["metadata"].get("page"),
                                "score": item["score"], "text": item["text"]} for item in results],
                }
            records.append({"document_id": doc_id, "split": entry["split"],
                            "question_id": question["id"], "question": question["question"],
                            "gold_page": question["page"], "answer_fragments": question["answer_fragments"],
                            "variants": variants})
    result = {"manifest": input_fingerprint, "index_progress": progress,
              "summary": {name: summary(records, name) for name in stores},
              "splits": {split: {name: summary([row for row in records if row["split"] == split], name)
                                 for name in stores} for split in ("dev", "test")},
              "records": records,
              "limits": ["First-pass PDF labels need independent second review.",
                         "This is dense retrieval on Chroma, not Milvus hybrid or generated answer evaluation.",
                         "Answer-fragment matching checks supplied text and page; it cannot verify table cell coordinates."]}
    output = args.run_dir / "report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": result["summary"],
                      "splits": result["splits"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
