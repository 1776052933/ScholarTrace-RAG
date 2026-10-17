"""Compare generated answers from frozen PyMuPDF and Docling Top-5 evidence.

Uses the existing PaperQA prompt and DeepSeek client. One result file is written
after each API call so reruns do not duplicate paid requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_reader.llm_client import DeepSeekClient
from rag_reader.qa import PaperQA
from rag_reader.schema import DocumentChunk


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


class CachedRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def retrieve(self, paper_id: str, question: str, *, top_k: int) -> list[DocumentChunk]:
        return [DocumentChunk(chunk_id=item["chunk_id"], paper_id=paper_id,
                              text=item["text"], page=item["page"], section="",
                              metadata={"title": paper_id, "retrieval_score": item["score"]})
                for item in self.results[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-report", type=Path, default=Path("evaluation/runs/m3_parser_retrieval_10pdf_20261017/report.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/runs/m3_parser_answers_10pdf_20261017"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all frozen questions")
    args = parser.parse_args()
    report_hash = hashlib.sha256(args.retrieval_report.read_bytes()).hexdigest()
    report = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    client = DeepSeekClient()
    manifest = {"task": "m3_parser_answer_ablation", "retrieval_report_sha256": report_hash,
                "model": client.model, "top_k": 5, "question_count": len(report["records"]),
                "variants": ["pymupdf", "docling"]}
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("Existing answer run has different input/model settings")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = report["records"] if args.limit == 0 else report["records"][:args.limit]
    for index, item in enumerate(selected, 1):
        for variant in ("pymupdf", "docling"):
            output = args.out_dir / f"{index:03d}_{item['document_id']}_{item['question_id']}_{variant}.json"
            if output.is_file():
                print(f"{index}/{len(selected)} {variant} skipped_existing", flush=True)
                continue
            qa = PaperQA(CachedRetriever(item["variants"][variant]["top_10"]), llm_client=client, top_k=5)
            started = perf_counter()
            try:
                answer = qa.ask(item["document_id"], item["question"])
                value = {
                    "status": "completed", "document_id": item["document_id"],
                    "question_id": item["question_id"], "question": item["question"],
                    "split": item["split"], "variant": variant,
                    "gold_page": item["gold_page"], "answer_fragments": item["answer_fragments"],
                    "answer": answer.answer,
                    "answer_contains_gold_fragments": all(normalized(part) in normalized(answer.answer)
                                                         for part in item["answer_fragments"]),
                    "cited_gold_page": any(citation.page == item["gold_page"] for citation in answer.citations),
                    "sources": answer.sources,
                    "citations": [citation.model_dump(mode="json") for citation in answer.citations],
                    "confidence": answer.confidence,
                    "token_usage": client.last_usage.to_dict(),
                    "seconds": round(perf_counter() - started, 4),
                }
            except Exception as exc:
                value = {"status": "failed", "document_id": item["document_id"],
                         "question_id": item["question_id"], "split": item["split"],
                         "variant": variant, "error": f"{type(exc).__name__}: {exc}",
                         "seconds": round(perf_counter() - started, 4)}
            output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"item": index, "variant": variant, "status": value["status"],
                              "seconds": value["seconds"]}, ensure_ascii=False), flush=True)
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.out_dir.glob("[0-9][0-9][0-9]_*.json"))]
    summary = {}
    for variant in ("pymupdf", "docling"):
        entries = [row for row in rows if row["variant"] == variant]
        completed = [row for row in entries if row["status"] == "completed"]
        summary[variant] = {"attempted": len(entries), "completed": len(completed),
                            "answer_contains_gold_fragments": sum(row["answer_contains_gold_fragments"] for row in completed),
                            "cited_gold_page": sum(row["cited_gold_page"] for row in completed),
                            "total_tokens": sum(row["token_usage"]["total_tokens"] for row in completed)}
    output = {"manifest": manifest, "summary": summary,
              "metric_note": "Exact answer-fragment match is an automatic proxy; qualitative answers and citations require manual review."}
    (args.out_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
