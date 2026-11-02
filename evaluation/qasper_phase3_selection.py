"""Compare fixed evidence selectors on saved QASPER candidate rankings.

No retrieval, model or gold information enters selection. The scorer loads
gold annotations only after every variant has selected its five chunk IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import statistics


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/benchmarks/qasper/qasper-dev-v0.3.json"
EVALUATOR = PROJECT / "data/benchmarks/qasper/qasper_evaluator.py"
RUNS = PROJECT / "evaluation/runs/qasper_followup_20261022"
OUT = RUNS / "phase3_selection"
VARIANTS = ("baseline5", "lexical_tail5", "section_tail5")
STOP = set("a an the and or of to in on for with by from this that which what who when where how many much do does did is are was were can could will would has have had should paper authors study method methods model models result results".split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}|\d+(?:\.\d+)?")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terms(text: str) -> set[str]:
    return {term.lower() for term in TOKEN.findall(text) if term.lower() not in STOP}


def lexical_score(question: str, chunk: dict) -> float:
    query = terms(question)
    if not query:
        return 0.0
    text = terms(chunk["text"])
    return len(query & text) / len(query)


def select(question: str, ranked: list[dict], variant: str) -> list[dict]:
    if len(ranked) < 5 or variant == "baseline5":
        return ranked[:5]
    if variant == "lexical_tail5":
        fifth = ranked[4]
        tail = max(enumerate(ranked[4:10], 4),
                   key=lambda pair: (lexical_score(question, pair[1]), -pair[0]))[1]
        # Only replace the fifth rank if the textual question match is stronger.
        chosen = tail if lexical_score(question, tail) > lexical_score(question, fifth) else fifth
        return [*ranked[:4], chosen]
    if variant == "section_tail5":
        used = {chunk.get("section", "").strip().casefold() for chunk in ranked[:4]}
        fifth_section = ranked[4].get("section", "").strip().casefold()
        if fifth_section and fifth_section not in used:
            return ranked[:5]
        for chunk in ranked[5:10]:
            section = chunk.get("section", "").strip().casefold()
            if section and section not in used and (
                "related work" not in section or
                bool(re.search(r"previous|prior|related|compare|other work", question, re.I))
            ):
                return [*ranked[:4], chunk]
        return ranked[:5]
    raise ValueError(variant)


def normalize(text: str) -> str:
    return " ".join(text.split())


def project(excerpts: list[str], paragraphs: list[str]) -> set[str]:
    selected = set()
    for excerpt in excerpts:
        text = normalize(excerpt)
        selected.update(normalize(p) for p in paragraphs if text and normalize(p) and
                        (normalize(p) in text or (len(text) >= 40 and text in normalize(p))))
    return selected


def official():
    spec = importlib.util.spec_from_file_location("qasper_official_phase3", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases(cohort: str) -> list[dict]:
    if cohort == "200":
        old = read(PROJECT / "evaluation/runs/qasper_qwen_dev200_20261021/manifest.json")
        return [{"question_id": qid} for qid in old["case_ids"]]
    if cohort == "holdout60":
        return read(RUNS / "holdout_60.json")["cases"]
    if cohort == "new_papers":
        return read(PROJECT / "evaluation/runs/qasper_new_papers_20261022/frozen_manifest.json")["cases"]
    raise ValueError(cohort)


def score(cohort: str, *, include_model: bool = False) -> dict:
    if cohort == "new_papers":
        variants = ("baseline5", "model_selected5") if include_model else ("baseline5",)
    else:
        variants = (*VARIANTS, "model_selected5") if include_model else VARIANTS
    source = read(SOURCE)
    gold = official().get_answers_and_evidence(source, False)
    question_map = {q["question_id"]: (pid, q["question"])
                    for pid, paper in source.items() for q in paper["qas"]}
    rows = []
    for case in cases(cohort):
        qid = case["question_id"]
        pid, question = question_map[qid]
        retrieval_root = (RUNS / "evidence_budget" / cohort / "retrieval" if cohort != "new_papers"
                          else PROJECT / "evaluation/runs/qasper_new_papers_20261022/retrieval")
        record = read(retrieval_root / f"{qid}.json")
        if record["status"] != "completed" or record["paper_id"] != pid:
            raise ValueError(f"Incomplete or mismatched retrieval: {qid}")
        paragraphs = [p for section in source[pid]["full_text"] for p in section["paragraphs"]]
        keys = {normalize(p) for p in paragraphs}
        groups = [{normalize(p) for p in ref["evidence"] if normalize(p) in keys}
                  for ref in gold[qid]]
        groups = [group for group in groups if group]
        row = {"question_id": qid, "paper_id": pid, "has_text_gold": bool(groups),
               "variants": {}}
        for variant in variants:
            if variant == "model_selected5":
                choice = read(RUNS / "phase3_model_selection" / cohort / f"{qid}.json")
                ids = choice.get("selected_ids") if choice["status"] == "completed" else None
                if ids:
                    by_id = {chunk["chunk_id"]: chunk for chunk in record["ranked"]}
                    chosen = [by_id[chunk_id] for chunk_id in ids]
                else:
                    chosen = record["ranked"][:5]
            else:
                chosen = select(question, record["ranked"], variant)
            matched = project([c["text"] for c in chosen], paragraphs)
            row["variants"][variant] = {
                "chunk_ids": [c["chunk_id"] for c in chosen],
                "hit": int(any(matched & group for group in groups)) if groups else None,
                "input_chars": len(json.dumps({"paper_id": pid, "question": question,
                    "chunks": [{k: c[k] for k in ("chunk_id", "paper_id", "section", "text")}
                               for c in chosen]}, ensure_ascii=False)),
            }
        rows.append(row)
    valid = [r for r in rows if r["has_text_gold"]]
    baseline = {r["question_id"]: r["variants"]["baseline5"]["hit"] for r in valid}
    summary = {"cohort": cohort, "questions": len(rows), "text_gold_questions": len(valid),
               "variants": {}, "note": "Fixed selectors on existing top-10 BGE rankings; gold only in scorer."}
    for variant in variants:
        eligible = [r["variants"][variant] for r in valid]
        summary["variants"][variant] = {
            "text_hit": sum(r["hit"] for r in eligible),
            "text_hit_rate": statistics.mean(r["hit"] for r in eligible),
            "gained_vs_baseline": sum(r["hit"] and not baseline[r0["question_id"]]
                                      for r0, r in zip(valid, eligible)),
            "lost_vs_baseline": sum(not r["hit"] and baseline[r0["question_id"]]
                                    for r0, r in zip(valid, eligible)),
            "changed_questions": sum(r["variants"][variant]["chunk_ids"] !=
                                     r["variants"]["baseline5"]["chunk_ids"] for r in rows),
            "mean_input_chars": statistics.mean(r["variants"][variant]["input_chars"]
                                                for r in rows),
        }
    write(OUT / cohort / "cases.json", rows)
    write(OUT / cohort / "scores.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["200", "holdout60", "new_papers"], required=True)
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    print(json.dumps(score(args.cohort, include_model=args.include_model), ensure_ascii=False, indent=2))
