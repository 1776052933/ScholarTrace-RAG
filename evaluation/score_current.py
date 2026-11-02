"""Local-only scoring of saved current-core results; never loads a model/store."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path

from .m0_common import DATA, sha256, write_json


def project_paragraphs(excerpts, paragraphs):
    """Conservative source-only containment projection, never gold-assisted.

    A whole paragraph inside an excerpt, or an excerpt >=40 characters wholly
    inside a paragraph, selects that original paragraph. Boundary fragments that
    satisfy neither condition remain unmatched rather than fuzzy-matching gold.
    """
    normalize = lambda s: " ".join(s.split())
    selected, unmatched = [], 0
    for excerpt in excerpts:
        text = normalize(excerpt)
        matches = [p for p in paragraphs if text and normalize(p) and (
            normalize(p) in text or (len(text) >= 40 and text in normalize(p)))]
        if not matches:
            unmatched += 1
        selected.extend(matches)
    return list(dict.fromkeys(selected)), unmatched


def retrieval_scores(ranked, gold, cutoffs):
    ranked = list(dict.fromkeys(ranked))
    gold = set(gold)
    if not gold:
        raise ValueError("Retrieval reference has no relevant papers")
    scores = {}
    for k in cutoffs:
        hits = [int(p in gold) for p in ranked[:k]]
        scores[f"Recall@{k}"] = sum(hits) / len(gold)
        scores[f"HitRate@{k}"] = float(any(hits))
        scores[f"MRR@{k}"] = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        scores[f"nDCG@{k}"] = dcg / ideal
    return scores


def score_run(directory: Path, *, phase="answer", data_root=DATA, cutoffs=(1, 5, 10, 20)):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    ids = manifest["case_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid selected question IDs")
    records = {}
    for path in sorted((directory / phase).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["case_id"]
        if key not in ids or key in records:
            raise ValueError("Unknown or duplicate prediction ID")
        records[key] = row
    statuses = Counter(records.get(key, {}).get("status", "pending") for key in ids)
    completed = {key: row for key, row in records.items() if row["status"] == "completed"}
    report = {"dataset": manifest["dataset"], "phase": phase, "questions": len(ids),
              "statuses": dict(statuses), "completed": len(completed),
              "coverage": len(completed) / len(ids), "quality_scores": None,
              "external_model_calls": 0, "manifest_sha256": sha256(directory / "manifest.json"),
              "denominator": "all selected questions; missing/failed predictions score zero"}
    references = json.loads((directory / "references.json").read_text(encoding="utf-8"))
    if set(references) != set(ids):
        raise ValueError("Reference IDs differ from manifest")
    per_case = []
    if manifest["dataset"] == "litsearch":
        if phase != "retrieve":
            raise ValueError("LitSearch must be scored from retrieve results")
        max_k = manifest["settings"]["retrieval_top_k"]
        if not cutoffs or any(k < 1 or k > max_k for k in cutoffs):
            raise ValueError(f"Cutoffs must be positive and <= recorded top-k ({max_k})")
        for key in ids:
            ranked = [str(c["paper_id"]).removeprefix("litsearch-")
                      for c in completed.get(key, {}).get("retrieved", [])]
            metrics = retrieval_scores(ranked, map(str, references[key]), cutoffs)
            per_case.append({"case_id": key, **metrics})
        report["quality_scores"] = {
            metric: sum(row[metric] for row in per_case) / len(ids)
            for metric in per_case[0] if metric != "case_id"}
        report["protocol"] = manifest["protocol"]
        report["full_corpus"] = (manifest["protocol"]["target_papers"] ==
                                 manifest["protocol"]["available_papers"])
        report["metric_note"] = "Macro query average, binary relevance, deduplicated paper ranks."
    elif manifest["dataset"] == "qasper":
        source = data_root / "qasper" / f"qasper-{manifest['protocol']['split']}-v0.3.json"
        if sha256(source) != manifest["files"][0]["sha256"]:
            raise ValueError("QASPER source hash differs from indexed data")
        script = data_root / "qasper/qasper_evaluator.py"
        spec = importlib.util.spec_from_file_location("qasper_official", script)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        data = json.loads(source.read_text(encoding="utf-8"))
        gold = official.get_answers_and_evidence(data, False)
        text_gold = official.get_answers_and_evidence(data, True)
        gold = {key: gold[key] for key in ids}
        text_gold = {key: text_gold[key] for key in ids}
        paragraphs = {}
        for paper in data.values():
            source_paragraphs = [p for section in paper.get("full_text", [])
                                 for p in section.get("paragraphs", [])]
            for question in paper["qas"]:
                paragraphs[question["question_id"]] = source_paragraphs
        predicted, unmatched = {}, 0
        for key, row in completed.items():
            if phase == "answer":
                answer = row.get("answer")
                if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                    raise ValueError("Completed answer has no answer text")
                excerpts = [c["excerpt"] for c in answer.get("citations", [])]
                text = answer["answer"]  # No gold-driven normalization or abstention rewriting.
            else:
                excerpts = [c["text"] for c in row.get("retrieved", [])]
                text = ""
            evidence, missing = project_paragraphs(excerpts, paragraphs[key])
            unmatched += missing
            predicted[key] = {"answer": text, "evidence": evidence}
            per_case.append({"question_id": key, "predicted_answer": text,
                             "predicted_evidence": evidence, "unmapped_excerpts": missing})
        scores = official.evaluate(gold, predicted)
        text_scores = official.evaluate(text_gold, predicted)
        if phase != "answer":
            scores = {"Retrieved paragraph Evidence F1": scores["Evidence F1"],
                      "Missing predictions": scores["Missing predictions"]}
        report["quality_scores"] = scores
        report["text_evidence_only_f1"] = text_scores["Evidence F1"]
        report["unmapped_excerpts"] = unmatched
        report["evaluator_sha256"] = sha256(script)
        report["metric_note"] = (
            "Official QASPER scoring algorithm on selected questions. Evidence uses source-only "
            "containment projection to original full_text paragraphs; captions/tables are not "
            "indexed. Answer phase scores cited excerpts, retrieve phase scores retrieved chunks. "
            "Projection is conservative, not semantic support or human correctness evaluation.")
    else:
        report["quality_status"] = "requires human or model rubric judgment"
        report["metric_note"] = (
            "ScholarQA-CS rubric scoring is not approximated by lexical overlap. "
            "No judge was called; official quality scores remain null.")
    report["prediction_hashes"] = {
        p.name: sha256(p) for p in sorted((directory / phase).glob("*.json"))}
    write_json(directory / f"scores_{phase}.json", report)
    write_json(directory / f"scored_cases_{phase}.json", per_case)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["retrieve", "answer"], default="answer")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir, phase=args.phase, data_root=args.data_root,
                               cutoffs=args.cutoffs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
