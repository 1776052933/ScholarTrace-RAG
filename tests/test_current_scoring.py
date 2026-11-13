"""Scoring math, denominators and source-only evidence projection."""
import json
from pathlib import Path

import pytest

from evaluation.score_current import project_paragraphs, retrieval_scores, score_run
from evaluation.m0_common import DATA, sha256


def test_rank_metrics_deduplicate_and_use_full_gold_denominator():
    score = retrieval_scores(["x", "x", "a", "b"], ["a", "b"], [1, 2, 3])
    assert score["Recall@1"] == 0
    assert score["Recall@2"] == .5
    assert score["Recall@3"] == 1
    assert score["MRR@3"] == .5
    assert 0 < score["nDCG@3"] < 1
    with pytest.raises(ValueError):
        retrieval_scores([], [], [1])


def test_projection_keeps_original_and_rejects_number_only_matches():
    original = "A sufficiently long paragraph about the value 92.7 and its exact context."
    evidence, unmatched = project_paragraphs(["92.7", original.replace(" ", "  ")], [original])
    assert evidence == [original]
    assert unmatched == 1
    assert project_paragraphs(["Unrelated evidence."], [original]) == ([], 1)


def lit_run(tmp_path):
    manifest = {"dataset": "litsearch", "case_ids": ["q1", "q2"],
                "settings": {"retrieval_top_k": 5},
                "protocol": {"target_papers": 10, "available_papers": 10}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "references.json").write_text(json.dumps({"q1": [1], "q2": [2]}))
    (tmp_path / "retrieve").mkdir()
    (tmp_path / "retrieve/a.json").write_text(json.dumps({
        "case_id": "q1", "status": "completed", "retrieved": [{"paper_id": "litsearch-1"}]}))
    return tmp_path


def test_missing_predictions_count_as_zero_and_never_load_runtime(tmp_path):
    report = score_run(lit_run(tmp_path), phase="retrieve", cutoffs=[1, 5])
    assert report["quality_scores"]["Recall@1"] == .5
    assert report["coverage"] == .5
    assert report["statuses"] == {"completed": 1, "pending": 1}
    assert report["external_model_calls"] == 0
    with pytest.raises(ValueError, match="top-k"):
        score_run(tmp_path, phase="retrieve", cutoffs=[20])
    (tmp_path / "retrieve/b.json").write_text((tmp_path / "retrieve/a.json").read_text())
    with pytest.raises(ValueError, match="duplicate"):
        score_run(tmp_path, phase="retrieve", cutoffs=[1])


def test_scholarqa_does_not_invent_rubric_score(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset": "scholarqa-cs", "case_ids": ["q"]}))
    (tmp_path / "references.json").write_text('{"q": {}}')
    report = score_run(tmp_path)
    assert report["quality_scores"] is None
    assert report["coverage"] == 0


def test_litsearch_retains_missing_abstracts_without_gold_imputation(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from evaluation.current_datasets import load_dataset
    (tmp_path / "litsearch/corpus_clean").mkdir(parents=True)
    (tmp_path / "litsearch/query").mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"corpusid": 1, "title": "", "abstract": ""},
        {"corpusid": 2, "title": "A source", "abstract": "Source text"},
    ]), tmp_path / "litsearch/corpus_clean/part.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"query": "SECRET_QUERY", "corpusids": [1]},
    ]), tmp_path / "litsearch/query/full-00000-of-00001.parquet")
    dataset = load_dataset("litsearch", tmp_path)
    sources = list(dataset.sources())
    assert len(sources) == 2
    assert sources[0].text == "Title: \nAbstract: "
    assert sources[1].text == "Title: A source\nAbstract: Source text"
    assert "SECRET_QUERY" not in str(sources)


@pytest.mark.skipif(not (DATA / "qasper/qasper_evaluator.py").exists(),
                    reason="official downloaded evaluator is not bundled in CI")
def test_qasper_official_script_exact_and_missing_predictions(tmp_path):
    import shutil
    root = tmp_path / "data"
    (root / "qasper").mkdir(parents=True)
    shutil.copyfile(DATA / "qasper/qasper_evaluator.py", root / "qasper/qasper_evaluator.py")
    paragraph = "This source paragraph states that the model has six layers."
    answer = {"unanswerable": False, "extractive_spans": ["six layers"],
              "free_form_answer": "", "yes_no": None, "evidence": [paragraph]}
    data = {"p": {"full_text": [{"paragraphs": [paragraph]}], "qas": [
        {"question_id": key, "answers": [{"answer": answer}]} for key in ["q1", "q2"]]}}
    source = root / "qasper/qasper-dev-v0.3.json"
    source.write_text(json.dumps(data))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset": "qasper", "case_ids": ["q1", "q2"], "protocol": {"split": "dev"},
        "files": [{"sha256": sha256(source)}]}))
    (tmp_path / "references.json").write_text('{"q1": [], "q2": []}')
    (tmp_path / "answer").mkdir()
    (tmp_path / "answer/a.json").write_text(json.dumps({
        "case_id": "q1", "status": "completed", "answer": {
            "answer": "six layers", "citations": [{"excerpt": paragraph}]}}))
    result = score_run(tmp_path, data_root=root)
    assert result["quality_scores"]["Answer F1"] == .5
    assert result["quality_scores"]["Evidence F1"] == .5
    assert result["quality_scores"]["Missing predictions"] == 1
