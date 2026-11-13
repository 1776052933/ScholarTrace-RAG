"""Behavioral checks for M0 dataset adaptation and diagnostic metrics."""

from evaluation.m0_metrics import answer_f1, evidence_recall, reference_answer, token_f1
from evaluation.run_m0_qasper import paper_text, select_questions


def test_qasper_input_contains_only_paper_text():
    paper = {
        "title": "A paper", "abstract": "Study of retrieval",
        "full_text": [{"section_name": "Results", "paragraphs": ["The method improves recall."]}],
        "qas": [{"question": "What improves?", "answers": [{"answer": {"free_form_answer": "SECRET GOLD ANSWER"}}]}],
    }
    text = paper_text(paper)
    assert "The method improves recall." in text
    assert "SECRET GOLD ANSWER" not in text
    assert "What improves?" not in text


def test_evidence_recall_uses_gold_coverage_and_best_annotator():
    annotations = [
        {"answer": {"highlighted_evidence": ["an unrelated evidence paragraph"]}},
        {"answer": {"highlighted_evidence": ["the method improves retrieval recall"]}},
    ]
    chunks = ["Long surrounding text; the method improves retrieval recall across all tasks with many extra words."]
    assert evidence_recall(annotations, chunks) == 1.0
    assert evidence_recall(annotations, ["no matching text here"]) == 0.0
    assert evidence_recall([{"answer": {"unanswerable": True, "evidence": []}}], chunks) is None


def test_answer_references_and_deterministic_selection():
    annotations = [
        {"answer": {"yes_no": True}},
        {"answer": {"extractive_spans": ["BGE", "M3"]}},
    ]
    assert reference_answer(annotations[0]) == "yes"
    assert answer_f1("BGE M3", annotations) == 1.0
    assert token_f1("some answer", "other answer") > 0.0
    data = {
        "paper_b": {"qas": [{"question_id": "q1", "question": "one"}]},
        "paper_a": {"qas": [{"question_id": "q2", "question": "two"}]},
    }
    assert select_questions(data, 1, 42) == select_questions(data, 1, 42)
