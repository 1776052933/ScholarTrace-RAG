from rag_reader.support_gate import gate, is_refusal, normalize_refusal, strict_gate, validate_verdict
from evaluation.qasper_evidence_budget import constituent_texts, merge_adjacent


def test_verifier_must_quote_the_cited_chunk_exactly():
    chunks = [{"chunk_id": "p-chunk-00001", "paper_id": "p", "text": "The model uses three layers."}]
    assert gate("Three layers", ["p-chunk-00001"], chunks)["action"] == "verify"
    valid = {"decision": "supported", "quotes": [
        {"chunk_id": "p-chunk-00001", "quote": "uses three layers"}]}
    assert gate("Three layers", ["p-chunk-00001"], chunks, valid)["action"] == "answer"
    invalid = {"decision": "supported", "quotes": [
        {"chunk_id": "p-chunk-00001", "quote": "uses four layers"}]}
    assert gate("Four layers", ["p-chunk-00001"], chunks, invalid)["action"] == "retry"
    assert validate_verdict({"decision": "supported", "quotes": []},
                            {"p-chunk-00001": chunks[0]["text"]})["supported"] is False


def test_uncited_missing_information_sentence_is_normalized_but_no_is_not():
    chunks = [{"chunk_id": "x", "text": "No details are provided."}]
    answer = "The algorithm is not explicitly stated in the provided chunks."
    assert normalize_refusal(answer, [])
    assert gate(answer, [], chunks)["action"] == "abstain"
    assert gate("No", [], chunks)["action"] == "retry"


def test_strict_gate_requires_every_paper_quote_and_answer_number():
    chunks = [
        {"chunk_id": "a", "paper_id": "paper-a", "text": "Paper A reports a score of 28.4."},
        {"chunk_id": "b", "paper_id": "paper-b", "text": "Paper B reports a score of 41.8."},
    ]
    answer = "Paper A scores 28.4 and Paper B scores 41.8."
    quotes = {"decision": "supported", "quotes": [
        {"chunk_id": "a", "quote": chunks[0]["text"]},
        {"chunk_id": "b", "quote": chunks[1]["text"]},
    ]}
    assert strict_gate(answer, ["a", "b"], chunks, paper_ids=["paper-a", "paper-b"],
                       verdict=quotes)["action"] == "answer"
    assert strict_gate(answer, ["a"], chunks, paper_ids=["paper-a", "paper-b"])["reason"] == "missing_paper_citation"
    assert strict_gate(answer, ["a", "b"], chunks, paper_ids=["paper-a", "paper-b"],
                       verdict={"decision": "supported", "quotes": quotes["quotes"][:1]})["reason"] == "missing_source_quote"
    wrong = answer.replace("41.8", "42.8")
    assert strict_gate(wrong, ["a", "b"], chunks, paper_ids=["paper-a", "paper-b"],
                       verdict=quotes)["reason"] == "unquoted_number"
    assert is_refusal("根据提供的论文片段无法确定。")
    assert not is_refusal("No")


def test_adjacent_merge_preserves_provenance_and_page_boundaries():
    chunks = [
        {"chunk_id": "p-chunk-00002", "paper_id": "p", "page": 1, "section": "S", "text": "B"},
        {"chunk_id": "p-chunk-00001", "paper_id": "p", "page": 1, "section": "S", "text": "A"},
        {"chunk_id": "p-chunk-00003", "paper_id": "p", "page": 2, "section": "S", "text": "C"},
    ]
    merged = merge_adjacent(chunks)
    assert len(merged) == 2
    assert merged[0]["source_chunk_ids"] == ["p-chunk-00001", "p-chunk-00002"]
    assert merged[0]["text"] == "A\nB"
    assert merged[1]["source_chunk_ids"] == ["p-chunk-00003"]
    assert constituent_texts(merged, chunks) == ["A", "B", "C"]
