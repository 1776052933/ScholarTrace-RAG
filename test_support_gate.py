from rag_reader.support_gate import gate, normalize_refusal, validate_verdict
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
