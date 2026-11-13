from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.litsearch import load_queries
from evaluation.metrics import aggregate_results, evaluate_query
from evaluation.pasa_metrics import evaluate_pasa_query, normalize_pasa_title
from evaluation.pasa_realscholar import load_real_scholar_queries
from evaluation.semantic_scholar import SemanticScholarResolver, _paper_identifier
from evaluation.scholargym import (
    benchmark_paper_arxiv_id,
    evaluate_scholargym_query,
    load_scholargym_queries,
    normalize_arxiv_id,
    paper_arxiv_id,
)


def test_metrics_count_unresolved_predictions_as_incorrect():
    metrics = evaluate_query([10, 20], [10, None, 99, 20], [2, 4])

    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["f1@2"] == 0.5
    assert metrics["precision@4"] == 0.5
    assert metrics["recall@4"] == 1.0
    assert round(metrics["f1@4"], 6) == round(2 / 3, 6)
    assert metrics["mrr@4"] == 1.0


def test_precision_uses_cutoff_when_fewer_results_are_returned():
    metrics = evaluate_query([10], [10, None], [5])

    assert metrics["precision@5"] == 0.2
    assert metrics["recall@5"] == 1.0


def test_aggregate_results_uses_macro_average():
    records = [
        {
            "status": "completed",
            "metrics": {"f1@5": 1.0},
            "workflow_seconds": 2.0,
            "prediction_count": 5,
            "resolved_prediction_count": 4,
        },
        {
            "status": "completed",
            "metrics": {"f1@5": 0.0},
            "workflow_seconds": 4.0,
            "prediction_count": 5,
            "resolved_prediction_count": 3,
        },
        {"status": "failed"},
    ]

    summary = aggregate_results(records)

    assert summary["query_count"] == 3
    assert summary["successful_query_count"] == 2
    assert summary["macro_metrics"]["f1@5"] == 0.5
    assert summary["average_workflow_seconds"] == 3.0
    assert summary["prediction_resolution_rate"] == 0.7


def test_litsearch_loader_uses_official_columns(tmp_path):
    path = tmp_path / "queries.parquet"
    table = pa.table({
        "query_set": ["inline_acl", "manual_iclr"],
        "query": ["first query", "second query"],
        "specificity": [0, 1],
        "quality": [1, 2],
        "corpusids": [[11], [22, 33]],
    })
    pq.write_table(table, path)

    rows = load_queries(path, query_set="manual_iclr", minimum_quality=2)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "litsearch_0001"
    assert rows[0]["gold_corpus_ids"] == [22, 33]


def test_identifier_prefers_doi_and_extracts_arxiv():
    assert _paper_identifier({"doi": "https://doi.org/10.1/example"}) == "DOI:10.1/example"
    assert _paper_identifier({"url": "https://arxiv.org/abs/2407.18940v2"}) == "ARXIV:2407.18940"
    assert _paper_identifier({"arxiv_id": "2305.12330v1"}) == "ARXIV:2305.12330"


def test_identifier_extracts_doi_and_acl_from_urls():
    assert _paper_identifier({
        "pdf_url": "https://dl.acm.org/doi/pdf/10.1145/3699518",
    }) == "DOI:10.1145/3699518"
    assert _paper_identifier({
        "pdf_url": "https://link.springer.com/content/pdf/10.1007/s10462-025-11423-3.pdf",
    }) == "DOI:10.1007/s10462-025-11423-3"
    assert _paper_identifier({
        "url": "https://aclanthology.org/2021.emnlp-main.30/",
    }) == "ACL:2021.emnlp-main.30"


class FakeResolver(SemanticScholarResolver):
    def __init__(self, cache_path: Path):
        super().__init__(cache_path, minimum_interval=0)
        self.requests = []

    def _request_json(self, method, path, *, params, json_body=None):
        self.requests.append((method, path))
        if path == "/paper/batch":
            return [{
                "paperId": "hash-1",
                "corpusId": 101,
                "title": "Exact Paper",
                "year": 2024,
            }]
        return None


def test_resolver_uses_identifier_batch_and_cache(tmp_path):
    resolver = FakeResolver(tmp_path / "cache.json")
    papers = [{"title": "Exact Paper", "year": 2024, "doi": "10.1/example"}]

    first = resolver.resolve_many(papers)
    second = resolver.resolve_many(papers)

    assert first[0]["corpus_id"] == 101
    assert second[0]["corpus_id"] == 101
    assert resolver.requests == [("POST", "/paper/batch")]


def test_pasa_title_matching_follows_official_keep_letters():
    assert normalize_pasa_title("BERT: Pre-training 2.0!") == "bertpretraining"


def test_pasa_metrics_distinguish_candidate_and_ranked_recall():
    result = evaluate_pasa_query(
        ["Gold Paper", "Second Gold"],
        ["Gold Paper", "Second Gold", "Noise"],
        ["Noise", "Gold Paper", "Second Gold"],
        [1, 3],
        selected_titles=["Noise", "Gold Paper"],
    )

    assert result["metrics"]["crawler_recall"] == 1.0
    assert result["metrics"]["recall@1"] == 0.0
    assert result["metrics"]["recall@3"] == 1.0
    assert result["metrics"]["selector_precision"] == 0.5
    assert result["metrics"]["selector_recall"] == 0.5
    assert result["best_gold_rank"] == 2


def test_load_real_scholar_queries(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"question":"Find papers","answer":["Paper A","Paper B"],'
        '"source_meta":{"published_time":"20250108"}}\n',
        encoding="utf-8",
    )

    rows = load_real_scholar_queries(path)

    assert rows[0]["query_id"] == "pasa_real_000"
    assert rows[0]["gold_titles"] == ["Paper A", "Paper B"]
    assert rows[0]["pasa_search_cutoff"] == "20250101"


def test_scholargym_loader_selects_pasa_real(tmp_path):
    path = tmp_path / "scholargym_bench.jsonl"
    path.write_text(
        '{"query_id":"pasa_real_001","query":"Find papers",'
        '"gt_arxiv_ids":["2401.12345v2"],"source":"pasa_real","split":"test"}\n'
        '{"query_id":"litsearch_001","query":"Ignore me",'
        '"gt_arxiv_ids":["2301.00001"],"source":"litsearch","split":"test"}\n',
        encoding="utf-8",
    )

    rows = load_scholargym_queries(path, source="PaSa-RealScholar")

    assert len(rows) == 1
    assert rows[0]["query_id"] == "pasa_real_001"
    assert rows[0]["gold_arxiv_ids"] == ["2401.12345"]


def test_scholargym_loader_supports_released_schema(tmp_path):
    path = tmp_path / "scholargym_bench.jsonl"
    path.write_text(
        '{"query":"Find released papers","cited_paper":['
        '{"arxiv_id":"2401.11111"},{"arxiv_id":"2402.22222"}],'
        '"gt_label":[1,0],"date":"2024-09","source":"PASA_RealScholar",'
        '"qid":"RealScholarQuery_0","valid":true}\n'
        '{broken json}\n',
        encoding="utf-8",
    )

    rows = load_scholargym_queries(path)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "RealScholarQuery_0"
    assert rows[0]["gold_arxiv_ids"] == ["2401.11111"]
    assert rows[0]["gold_papers"][0]["title"] == ""
    assert rows[0]["date_constraint"] == "2024-09"


def test_scholargym_arxiv_extraction_and_metrics():
    assert normalize_arxiv_id("https://arxiv.org/pdf/2401.12345v3.pdf") == "2401.12345"
    assert paper_arxiv_id({"url": "https://arxiv.org/abs/2305.00123v2"}) == "2305.00123"

    result = evaluate_scholargym_query(
        ["2401.12345", "2305.00123"],
        ["2401.12345", None],
        [None, "2401.12345", "9999.99999"],
        [1, 3],
        selected_ids=[None, "2401.12345"],
    )

    assert result["metrics"]["crawler_recall"] == 0.5
    assert result["metrics"]["recall@1"] == 0.0
    assert result["metrics"]["recall@3"] == 0.5
    assert result["best_gold_rank"] == 2


def test_scholargym_uses_exact_gold_title_when_source_omits_arxiv_id():
    gold = [{"arxiv_id": "2402.09668", "title": "How to Train Data-Efficient LLMs"}]

    matched = benchmark_paper_arxiv_id(
        {"title": "How to train data-efficient llms", "url": "https://example.org/paper"},
        gold,
    )

    assert matched == "2402.09668"
