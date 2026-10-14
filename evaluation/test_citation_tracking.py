from search_agent.citation_tracker import CitationTracker
from search_agent.schema import PaperMetadata


class FakeEmbeddingModel:
    def encode_query(self, query):
        return [1.0, 0.0]

    def encode_papers(self, papers):
        return [
            [1.0, 0.0] if "relevant" in title.casefold() else [0.5, 0.5]
            for title, _ in papers
        ]


class FakeOpenAlexClient:
    api_key = "configured-for-test"

    def __init__(self):
        self.api_call_count = 0

    def resolve(self, paper):
        self.api_call_count += 1
        if paper.openalex_id:
            suffix = int(str(paper.openalex_id).lstrip("Ww"))
        else:
            suffix = 100 + int(paper.doi.rsplit("/", 1)[-1])
        return {
            "id": f"https://openalex.org/W{suffix}",
            "referenced_works": [],
        }

    def neighbors(self, work, *, publication_date_to=None):
        self.api_call_count += 1
        if str(work.get("id") or "").endswith("W900"):
            return {
                "references": [_work("W910", "Relevant second hop paper", "2022-05-01")],
                "citations": [],
            }
        if str(work.get("id") or "").endswith("W910"):
            return {"references": [], "citations": []}
        return {
            "references": [
                _work("W900", "Relevant historical paper", "2023-04-01"),
                _work("W901", "Seed paper 0", "2023-01-01"),
                _work("W902", "Relevant future paper", "2025-01-01"),
            ],
            "citations": [
                _work("W900", "Relevant historical paper", "2023-04-01"),
            ],
        }


def _work(openalex_id, title, publication_date):
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "title": title,
        "publication_year": int(publication_date[:4]),
        "publication_date": publication_date,
        "cited_by_count": 3,
        "primary_location": {},
        "best_oa_location": {},
        "authorships": [],
        "ids": {},
        "referenced_works": [],
    }


def _initial_papers():
    return [
        PaperMetadata(
            title=f"Seed paper {index}",
            abstract="relevant seed",
            year=2023,
            doi=f"10.test/{index}",
            source="arxiv",
        )
        for index in range(5)
    ]


def test_two_hop_tracking_filters_deduplicates_links_and_reuses_cache(tmp_path):
    client = FakeOpenAlexClient()
    tracker = CitationTracker(
        embedding_model=FakeEmbeddingModel(),
        api_client=client,
        cache_path=tmp_path / "citation.sqlite3",
    )

    first = tracker.expand(
        "relevant research",
        _initial_papers(),
        publication_date_to="2024-06-30",
    )
    calls_after_first = client.api_call_count
    second = tracker.expand(
        "relevant research",
        _initial_papers(),
        publication_date_to="2024-06-30",
    )

    assert len(first.papers) == 7
    assert [paper.title for paper in first.papers].count("Relevant historical paper") == 1
    assert all(paper.title != "Relevant future paper" for paper in first.papers)
    assert first.summary["retained_count"] == 2
    assert first.summary["first_layer_retained_count"] == 1
    assert first.summary["second_layer_retained_count"] == 1
    assert first.summary["resolved_seed_count"] == 6
    second_hop = next(paper for paper in first.papers if paper.title == "Relevant second hop paper")
    assert second_hop.citation_depth == 2
    assert second_hop.citation_parent_ids == ["openalex:w900"]
    assert second.summary["cache_hit_count"] == 12
    assert second.summary["api_call_count"] == 0
    assert client.api_call_count == calls_after_first


def test_tracking_skips_cleanly_without_openalex_key(tmp_path):
    class NoKeyClient(FakeOpenAlexClient):
        api_key = ""

    tracker = CitationTracker(
        embedding_model=FakeEmbeddingModel(),
        api_client=NoKeyClient(),
        cache_path=tmp_path / "citation.sqlite3",
    )
    result = tracker.expand("query", _initial_papers())

    assert len(result.papers) == 5
    assert result.summary["api_call_count"] == 0
    assert "OPENALEX_API_KEY" in result.summary["skipped_reason"]


def test_failed_relation_requests_are_not_cached(tmp_path):
    class FailingRelationClient(FakeOpenAlexClient):
        def neighbors(self, work, *, publication_date_to=None):
            self.api_call_count += 1
            return None

    client = FailingRelationClient()
    tracker = CitationTracker(
        embedding_model=FakeEmbeddingModel(),
        api_client=client,
        cache_path=tmp_path / "citation.sqlite3",
    )

    tracker.expand("query", _initial_papers())
    calls_after_first = client.api_call_count
    second = tracker.expand("query", _initial_papers())

    assert calls_after_first == 10
    assert client.api_call_count == 15
    assert second.summary["cache_hit_count"] == 5


def test_identifier_completion_and_seed_backfill(tmp_path):
    papers = [
        PaperMetadata(
            title=f"Unidentified high score {index}",
            abstract="relevant",
            citation_count=100 - index,
            source="scholar",
        )
        for index in range(2)
    ]
    papers.extend(
        PaperMetadata(
            title=f"Identifiable seed {index}",
            abstract="relevant",
            citation_count=10 - index,
            url=f"https://doi.org/10.test/{index}",
            source="scholar",
        )
        for index in range(5)
    )
    client = FakeOpenAlexClient()
    tracker = CitationTracker(
        embedding_model=FakeEmbeddingModel(),
        api_client=client,
        cache_path=tmp_path / "citation.sqlite3",
    )

    result = tracker.expand("relevant", papers)

    assert result.summary["seed_count"] == 5
    assert result.summary["identifiable_seed_count"] == 5
    assert all(paper.doi for paper in result.papers if paper.title.startswith("Identifiable"))
