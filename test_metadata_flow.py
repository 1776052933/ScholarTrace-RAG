"""Regression test for download identifiers across search and ranking."""

from ranking_agent.pipeline import _paper as ranking_paper
from search_agent.schema import PaperMetadata
from search_agent.searcher import deduplicate_papers, to_internal_paper


def test_download_metadata_survives_deduplication_and_ranking():
    scholar = to_internal_paper(PaperMetadata(
        title="Swin UNETR",
        authors=["Ali Hatamizadeh"],
        url="https://example.org/landing-page",
        source="google_scholar",
        sources=["google_scholar"],
    ))
    arxiv = to_internal_paper(PaperMetadata(
        title="Swin UNETR",
        authors=["Ali Hatamizadeh"],
        url="http://arxiv.org/abs/2201.01266v1",
        source="arxiv",
        arxiv_id="2201.01266v1",
        pdf_url="https://arxiv.org/pdf/2201.01266v1.pdf",
        sources=["arxiv"],
    ))

    merged = deduplicate_papers([scholar, arxiv])
    ranked = ranking_paper(merged[0])

    assert len(merged) == 1
    assert ranked.arxiv_id == "2201.01266v1"
    assert ranked.pdf_url == "https://arxiv.org/pdf/2201.01266v1.pdf"
    assert set(ranked.sources) == {"google_scholar", "arxiv"}
