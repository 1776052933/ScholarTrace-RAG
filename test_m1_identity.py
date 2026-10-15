"""Stable paper identifiers across search, ranking, and registration."""

from literature_database.paper_manager import _metadata
from ranking_agent.pipeline import _paper as ranking_paper
from search_agent.schema import PaperMetadata
from search_agent.searcher import deduplicate_papers, to_internal_paper


def test_search_ranking_and_database_keep_one_arxiv_identity():
    paper = PaperMetadata(
        title="A stable identity example",
        arxiv_id="2401.12345v2",
        url="https://arxiv.org/abs/2401.12345v2",
        pdf_url="https://arxiv.org/pdf/2401.12345v2.pdf",
        source="arxiv",
    )
    discovered = to_internal_paper(paper)
    ranked = ranking_paper(discovered.model_dump())
    registered = _metadata(ranked, rank_score=0.8)
    assert discovered.paper_id == ranked.paper_id == registered.paper_id == "arxiv:2401.12345"


def test_dedup_prefers_doi_when_later_source_adds_it():
    arxiv = to_internal_paper(PaperMetadata(
        title="Shared paper", arxiv_id="2501.12345v1", source="arxiv",
    ))
    doi = to_internal_paper(PaperMetadata(
        title="Shared paper", doi="https://doi.org/10.1234/EXAMPLE", source="crossref",
    ))
    merged = deduplicate_papers([arxiv, doi])
    assert len(merged) == 1
    assert merged[0].paper_id == "doi:10.1234/example"
