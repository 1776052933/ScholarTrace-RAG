"""Checks for report exports used by the current HTTP-backed interface."""

from Auto_Paper_UI.reporting import papers_csv_bytes, ranked_papers, selected_papers


def test_csv_uses_all_ranked_papers_not_only_top_papers():
    result = {
        "stages": {
            "search_agent": {"result": {"papers": [
                {"title": "First", "abstract": "A"},
                {"title": "Second", "abstract": "B"},
                {"title": "Search duplicate", "abstract": "C"},
            ]}},
            "ranking_agent": {"result": {"papers": [
                {"title": "First", "abstract": "A"},
                {"title": "Second", "abstract": "B"},
            ]}},
            "literature_database": {"selected_paper_count": 1},
        }
    }

    assert len(ranked_papers(result)) == 2
    assert len(selected_papers(result)) == 1
    csv_text = papers_csv_bytes(ranked_papers(result)).decode("utf-8-sig")
    assert "First" in csv_text
    assert "Second" in csv_text
    assert "Search duplicate" not in csv_text
    assert "摘要" in csv_text


def test_selected_papers_use_downloaded_ids_and_keep_original_rank():
    result = {
        "stages": {
            "ranking_agent": {"result": {"papers": [
                {"paper_id": "paper-1", "title": "First"},
                {"paper_id": "paper-2", "title": "Second"},
                {"paper_id": "paper-3", "title": "Third"},
                {"paper_id": "paper-4", "title": "Fourth"},
            ]}},
            "literature_database": {
                "selected_paper_count": 2,
                "selected_paper_ids": ["paper-3", "paper-4"],
            },
        }
    }

    papers = selected_papers(result)

    assert [paper["paper_id"] for paper in papers] == ["paper-3", "paper-4"]
    assert [paper["original_rank"] for paper in papers] == [3, 4]
