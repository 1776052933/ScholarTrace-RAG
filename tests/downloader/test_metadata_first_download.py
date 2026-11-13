"""Metadata-first download behavior and title-search fallback."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from paper_downloader.src.exceptions import DownloadError
from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.interface import _paper_from_input, download_pdf
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.utils.pdf_identity import pdf_matches_title, title_similarity


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    return path


def test_paper_input_extracts_arxiv_id_and_direct_pdf():
    arxiv = _paper_from_input({
        "paper_id": "http://arxiv.org/abs/2201.01266v1",
        "title": "Swin UNETR",
    })
    direct = _paper_from_input({
        "paper_id": "https://example.org/paper.pdf",
        "title": "Direct Paper",
    })

    assert arxiv.arxiv_id == "2201.01266v1"
    assert arxiv.url == "http://arxiv.org/abs/2201.01266v1"
    assert direct.pdf_url == "https://example.org/paper.pdf"


def test_paper_input_recognizes_ojs_article_download_url():
    url = "https://ojs.aaai.org/index.php/AAAI/article/download/29120/30118"
    paper = _paper_from_input({
        "paper_id": url,
        "title": "Double-layer hybrid-label identification feature selection",
    })

    assert paper.pdf_url == url


def test_title_matching_rejects_unrelated_first_search_result():
    requested = (
        "Double-layer hybrid-label identification feature selection for "
        "multi-view multi-label learning"
    )
    unrelated = Paper(
        title=(
            "Redundancy-optimized Multi-head Attention Networks for "
            "Multi-View Multi-Label Feature Selection"
        )
    )

    best, score = PaperDownloader._find_best_match(
        requested,
        [unrelated],
        min_similarity=0.60,
        strict=True,
    )

    assert best is None
    assert score < 0.60


def test_confusing_shared_suffix_titles_are_not_the_same_paper():
    requested = (
        "Feature relevance and redundancy coefficients for multi-view "
        "multi-label feature selection"
    )
    wrong = (
        "Redundancy-optimized Multi-head Attention Networks for Multi-View "
        "Multi-Label Feature Selection"
    )

    best, score = PaperDownloader._find_best_match(
        requested,
        [Paper(title=wrong)],
        min_similarity=0.82,
        strict=True,
    )

    assert title_similarity(requested, wrong) < 0.82
    assert best is None
    assert score < 0.82


def test_pdf_body_terms_cannot_replace_a_matching_title(tmp_path):
    import fitz

    requested = (
        "Feature relevance and redundancy coefficients for multi-view "
        "multi-label feature selection"
    )
    wrong = (
        "Redundancy-optimized Multi-head Attention Networks for Multi-View "
        "Multi-Label Feature Selection"
    )
    path = tmp_path / "wrong-paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), wrong, fontsize=12)
    page.insert_text((72, 120), "Abstract", fontsize=10)
    page.insert_text(
        (72, 145),
        "Feature relevance and redundancy coefficients are discussed in the body.",
        fontsize=10,
    )
    document.save(path)
    document.close()

    assert pdf_matches_title(path, requested) is False
    assert pdf_matches_title(path, wrong) is True


def test_exact_arxiv_metadata_skips_title_search(tmp_path):
    path = _pdf(tmp_path / "swin.pdf")
    downloaded = Paper(
        title="Swin UNETR",
        arxiv_id="2201.01266v1",
        source="arxiv",
        pdf_path=str(path),
        file_size=path.stat().st_size,
    )
    downloader = MagicMock()
    downloader.download.return_value = [downloaded]

    with patch(
        "paper_downloader.src.interface._new_downloader",
        return_value=downloader,
    ):
        result = download_pdf({
            "paper_id": "http://arxiv.org/abs/2201.01266v1",
            "title": "Swin UNETR",
            "arxiv_id": "2201.01266v1",
        }, output_dir=str(tmp_path), engine="academic_oa")

    assert result["success"] is True
    downloader.search.assert_not_called()
    passed_paper = downloader.download.call_args.args[0]
    assert passed_paper.arxiv_id == "2201.01266v1"


def test_failed_exact_locator_falls_back_to_title_search(tmp_path):
    path = _pdf(tmp_path / "fallback.pdf")
    direct = MagicMock()
    direct.download.side_effect = DownloadError("direct failed")

    fallback_paper = Paper(
        title="Fallback Paper",
        source="arxiv",
        arxiv_id="2401.00001",
        pdf_path=str(path),
        file_size=path.stat().st_size,
    )
    fallback = MagicMock()
    fallback.search.return_value = [fallback_paper]
    fallback._find_best_match.return_value = (fallback_paper, 1.0)
    fallback.download.return_value = [fallback_paper]

    with patch(
        "paper_downloader.src.interface._new_downloader",
        side_effect=[direct, fallback],
    ):
        result = download_pdf({
            "title": "Fallback Paper",
            "pdf_url": "https://invalid.example/paper.pdf",
        }, output_dir=str(tmp_path), engine="arxiv")

    assert result["success"] is True
    fallback.search.assert_called_once_with(
        "Fallback Paper",
        max_results=3,
        engines=["arxiv"],
    )
