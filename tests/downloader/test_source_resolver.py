"""多下载源解析与回退测试。"""

from unittest.mock import MagicMock

from PyPDF2 import PdfWriter

from paper_downloader.src.downloaders.download_manager import DownloadManager, DownloadStatus
from paper_downloader.src.downloaders.source_resolver import DownloadSourceResolver
from paper_downloader.src.models.paper import Paper


def test_arxiv_candidate_has_highest_priority():
    resolver = DownloadSourceResolver({
        "download": {"source_engines": ["arxiv", "direct"]},
    })
    paper = Paper(
        title="Attention Is All You Need",
        arxiv_id="1706.03762v7",
        pdf_url="https://mirror.example/paper.pdf",
    )

    candidates = resolver.resolve(paper)

    assert candidates[0].source == "arxiv"
    assert candidates[0].url == "https://arxiv.org/pdf/1706.03762v7.pdf"
    assert candidates[1].source == "direct"


def test_openalex_resolves_oa_pdf_without_direct_url():
    resolver = DownloadSourceResolver({
        "download": {"source_engines": ["openalex"]},
    })
    resolver._get_json = MagicMock(return_value={
        "best_oa_location": {"pdf_url": "https://repository.example/paper.pdf"},
    })

    candidates = resolver.resolve(Paper(title="Example", doi="10.1234/example"))

    assert [(c.source, c.url) for c in candidates] == [
        ("openalex", "https://repository.example/paper.pdf"),
    ]


def test_openalex_resolver_passes_configured_api_key():
    resolver = DownloadSourceResolver({
        "download": {"source_engines": ["openalex"]},
        "openalex_api_key": "test-key",
        "contact_email": "researcher@example.test",
    })
    resolver._get_json = MagicMock(return_value={})

    resolver.resolve(Paper(title="Example", doi="10.1234/example"))

    _, kwargs = resolver._get_json.call_args
    assert kwargs["params"] == {
        "api_key": "test-key",
        "mailto": "researcher@example.test",
    }


def test_metadata_endpoint_does_not_block_open_access_resolution():
    resolver = DownloadSourceResolver({
        "download": {"source_engines": ["direct", "openalex"]},
    })
    resolver._get_json = MagicMock(return_value={
        "best_oa_location": {"pdf_url": "https://repository.example/paper.pdf"},
    })
    metadata_url = (
        "https://api.elsevier.com/content/article/PII:S1234?httpAccept=text/xml"
    )

    candidates = resolver.resolve(Paper(
        title="Example",
        doi="10.1234/example",
        pdf_url=metadata_url,
    ))

    assert [(candidate.source, candidate.url) for candidate in candidates] == [
        ("openalex", "https://repository.example/paper.pdf"),
        ("direct", metadata_url),
    ]


def test_download_manager_falls_back_to_second_candidate(tmp_path):
    manager = DownloadManager(
        config={"retry": {"max_attempts": 1, "backoff_factor": 0}},
        max_workers=1,
        history_file=str(tmp_path / "history.json"),
    )
    fake_downloader = MagicMock()
    valid_pdf = tmp_path / "valid.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with valid_pdf.open("wb") as handle:
        writer.write(handle)
    fake_downloader.download.side_effect = [None, valid_pdf]
    manager._downloader = fake_downloader
    manager.add_task(
        "https://first.example/paper.pdf",
        title="Fallback Paper",
        save_dir=str(tmp_path),
        candidate_urls=["https://second.example/paper.pdf"],
    )

    tasks = manager.run_all()

    assert tasks[0].status == DownloadStatus.COMPLETED
    assert tasks[0].url == "https://second.example/paper.pdf"
    assert fake_downloader.download.call_count == 2
