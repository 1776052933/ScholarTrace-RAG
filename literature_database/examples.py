"""LiteratureService 资源准备流程示例。

运行方式：
    python -m literature_database.examples
或：
    python literature_database/examples.py
"""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from literature_database.database import DatabaseManager
    from literature_database.download_tracker import DownloadTracker
    from literature_database.models import DownloadTaskModel
    from literature_database.paper_manager import PaperManager
    from literature_database.repository import FileRepository, PaperRepository
    from literature_database.schema import PaperMetadata
    from literature_database.service import LiteratureService
else:
    from .database import DatabaseManager
    from .download_tracker import DownloadTracker
    from .models import DownloadTaskModel
    from .paper_manager import PaperManager
    from .repository import FileRepository, PaperRepository
    from .schema import PaperMetadata
    from .service import LiteratureService

from sqlalchemy import select


RANKING_RESULT = {
    "papers": [
        {
            "paper_id": "paper-a",
            "title": "Paper A: Transformer Medical Segmentation",
            "authors": ["Author A"],
            "abstract": "Transformer medical image segmentation experiments.",
            "year": 2025,
            "venue": "MICCAI",
            "citation_count": 80,
            "rank_score": 0.95,
        },
        {
            "paper_id": "paper-b",
            "title": "Paper B: Medical Segmentation Review",
            "authors": ["Author B"],
            "abstract": "A review of medical image segmentation methods.",
            "year": 2024,
            "venue": "Medical Image Analysis",
            "citation_count": 30,
            "rank_score": 0.80,
        },
        {
            "paper_id": "paper-c",
            "title": "Paper C: Core Transformer Segmentation Study",
            "authors": ["Author C"],
            "abstract": "A highly relevant Transformer segmentation study.",
            "year": 2025,
            "venue": "CVPR",
            "citation_count": 50,
            "rank_score": 0.98,
        },
    ]
}


def run_example() -> None:
    with TemporaryDirectory(prefix="auto_paper_literature_") as temporary_directory:
        root = Path(temporary_directory)
        pdf_path = root / "paper-a.pdf"
        text_path = root / "paper-a.txt"
        pdf_path.write_bytes(b"%PDF-1.4\n% example PDF")
        text_path.write_text("Paper A parsed full text for RAG.", encoding="utf-8")

        database_path = (root / "literature.db").as_posix()
        database = DatabaseManager(f"sqlite:///{database_path}")
        service = LiteratureService(database)

        # Paper A 已有 PDF、解析文本和 embedding。
        # Paper C 已发生一次下载失败，用于触发 CORE 人工上传队列。
        with database.get_session() as session:
            papers = PaperRepository(session)
            files = FileRepository(session)
            paper_a = papers.add_paper(_metadata(RANKING_RESULT["papers"][0]))
            files.add_file("paper-a", str(pdf_path), status="AVAILABLE", file_type="pdf")
            files.add_file("paper-a", str(text_path), status="PARSED", file_type="text")
            paper_a.embedding_status = "COMPLETED"

            PaperManager(session).register_paper(RANKING_RESULT["papers"][2])
            paper_c_task = session.scalar(
                select(DownloadTaskModel)
                .where(DownloadTaskModel.paper_id == "paper-c")
                .order_by(DownloadTaskModel.id.desc())
            )
            assert paper_c_task is not None
            DownloadTracker(session).fail_task(paper_c_task.id, "自动下载未找到可访问全文")
            session.commit()

        rag_documents = service.prepare_documents(RANKING_RESULT)

        with database.get_session() as session:
            manager = PaperManager(session)
            states = {
                paper_id: manager.prepare_for_rag(paper_id)
                for paper_id in ("paper-a", "paper-b", "paper-c")
            }
            session.commit()

        display_status = {
            "paper-a": states["paper-a"]["status"],
            "paper-b": (
                "DOWNLOAD_REQUIRED"
                if states["paper-b"]["need_download"]
                else states["paper-b"]["status"]
            ),
            "paper-c": states["paper-c"]["status"],
        }
        assert display_status == {
            "paper-a": "FULL_AVAILABLE",
            "paper-b": "DOWNLOAD_REQUIRED",
            "paper-c": "USER_UPLOAD_REQUIRED",
        }
        assert [item.paper.paper_id for item in rag_documents] == ["paper-a"]

        output = {
            "papers": [
                {
                    "paper_id": item["paper_id"],
                    "title": item["title"],
                    "rank_score": item["rank_score"],
                    "status": display_status[item["paper_id"]],
                }
                for item in RANKING_RESULT["papers"]
            ],
            "rag_documents": [item.model_dump(mode="json") for item in rag_documents],
            "missing_core_papers": [
                item.model_dump(mode="json")
                for item in service.get_missing_core_papers()
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        database.dispose()


def _metadata(value: dict) -> PaperMetadata:
    return PaperMetadata(**value)


if __name__ == "__main__":
    run_example()
