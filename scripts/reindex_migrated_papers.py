"""Rebuild Chroma for paper PDFs imported from the M1 SQLite database."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import select

from literature_database.database import DatabaseManager
from literature_database.models import PaperFileModel, PaperModel
from literature_database.schema import PaperAvailability, PaperMetadata, PaperResource
from rag_reader.pipeline import RAGReader
from rag_reader.vector_store import VectorStore


def rebuild(database_url: str, data_root: Path, paper_id: str | None = None) -> dict[str, int]:
    database = DatabaseManager(database_url)
    reader = RAGReader(vector_store=VectorStore(persist_directory=str(data_root / "chroma")))
    success = failed = 0
    try:
        with database.get_session() as session:
            statement = select(PaperModel)
            if paper_id:
                statement = statement.where(PaperModel.paper_id == paper_id)
            papers = session.scalars(statement).all()
            resources = []
            for paper in papers:
                files = session.scalars(select(PaperFileModel).where(PaperFileModel.paper_id == paper.paper_id)).all()
                pdf = next((file for file in files if file.file_type.casefold() in {"pdf", "fulltext", "full_text"} and Path(file.file_path).is_file()), None)
                if pdf is None:
                    continue
                text = next((file for file in files if file.file_type.casefold() in {"text", "txt", "parsed_text"} and Path(file.file_path).is_file()), None)
                metadata = PaperMetadata(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    authors=paper.authors or [],
                    abstract=paper.abstract,
                    doi=paper.doi,
                    arxiv_id=paper.arxiv_id,
                    year=paper.year,
                    venue=paper.venue,
                    citation_count=paper.citation_count,
                    rank_score=paper.rank_score,
                )
                resources.append(PaperResource(
                    paper=metadata,
                    availability=PaperAvailability.FULL_AVAILABLE,
                    file_path=pdf.file_path,
                    text_path=text.file_path if text else None,
                    embedding_status=paper.embedding_status,
                ))
        for resource in resources:
            pid = resource.paper.paper_id
            try:
                chunks = reader.ingest_paper(resource)
            except Exception as exc:
                failed += 1
                print(f"FAILED {pid}: {type(exc).__name__}: {exc}", flush=True)
                with database.get_session() as session:
                    row = session.get(PaperModel, pid)
                    row.embedding_status = "FAILED"
                    session.commit()
            else:
                success += 1
                print(f"READY {pid}: {len(chunks)} chunks", flush=True)
                with database.get_session() as session:
                    row = session.get(PaperModel, pid)
                    row.embedding_status = "COMPLETED"
                    session.commit()
    finally:
        database.dispose()
    return {"indexed": success, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("AUTO_PAPER_DATA_ROOT", "data/m2")))
    parser.add_argument("--paper-id")
    args = parser.parse_args()
    if not args.database_url:
        parser.error("set DATABASE_URL or pass --database-url")
    result = rebuild(args.database_url, args.data_root, args.paper_id)
    print(result)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
