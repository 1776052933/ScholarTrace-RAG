"""Seed a completed discovery job from local PDFs or paper metadata for M4 acceptance.

Run inside the API container after migrations. This helper itself calls no search or model API.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from app.storage.models import ResearchJobModel
from literature_database.database import DatabaseManager
from literature_database.models import PaperFileModel, PaperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="append", default=[],
                        help="paper_id|title|absolute_pdf_path; repeat for each paper")
    parser.add_argument("--metadata-only", action="append", default=[],
                        help="paper_id|title; repeat to test the real download path")
    args = parser.parse_args()
    if not args.paper and not args.metadata_only:
        parser.error("at least one --paper or --metadata-only is required")
    papers = []
    for raw in args.paper:
        try:
            paper_id, title, filename = (part.strip() for part in raw.split("|", 2))
        except ValueError as exc:
            raise SystemExit("--paper requires paper_id|title|absolute_pdf_path") from exc
        path = Path(filename)
        if not paper_id or not title or not path.is_absolute() or not path.is_file():
            raise SystemExit(f"invalid paper entry or missing PDF: {raw}")
        papers.append((paper_id, title, str(path.resolve())))
    for raw in args.metadata_only:
        try:
            paper_id, title = (part.strip() for part in raw.split("|", 1))
        except ValueError as exc:
            raise SystemExit("--metadata-only requires paper_id|title") from exc
        if not paper_id or not title:
            raise SystemExit(f"invalid metadata-only entry: {raw}")
        papers.append((paper_id, title, None))
    if len({item[0] for item in papers}) != len(papers):
        raise SystemExit("paper IDs must be unique")

    database = DatabaseManager(os.environ["DATABASE_URL"])
    source_id = uuid4().hex
    try:
        with database.get_session() as session:
            for paper_id, title, filename in papers:
                paper = session.get(PaperModel, paper_id)
                if paper is None:
                    paper = PaperModel(paper_id=paper_id, title=title, authors=[])
                    session.add(paper)
                session.flush()
                if filename is not None:
                    session.add(PaperFileModel(paper_id=paper_id, file_path=filename,
                                               status="AVAILABLE", file_type="pdf"))
            session.add(ResearchJobModel(
                job_id=source_id, job_type="research", query="M4 offline acceptance",
                options={"query": "M4 offline acceptance", "download_papers": False},
                status="completed", stage="ranking_agent", progress=100,
                logs=[], retryable=False,
                result={"stages": {
                    "ranking_agent": {"result": {"papers": [
                        {"paper_id": paper_id, "title": title}
                        for paper_id, title, _ in papers
                    ]}},
                    "literature_database": {"download_enabled": False},
                }},
            ))
            session.commit()
    finally:
        database.dispose()
    print(json.dumps({"discovery_job_id": source_id,
                      "paper_ids": [item[0] for item in papers]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
