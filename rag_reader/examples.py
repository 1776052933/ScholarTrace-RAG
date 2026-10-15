"""使用真实 PDF、向量模型、Chroma 和 DeepSeek 的完整 RAG 示例。

运行方式：

    python -m rag_reader.examples path/to/paper.pdf --title "论文标题"

也支持从项目根目录直接运行：

    python rag_reader/examples.py path/to/paper.pdf --title "论文标题"
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from literature_database.schema import (  # noqa: E402
    PaperAvailability,
    PaperMetadata,
    PaperResource,
)
from rag_reader.extractor import PaperKnowledgeExtractor  # noqa: E402
from rag_reader.llm_client import DeepSeekClient  # noqa: E402
from rag_reader.pipeline import RAGReader  # noqa: E402
from rag_reader.vector_store import VectorStore  # noqa: E402


DEFAULT_QUESTION = "该论文提出的方法是什么？"


def run_example(
    pdf_path: str,
    *,
    title: str = "",
    paper_id: str = "",
    question: str = DEFAULT_QUESTION,
    persist_directory: str = "./rag_reader_chroma",
) -> Dict[str, Any]:
    """执行 PDF 导入、建库、知识抽取、摘要和问答完整流程。"""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    if path.suffix.casefold() != ".pdf":
        raise ValueError(f"输入文件必须是 PDF：{path}")

    resolved_title = title.strip() or path.stem
    resolved_paper_id = paper_id.strip() or path.stem
    resource = PaperResource(
        paper=PaperMetadata(
            paper_id=resolved_paper_id,
            title=resolved_title,
        ),
        availability=PaperAvailability.FULL_AVAILABLE,
        file_path=str(path),
        embedding_status="pending",
    )

    # 共享同一个 DeepSeekClient，使知识抽取、摘要和问答共用配置与 token 统计。
    llm_client = DeepSeekClient()
    vector_store = VectorStore(persist_directory=persist_directory)
    extractor = PaperKnowledgeExtractor(llm_client=llm_client)
    reader = RAGReader(
        vector_store=vector_store,
        extractor=extractor,
        llm_client=llm_client,
    )

    chunks = reader.ingest_paper(resource)
    knowledge = reader.extract_knowledge(resolved_paper_id)
    summary = reader.summarize(resolved_paper_id)
    answer = reader.ask(resolved_paper_id, question)

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    source_chunks = []
    for chunk_id in answer.sources:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        source_chunks.append({
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
        })

    return {
        "ingestion": {
            "paper_id": resolved_paper_id,
            "title": resolved_title,
            "pdf_path": str(path),
            "vector_store": str(Path(persist_directory).expanduser().resolve()),
            "chunk_count": len(chunks),
        },
        "knowledge": knowledge.model_dump(mode="json"),
        "summary": summary,
        "qa": {
            **answer.model_dump(mode="json"),
            "source_chunks": source_chunks,
        },
        "token_usage": llm_client.get_token_usage(),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单篇论文的 RAG Reader 完整示例")
    parser.add_argument("pdf", help="本地论文 PDF 路径")
    parser.add_argument("--title", default="", help="论文标题，默认使用 PDF 文件名")
    parser.add_argument("--paper-id", default="", help="论文 ID，默认使用 PDF 文件名")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="向论文提出的问题")
    parser.add_argument(
        "--persist-directory",
        default="./rag_reader_chroma",
        help="Chroma 持久化目录",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = run_example(
        args.pdf,
        title=args.title,
        paper_id=args.paper_id,
        question=args.question,
        persist_directory=args.persist_directory,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
