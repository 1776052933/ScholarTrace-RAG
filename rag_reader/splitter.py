"""使用 RecursiveCharacterTextSplitter 切分论文文档。"""

from collections.abc import Mapping, Sequence
import re
from typing import Any, List, Optional

from .schema import DocumentChunk, PaperDocument


DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]


class DocumentSplitter:
    """按字符递归切片，并保留 paper_id、page 和 section。"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        splitter: Any = None,
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if (
            isinstance(chunk_overlap, bool)
            or not isinstance(chunk_overlap, int)
            or chunk_overlap < 0
            or chunk_overlap >= chunk_size
        ):
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = splitter

    def split(
        self,
        document: PaperDocument,
        *,
        pages: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[DocumentChunk]:
        """切分 PaperDocument；传入 pages 时保留原始页码信息。"""
        if not isinstance(document, PaperDocument):
            raise TypeError("document must be a PaperDocument")
        segments = (
            _page_segments(pages)
            if pages is not None
            else [{"text": document.content, "page": None, "section": "", "metadata": {}}]
        )
        return self._split_segments(document.paper_id, segments, title=document.title)

    def split_pages(
        self,
        paper_id: str,
        pages: Sequence[Mapping[str, Any]],
        *,
        title: str = "",
    ) -> List[DocumentChunk]:
        """直接切分 PDFParser 页面结构，并保留每个 Chunk 的页码。"""
        paper_id = paper_id.strip()
        if not paper_id:
            raise ValueError("paper_id must not be empty")
        return self._split_segments(paper_id, _page_segments(pages), title=title)

    def _split_segments(
        self,
        paper_id: str,
        segments: Sequence[Mapping[str, Any]],
        *,
        title: str,
    ) -> List[DocumentChunk]:
        splitter = self._get_splitter()
        chunks, chunk_index = [], 0
        current_section = ""
        for segment in segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            section = str(segment.get("section") or current_section).strip()
            for block_section, block_text in _section_blocks(text, section):
                if block_section:
                    current_section = block_section
                for chunk_text in splitter.split_text(block_text):
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    chunk_index += 1
                    metadata = dict(segment.get("metadata") or {})
                    metadata.update({
                        "chunk_index": chunk_index,
                        "character_count": len(chunk_text),
                    })
                    if title:
                        metadata["title"] = title
                    chunks.append(DocumentChunk(
                        chunk_id=f"{paper_id}-chunk-{chunk_index:05d}",
                        paper_id=paper_id,
                        text=chunk_text,
                        page=segment.get("page"),
                        section=block_section or current_section,
                        metadata=metadata,
                    ))
        return chunks

    def _get_splitter(self) -> Any:
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as exc:
                raise ImportError(
                    "DocumentSplitter requires langchain-text-splitters: "
                    "pip install langchain-text-splitters"
                ) from exc
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=DEFAULT_SEPARATORS,
                is_separator_regex=False,
            )
        return self._splitter


def _page_segments(pages: Sequence[Mapping[str, Any]]) -> List[dict[str, Any]]:
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
        raise TypeError("pages must be a sequence of mappings")
    result = []
    for index, page in enumerate(pages):
        if not isinstance(page, Mapping) or not isinstance(page.get("text"), str):
            raise TypeError(f"pages[{index}] must contain string text")
        page_number = page.get("page", index + 1)
        if page_number is not None and (
            isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1
        ):
            raise ValueError(f"pages[{index}].page must be a positive integer or None")
        result.append({
            "text": page["text"],
            "page": page_number,
            "section": page.get("section") or "",
            "metadata": page.get("metadata") or {},
        })
    return result


def _section_blocks(text: str, initial_section: str) -> List[tuple[str, str]]:
    """按常见论文标题行分块，使跨 Chunk 的 section 保持稳定。"""
    blocks, section, content = [], initial_section, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _is_section_heading(stripped):
            if content:
                blocks.append((section, "\n".join(content).strip()))
                content = []
            section = stripped
            content.append(stripped)
        else:
            content.append(line)
    if content:
        blocks.append((section, "\n".join(content).strip()))
    return [(section_name, block) for section_name, block in blocks if block]


def _is_section_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if re.match(r"^\d+(?:\.\d+)*[.)]?\s+[A-Z\u4e00-\u9fff]", line):
        return True
    normalized = line.casefold().strip(":：")
    return normalized in {
        "abstract", "introduction", "related work", "background", "method",
        "methods", "methodology", "experiments", "experiment", "results",
        "discussion", "conclusion", "limitations", "future work",
        "摘要", "引言", "相关工作", "方法", "实验", "结果", "讨论", "结论", "局限性",
    }
