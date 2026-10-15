"""Optional Docling path for PDFs with complex layout and tables."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .parser import PDFOpenError, PDFParserError, EmptyPDFTextError


class DoclingPDFParser:
    def __init__(self, *, converter: Any = None) -> None:
        self._converter = converter

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path).expanduser()
        if not path.is_file() or path.suffix.casefold() != ".pdf":
            raise PDFOpenError(f"PDF file not found or invalid: {path}")
        converter = self._converter
        if converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise ImportError("Docling PDF mode requires: pip install docling") from exc
            converter = DocumentConverter()
            self._converter = converter
        try:
            document = converter.convert(path).document
        except Exception as exc:
            raise PDFParserError(f"Docling failed to convert PDF: {path}") from exc

        records: list[dict[str, Any]] = []
        section = ""
        for item, _level in document.iterate_items():
            label = str(getattr(item, "label", "")).split(".")[-1].lower()
            if label == "table":
                text = item.export_to_markdown(document).strip()
            else:
                text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            if label in {"section_header", "title"}:
                section = text
            provenance = getattr(item, "prov", None) or []
            if not provenance:
                continue  # Never invent a PDF page for a cited passage.
            page = int(provenance[0].page_no)
            if page < 1:
                continue
            records.append({
                "page": page,
                "section": section,
                "text": text,
                "metadata": {
                    "parser": "docling",
                    "item_type": label,
                    "source_ref": str(getattr(item, "self_ref", "") or ""),
                },
            })
        if not records:
            raise EmptyPDFTextError(f"Docling found no page-located text: {path}")
        return records
