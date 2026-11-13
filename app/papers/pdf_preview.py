"""Render a cited PDF page and highlight text that can be located on that page."""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import re
from statistics import median
import unicodedata

import pymupdf as fitz


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[\w]+(?:[.\-][\w]+)*", normalized)


def _searchable_phrases(excerpt: str) -> list[str]:
    phrases: list[str] = []
    for raw_line in excerpt.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line or re.fullmatch(r"[|:\-\s]+", line):
            continue
        if "|" in line:
            for cell in line.split("|"):
                cell = cell.strip(" *`_")
                if len("".join(_tokens(cell))) >= 4 and len(cell) <= 100:
                    phrases.append(cell)
        else:
            # Short leading phrases survive most PDF line wraps and Docling cleanup.
            for segment in re.split(r"(?<=[.!?])\s+", line):
                segment = segment.strip(" *`_")
                if len(segment) >= 12:
                    phrases.append(segment[:120].rsplit(" ", 1)[0] if len(segment) > 120 else segment)
    return phrases


def _matching_rectangles(page: fitz.Page, excerpt: str) -> list[fitz.Rect]:
    candidates = _searchable_phrases(excerpt)
    is_table = "|" in excerpt
    found: list[fitz.Rect] = []
    if is_table:
        for row in excerpt.splitlines():
            cells = [cell.strip(" *`_") for cell in row.split("|")
                     if len("".join(_tokens(cell))) >= 4 and len(cell.strip()) <= 100]
            hits = [(cell, page.search_for(cell)) for cell in cells]
            anchors = [(rect.y0 + rect.y1) / 2 for _, matches in hits
                       if len(matches) == 1 for rect in matches]
            row_y = median(anchors) if anchors else None
            for _, matches in hits:
                if len(matches) == 1:
                    found.append(matches[0])
                elif row_y is not None and matches:
                    nearest = min(matches, key=lambda rect: abs((rect.y0 + rect.y1) / 2 - row_y))
                    if abs((nearest.y0 + nearest.y1) / 2 - row_y) <= 9:
                        found.append(nearest)
            if len(found) >= 48:
                break
    else:
        for phrase in candidates:
            matches = page.search_for(phrase)
            if matches:
                found.extend(matches[:8])
            if len(found) >= 32:
                break
    if found:
        unique = {tuple(round(value, 2) for value in rect): rect for rect in found}
        return list(unique.values())[:48]

    # PDF text may differ in punctuation or line breaks from a parsed chunk.
    words = page.get_text("words", sort=True)
    page_tokens: list[str] = []
    boxes: list[fitz.Rect] = []
    for word in words:
        for token in _tokens(str(word[4])):
            page_tokens.append(token)
            boxes.append(fitz.Rect(word[:4]))
    excerpt_tokens = _tokens(excerpt)
    if not page_tokens or not excerpt_tokens:
        return []
    blocks = SequenceMatcher(None, excerpt_tokens[:500], page_tokens, autojunk=False).get_matching_blocks()
    best = max(blocks, key=lambda block: block.size)
    if best.size < 6 or not any(len(token) >= 6 or token.isdigit()
                                     for token in page_tokens[best.b:best.b + best.size]):
        return []
    return boxes[best.b:best.b + min(best.size, 48)]


def render_cited_page(pdf_path: Path, page_number: int, excerpt: str) -> tuple[bytes, int]:
    """Return a PNG of the actual 1-based PDF page and the number of highlights."""
    with fitz.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("PDF 已加密，无法预览。")
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("引用页码超出 PDF 范围。")
        page = document[page_number - 1]
        rects = _matching_rectangles(page, excerpt)
        if rects:
            shape = page.new_shape()
            for rect in rects:
                shape.draw_rect(fitz.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1))
            shape.finish(color=(0.9, 0.55, 0), fill=(1, 0.85, 0.12),
                         fill_opacity=0.38)
            shape.commit(overlay=True)
        # Cap large-format pages while retaining readable text on normal pages.
        scale = min(1.7, (3_000_000 / max(page.rect.width * page.rect.height, 1)) ** 0.5)
        image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
        return image, len(rects)
