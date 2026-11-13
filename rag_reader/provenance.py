"""Conservative source-role hints; bibliography stays available for citation queries."""
from __future__ import annotations

import re
from typing import Any, Mapping


def bibliography_entry(text: str) -> bool:
    text = text.strip()
    # Author-list/year punctuation distinguishes an entry from a narrative citation.
    return bool(re.match(r"^(?:\[\d+\]\s*)?(?:(?:van|von|de|del|da)\s+)?[A-ZÀ-ÖØ-Þ][\w'’\-]+,\s*[A-Z]\.", text)
                and re.search(r"\((?:19|20)\d{2}[a-z]?\)\.\s", text[:350]))


def source_role(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    metadata = metadata or {}
    label = str(metadata.get('item_type') or '').casefold()
    section = str(metadata.get('section') or metadata.get('source_section') or '').strip().casefold()
    if (label in {'reference', 'bibliography'} or section in {'references', 'bibliography', 'citation:'}
            or re.match(r'^Citation:\s*\d+:', text.strip(), re.I) or bibliography_entry(text)):
        return 'bibliography'
    if label in {'section_header', 'title'}:
        return 'heading'
    if label == 'table':
        return 'table'
    from .splitter import _is_section_heading
    if '\n' not in text.strip() and _is_section_heading(text.strip()):
        return 'heading'
    return 'body'


def needs_paper_body(question: str) -> bool:
    """Gate by question text only; asking for citations is not a bibliography query."""
    if re.search(r'参考文献|引用了|被引用|书目|bibliograph|which.{0,50}(?:cited|referenced)|'
                 r'who.{0,30}cit|list.{0,30}references', question, re.I):
        return False
    return bool(re.search(
        r'(?:论文|本文|研究).{0,25}(?:研究什么|任务|贡献|提出|研究内容|主要研究|做了什么)|'
        r'(?:paper|article|study|work).{0,60}(?:about|focus|contribut|propos|task|investigat)|'
        r'(?:main|primary|research).{0,25}(?:task|contribution|objective|purpose)',
        question, re.I))


def answer_candidates(question: str, results: list[dict]) -> list[dict]:
    if not needs_paper_body(question):
        return results
    return [row for row in results
            if source_role(str(row['text']), row.get('metadata')) not in {'bibliography', 'heading'}]
