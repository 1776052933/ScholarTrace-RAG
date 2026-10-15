"""论文页面文本清洗、断行合并和重复内容删除。"""

from collections import Counter
import logging
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, List


logger = logging.getLogger(__name__)

FORMULA_PLACEHOLDER = "[FORMULA]"


class TextCleaner:
    """将 PDFParser 的页面列表清洗为连续论文文本。"""

    def __init__(self, *, repeated_line_ratio: float = 0.5) -> None:
        if not 0.0 < repeated_line_ratio <= 1.0:
            raise ValueError("repeated_line_ratio must be between 0 and 1")
        self.repeated_line_ratio = repeated_line_ratio

    def clean(self, raw_pages: Sequence[Mapping[str, Any]]) -> str:
        """Clean pages and return the combined text for summarization."""
        return "\n\n".join(page["text"] for page in self.clean_page_records(raw_pages))

    def clean_page_records(
        self, raw_pages: Sequence[Mapping[str, Any]]
    ) -> List[dict[str, Any]]:
        """Clean text while keeping each paragraph attached to its source page."""
        pages = _validate_pages(raw_pages)
        if not pages:
            return []
        removable = self._repeated_margins(pages)
        seen: set[str] = set()
        records: List[dict[str, Any]] = []
        paragraph_count = 0
        for page in pages:
            lines = page["text"].replace("\r\n", "\n").replace("\r", "\n").split("\n")
            lines = self._remove_margins(lines, removable)
            text = _replace_formulas(_remove_garbled_characters("\n".join(lines)))
            kept: List[str] = []
            for paragraph in _merge_broken_lines(text.splitlines()):
                value = re.sub(r"\s+", " ", paragraph).strip()
                if not value:
                    continue
                key = value.casefold()
                if value != FORMULA_PLACEHOLDER and key in seen:
                    continue
                seen.add(key)
                kept.append(value)
            if kept:
                records.append({"page": page["page"], "text": "\n\n".join(kept)})
                paragraph_count += len(kept)
        logger.info(
            "论文文本清洗完成：pages=%d, paragraphs=%d, characters=%d",
            len(pages), paragraph_count, sum(len(page["text"]) for page in records),
        )
        return records

    def clean_pages(self, raw_pages: Sequence[Mapping[str, Any]]) -> str:
        """Backward-compatible alias returning combined cleaned text."""
        return self.clean(raw_pages)
    def _repeated_margins(self, pages: List[dict[str, Any]]) -> set[str]:
        if len(pages) < 2:
            return set()
        candidates = []
        for page in pages:
            lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
            if lines:
                candidates.extend([_margin_key(lines[0]), _margin_key(lines[-1])])
        threshold = max(2, math.ceil(len(pages) * self.repeated_line_ratio))
        return {
            key for key, count in Counter(candidates).items()
            if key and count >= threshold
        }

    def _remove_margins(self, lines: List[str], removable: set[str]) -> List[str]:
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        if not nonempty:
            return lines
        first, last = nonempty[0], nonempty[-1]
        for index in {first, last}:
            stripped = lines[index].strip()
            if _margin_key(stripped) in removable or _is_page_number(stripped):
                lines[index] = ""
        return lines


def _validate_pages(raw_pages: Sequence[Mapping[str, Any]]) -> List[dict[str, Any]]:
    if isinstance(raw_pages, (str, bytes)) or not isinstance(raw_pages, Sequence):
        raise TypeError("raw_pages must be a sequence of page mappings")
    result = []
    for index, page in enumerate(raw_pages):
        if not isinstance(page, Mapping) or "text" not in page:
            raise TypeError(f"raw_pages[{index}] must contain text")
        text = page["text"]
        if not isinstance(text, str):
            raise TypeError(f"raw_pages[{index}].text must be a string")
        result.append({"page": page.get("page", index + 1), "text": text})
    return result


def _margin_key(value: str) -> str:
    normalized = re.sub(r"\d+", "#", value.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _is_page_number(value: str) -> bool:
    return bool(re.fullmatch(r"(?:page\s*)?\d+(?:\s*(?:/|of)\s*\d+)?", value, re.IGNORECASE))


def _remove_garbled_characters(text: str) -> str:
    cleaned = []
    for character in text.replace("\ufffd", ""):
        category = unicodedata.category(character)
        if category in {"Cc", "Cs", "Co", "Cn"} and character not in {"\n", "\t"}:
            continue
        cleaned.append(character)
    value = "".join(cleaned)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value


def _replace_formulas(text: str) -> str:
    patterns = (
        r"\$\$.*?\$\$",
        r"\\\[.*?\\\]",
        r"\\begin\{(?:equation\*?|align\*?|gather\*?)\}.*?\\end\{(?:equation\*?|align\*?|gather\*?)\}",
    )
    result = text
    for pattern in patterns:
        result = re.sub(pattern, FORMULA_PLACEHOLDER, result, flags=re.DOTALL)
    result = re.sub(r"(?<!\$)\$(?!\$)[^\n$]+\$(?!\$)", FORMULA_PLACEHOLDER, result)

    lines = []
    for line in result.splitlines():
        stripped = line.strip()
        if stripped and _looks_like_formula(stripped):
            lines.append(FORMULA_PLACEHOLDER)
        else:
            lines.append(line)
    return "\n".join(lines)


def _looks_like_formula(line: str) -> bool:
    if len(line) > 240 or re.search(r"[。！？.!?]", line):
        return False
    math_symbols = len(re.findall(r"[=+×÷∑∏√∞≈≠≤≥∫∂^_{}]|\\[a-zA-Z]+", line))
    alphanumerics = len(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]", line))
    return math_symbols >= 2 and math_symbols / max(1, alphanumerics + math_symbols) >= 0.2


def _merge_broken_lines(lines: Sequence[str]) -> List[str]:
    paragraphs, current = [], ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue
        if _is_standalone_line(line):
            if current:
                paragraphs.append(current.strip())
                current = ""
            paragraphs.append(line)
            continue
        if not current:
            current = line
        elif current.endswith("-") and re.match(r"^[a-z]", line):
            current = current[:-1] + line
        elif _cjk_boundary(current, line):
            current += line
        else:
            current += " " + line
        if re.search(r"[。！？.!?;；:]$", current):
            paragraphs.append(current.strip())
            current = ""
    if current:
        paragraphs.append(current.strip())
    return paragraphs


def _is_standalone_line(line: str) -> bool:
    return (
        line == FORMULA_PLACEHOLDER
        or bool(re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[•●▪◦*-]\s+)", line))
        or (len(line) <= 100 and line.isupper() and any(character.isalpha() for character in line))
    )


def _cjk_boundary(left: str, right: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"^[\u4e00-\u9fff]", right))


def _deduplicate_paragraphs(paragraphs: Sequence[str]) -> List[str]:
    seen, result = set(), []
    for paragraph in paragraphs:
        value = re.sub(r"\s+", " ", paragraph).strip()
        if not value:
            continue
        key = value.casefold()
        if value != FORMULA_PLACEHOLDER and key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
