"""arXiv Atom API 论文搜索。"""

import logging
import re
from typing import List
from urllib import error, parse, request
import xml.etree.ElementTree as ET

from ..schema import PaperMetadata


logger = logging.getLogger(__name__)


class ArxivTool:
    source = "arxiv"
    _NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, timeout_seconds: float = 20.0):
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        limit: int = 10,
        publication_date_to: str | None = None,
    ) -> List[PaperMetadata]:
        """搜索 arXiv；任何 API 失败均返回空列表。"""
        if not query.strip() or limit < 1:
            return []
        search_query = f"all:{query}"
        cutoff = _compact_date(publication_date_to)
        if cutoff:
            search_query += f" AND submittedDate:[000000000000 TO {cutoff}2359]"
        params = parse.urlencode({
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 100),
            "sortBy": "relevance",
        })
        req = request.Request(
            f"https://export.arxiv.org/api/query?{params}",
            headers={"User-Agent": "Auto-paper-search-agent/1.0"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                root = ET.fromstring(response.read())
        except (error.URLError, error.HTTPError, TimeoutError, OSError, ET.ParseError) as exc:
            logger.warning("arXiv API 搜索失败：%s", exc)
            return []

        papers = []
        for entry in root.findall("atom:entry", self._NS):
            try:
                title = _text(entry, "atom:title", self._NS)
                if not title:
                    continue
                published = _text(entry, "atom:published", self._NS)
                entry_url = _text(entry, "atom:id", self._NS)
                arxiv_id = _arxiv_id(entry_url)
                pdf_url = next(
                    (
                        str(link.attrib.get("href") or "").strip()
                        for link in entry.findall("atom:link", self._NS)
                        if link.attrib.get("title") == "pdf"
                        or link.attrib.get("type") == "application/pdf"
                    ),
                    "",
                )
                if not pdf_url and arxiv_id:
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                papers.append(PaperMetadata(
                    title=" ".join(title.split()),
                    authors=[_text(author, "atom:name", self._NS) for author in entry.findall("atom:author", self._NS)],
                    abstract=" ".join(_text(entry, "atom:summary", self._NS).split()),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    publication_date=published[:10] if len(published) >= 10 else None,
                    venue="arXiv",
                    citation_count=None,
                    url=entry_url,
                    source=self.source,
                    doi=_text(entry, "arxiv:doi", self._NS) or None,
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url or None,
                    sources=[self.source],
                ))
            except (TypeError, ValueError):
                logger.debug("跳过格式异常的 arXiv 论文记录", exc_info=True)
        return papers


def search(query: str, limit: int = 10) -> List[PaperMetadata]:
    """模块级便捷接口。"""
    return ArxivTool().search(query, limit)


def _text(node, path, namespaces) -> str:
    child = node.find(path, namespaces)
    return child.text.strip() if child is not None and child.text else ""


def _arxiv_id(value: str) -> str | None:
    match = re.search(r"(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None


def _compact_date(value: str | None) -> str:
    compact = re.sub(r"[^0-9]", "", value or "")
    return compact[:8] if len(compact) >= 8 else ""
