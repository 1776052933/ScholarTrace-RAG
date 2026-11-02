"""通过 SerpAPI 调用 Google Scholar 论文搜索。"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from ..schema import PaperMetadata


logger = logging.getLogger(__name__)
SERPAPI_URL = "https://serpapi.com/search.json"


class GoogleScholarTool:
    """Google Scholar 搜索工具，API Key 由 SerpAPI 提供。"""

    source = "google_scholar"

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 20.0):
        self.api_key = _get_api_key() if api_key is None else api_key.strip()
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        limit: int = 10,
        publication_date_to: str | None = None,
    ) -> List[PaperMetadata]:
        """搜索 Google Scholar；缺少 Key 或 API 失败时返回空列表。"""
        if not query.strip() or limit < 1:
            return []
        if not self.api_key:
            logger.warning("Google Scholar 搜索已跳过：未配置 SerpAPI Key")
            return []

        values = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": min(limit, 20),
        }
        cutoff_year = _date_year(publication_date_to)
        if cutoff_year is not None:
            values["as_yhi"] = cutoff_year
        params = parse.urlencode(values)
        req = request.Request(
            f"{SERPAPI_URL}?{params}",
            headers={"User-Agent": "Auto-paper-search-agent/1.0"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Google Scholar API 搜索失败：%s", exc)
            return []

        papers = []
        for item in payload.get("organic_results", []):
            try:
                paper = _parse_paper(item)
                if paper is not None:
                    papers.append(paper)
            except (TypeError, ValueError):
                logger.debug("跳过格式异常的 Google Scholar 论文记录", exc_info=True)
        return papers[:limit]


def search(query: str, limit: int = 10) -> List[PaperMetadata]:
    """模块级便捷接口，自动从环境变量或 API_key 读取 Key。"""
    return GoogleScholarTool().search(query, limit)


def _get_api_key() -> str:
    env_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        from API_key.key_manager import key_get

        return str(key_get("serpapi") or "").strip()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        logger.debug("无法从 API_key/key_manager.py 读取 serpapi", exc_info=True)
        return ""


def _parse_paper(item: Dict[str, Any]) -> Optional[PaperMetadata]:
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    publication = item.get("publication_info") or {}
    summary = str(publication.get("summary") or "")
    authors = [
        str(author.get("name") or "").strip()
        for author in publication.get("authors") or []
        if isinstance(author, dict) and author.get("name")
    ]
    if not authors and summary:
        author_text = summary.split(" - ", 1)[0]
        authors = [name.strip() for name in author_text.split(",") if name.strip()]

    years = re.findall(r"\b(?:19|20)\d{2}\b", summary)
    link = str(item.get("link") or "")
    pdf_url = next(
        (
            str(resource.get("link") or "").strip()
            for resource in item.get("resources") or []
            if isinstance(resource, dict)
            and str(resource.get("file_format") or "").casefold() == "pdf"
            and resource.get("link")
        ),
        "",
    )
    cited_by = ((item.get("inline_links") or {}).get("cited_by") or {}).get("total")
    return PaperMetadata(
        title=title,
        authors=authors,
        abstract=str(item.get("snippet") or ""),
        year=int(years[-1]) if years else None,
        venue=summary,
        citation_count=_non_negative_int(cited_by),
        url=link,
        source="google_scholar",
        doi=_extract_doi(link),
        arxiv_id=_extract_arxiv_id(link),
        pdf_url=pdf_url or (_direct_pdf_url(link) or None),
        sources=["google_scholar"],
    )


def _non_negative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def _extract_doi(value: str) -> Optional[str]:
    match = re.search(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^?#\s]+)", value, re.IGNORECASE)
    return match.group(1).rstrip(".,;)") if match else None


def _extract_arxiv_id(value: str) -> Optional[str]:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None


def _direct_pdf_url(value: str) -> str:
    lowered = value.casefold()
    return value if lowered.endswith(".pdf") or "/pdf/" in lowered else ""


def _date_year(value: str | None) -> Optional[int]:
    match = re.match(r"\s*((?:19|20)\d{2})", value or "")
    return int(match.group(1)) if match else None
