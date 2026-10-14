"""Semantic Scholar 学术论文搜索引擎。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from paper_downloader.src.search_engines.base_search import SearchEngine


class SemanticScholarSearch(SearchEngine):
    """使用 Semantic Scholar Graph API 搜索元数据和开放 PDF。"""

    ENGINE_NAME = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
    FIELDS = "title,authors,year,abstract,externalIds,openAccessPdf,url,citationCount,venue"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._session = requests.Session()
        api_key = self.config.get("semantic_scholar", {}).get("api_key")
        if api_key:
            self._session.headers.update({"x-api-key": api_key})
        self._session.headers.update({"User-Agent": "AutoPaper/0.2"})

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            response = self._session.get(
                f"{self.BASE_URL}/search",
                params={
                    "query": self._clean_query(query),
                    "limit": min(max_results, 100),
                    "fields": self.FIELDS,
                },
                timeout=int(self.config.get("timeout", {}).get("search", 30)),
            )
            response.raise_for_status()
            papers = [self._parse_paper(item) for item in response.json().get("data", [])]
            min_year = kwargs.get("min_year")
            if min_year:
                papers = [p for p in papers if p.get("year") and int(p["year"]) >= int(min_year)]
            return self.normalize_results(papers)
        except (requests.RequestException, ValueError) as exc:
            self.logger.error("Semantic Scholar 搜索失败: %s", exc)
            return []

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        clean = identifier.replace("https://doi.org/", "").strip()
        if clean.startswith("10."):
            clean = f"DOI:{clean}"
        elif self._looks_like_arxiv_id(clean):
            clean = f"ARXIV:{clean}"
        try:
            response = self._session.get(
                f"{self.BASE_URL}/{clean}",
                params={"fields": self.FIELDS},
                timeout=int(self.config.get("timeout", {}).get("search", 30)),
            )
            response.raise_for_status()
            return self._normalize_single(self._parse_paper(response.json()), self.ENGINE_NAME)
        except (requests.RequestException, ValueError):
            return None

    @staticmethod
    def _looks_like_arxiv_id(value: str) -> bool:
        import re
        return bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[\w-]+/\d{7})(?:v\d+)?", value))

    @staticmethod
    def _parse_paper(item: Dict[str, Any]) -> Dict[str, Any]:
        external = item.get("externalIds") or {}
        oa = item.get("openAccessPdf") or {}
        return {
            "title": item.get("title") or "",
            "authors": [a.get("name", "") for a in item.get("authors", []) if a.get("name")],
            "year": str(item["year"]) if item.get("year") else None,
            "abstract": item.get("abstract"),
            "doi": external.get("DOI"),
            "arxiv_id": external.get("ArXiv"),
            "pdf_url": oa.get("url"),
            "url": item.get("url"),
            "citation_count": item.get("citationCount"),
            "journal": item.get("venue"),
        }

    def close(self) -> None:
        self._session.close()
