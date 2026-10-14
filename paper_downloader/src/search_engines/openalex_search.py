"""OpenAlex 学术论文搜索引擎。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from paper_downloader.src.search_engines.base_search import SearchEngine


class OpenAlexSearch(SearchEngine):
    """使用 OpenAlex 公共 REST API 搜索论文及开放获取 PDF。"""

    ENGINE_NAME = "openalex"
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._session = requests.Session()
        email = self.config.get("contact_email", "auto-paper@example.com")
        self._session.headers.update({"User-Agent": f"AutoPaper/0.2 (mailto:{email})"})

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "search": self._clean_query(query),
            "per-page": min(max_results, 100),
        }
        api_key = str(self.config.get("openalex_api_key") or "").strip()
        if api_key:
            params["api_key"] = api_key
        if kwargs.get("min_year"):
            params["filter"] = f"from_publication_date:{int(kwargs['min_year'])}-01-01"
        try:
            response = self._session.get(
                self.BASE_URL,
                params=params,
                timeout=int(self.config.get("timeout", {}).get("search", 30)),
            )
            response.raise_for_status()
            works = response.json().get("results", [])
            return self.normalize_results([self._parse_work(work) for work in works])
        except (requests.RequestException, ValueError) as exc:
            logging.getLogger(self.__class__.__name__).error("OpenAlex 搜索失败: %s", exc)
            return []

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        clean = identifier.replace("https://doi.org/", "").strip()
        target = f"doi:{clean}" if clean.startswith("10.") else clean
        try:
            response = self._session.get(
                f"{self.BASE_URL}/{target}",
                params={"api_key": self.config["openalex_api_key"]}
                if self.config.get("openalex_api_key") else None,
                timeout=int(self.config.get("timeout", {}).get("search", 30)),
            )
            response.raise_for_status()
            return self._normalize_single(self._parse_work(response.json()), self.ENGINE_NAME)
        except (requests.RequestException, ValueError):
            return None

    @staticmethod
    def _parse_work(work: Dict[str, Any]) -> Dict[str, Any]:
        best_oa = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        doi = work.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        authors = [
            authorship.get("author", {}).get("display_name", "")
            for authorship in work.get("authorships", [])
            if authorship.get("author", {}).get("display_name")
        ]
        return {
            "title": work.get("title") or work.get("display_name") or "",
            "authors": authors,
            "year": str(work["publication_year"]) if work.get("publication_year") else None,
            "abstract": None,
            "doi": doi,
            "arxiv_id": None,
            "pdf_url": best_oa.get("pdf_url"),
            "url": primary.get("landing_page_url") or work.get("id"),
            "citation_count": work.get("cited_by_count"),
            "journal": source.get("display_name"),
        }

    def close(self) -> None:
        self._session.close()
