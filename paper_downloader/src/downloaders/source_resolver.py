"""将论文元数据解析为多个可回退的开放 PDF 下载地址。"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests

from paper_downloader.src.models.paper import Paper


@dataclass(frozen=True)
class DownloadCandidate:
    """一个待尝试的 PDF 地址及其来源。"""

    url: str
    source: str


class DownloadSourceResolver:
    """按优先级从 arXiv、OpenAlex、S2、Unpaywall 和 Europe PMC 找 PDF。"""

    DEFAULT_ENGINES = [
        "arxiv",
        "direct",
        "openalex",
        "semantic_scholar",
        "unpaywall",
        "europe_pmc",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._session = requests.Session()
        email = self.config.get("contact_email", "auto-paper@example.com")
        self._session.headers.update({
            "User-Agent": f"AutoPaper/0.2 (academic research; mailto:{email})",
            "Accept": "application/json",
        })

    def resolve(self, paper: Paper) -> List[DownloadCandidate]:
        """返回去重后的候选地址，强确定性直链排在开放获取 API 前。"""
        enabled = self.config.get("download", {}).get("source_engines", self.DEFAULT_ENGINES)
        candidates: List[DownloadCandidate] = []
        weak_direct: List[DownloadCandidate] = []

        if "arxiv" in enabled and paper.arxiv_id:
            arxiv_id = self._clean_arxiv_id(paper.arxiv_id)
            candidates.append(DownloadCandidate(
                f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "arxiv",
            ))

        if "direct" in enabled and paper.pdf_url:
            direct = DownloadCandidate(paper.pdf_url, "direct")
            if self._looks_like_pdf_url(paper.pdf_url):
                candidates.append(direct)
            else:
                weak_direct.append(direct)

        # arXiv/直链已经是强 PDF 地址，无需为每次下载额外请求多个元数据 API。
        has_strong_candidate = any(c.source in ("arxiv", "direct") for c in candidates)
        resolve_all = bool(self.config.get("download", {}).get("resolve_all_sources", False))

        if not has_strong_candidate or resolve_all:
            resolvers = {
                "openalex": self._resolve_openalex,
                "semantic_scholar": self._resolve_semantic_scholar,
                "unpaywall": self._resolve_unpaywall,
                "europe_pmc": self._resolve_europe_pmc,
            }
            jobs = [
                (name, resolvers[name])
                for name in enabled
                if name in resolvers
            ]
            resolved: Dict[str, List[DownloadCandidate]] = {}
            if jobs:
                workers = min(
                    len(jobs),
                    int(self.config.get("concurrency", {}).get("max_resolvers", 4)),
                )
                with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                    futures = {
                        executor.submit(resolver, paper): name
                        for name, resolver in jobs
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            resolved[name] = future.result()
                        except Exception as exc:  # 单个开放源不可用不应阻断下载
                            self.logger.warning("%s 地址解析失败: %s", name, exc)
            for name, _ in jobs:
                candidates.extend(resolved.get(name, []))

        candidates.extend(weak_direct)

        # 某些搜索源把真正的 PDF 放在 url 字段中。
        if "direct" in enabled and paper.url and self._looks_like_pdf_url(paper.url):
            candidates.append(DownloadCandidate(paper.url, "direct"))

        return self._deduplicate(candidates)

    def _resolve_openalex(self, paper: Paper) -> List[DownloadCandidate]:
        if paper.doi:
            url = f"https://api.openalex.org/works/doi:{quote(self._clean_doi(paper.doi), safe='')}"
            params = {}
        else:
            return []
        api_key = str(self.config.get("openalex_api_key") or "").strip()
        if api_key:
            params["api_key"] = api_key
        mailto = str(self.config.get("contact_email") or "").strip()
        if mailto:
            params["mailto"] = mailto
        data = self._get_json(url, params=params)
        if "results" in data:
            results = data.get("results") or []
            data = results[0] if results else {}
        location = data.get("best_oa_location") or {}
        pdf_url = location.get("pdf_url")
        return [DownloadCandidate(pdf_url, "openalex")] if pdf_url else []

    def _resolve_semantic_scholar(self, paper: Paper) -> List[DownloadCandidate]:
        base = "https://api.semanticscholar.org/graph/v1/paper"
        fields = "openAccessPdf,externalIds,title"
        if paper.doi:
            data = self._get_json(
                f"{base}/DOI:{quote(self._clean_doi(paper.doi), safe='')}",
                params={"fields": fields},
            )
        elif paper.arxiv_id:
            data = self._get_json(
                f"{base}/ARXIV:{quote(self._clean_arxiv_id(paper.arxiv_id), safe='')}",
                params={"fields": fields},
            )
        else:
            return []
        pdf_url = (data.get("openAccessPdf") or {}).get("url")
        return [DownloadCandidate(pdf_url, "semantic_scholar")] if pdf_url else []

    def _resolve_unpaywall(self, paper: Paper) -> List[DownloadCandidate]:
        if not paper.doi:
            return []
        email = self.config.get("contact_email")
        if not email:
            self.logger.debug("未配置 contact_email，跳过 Unpaywall")
            return []
        data = self._get_json(
            f"https://api.unpaywall.org/v2/{quote(self._clean_doi(paper.doi), safe='')}",
            params={"email": email},
        )
        locations = [data.get("best_oa_location") or {}]
        locations.extend(data.get("oa_locations") or [])
        return [
            DownloadCandidate(location["url_for_pdf"], "unpaywall")
            for location in locations
            if location.get("url_for_pdf")
        ]

    def _resolve_europe_pmc(self, paper: Paper) -> List[DownloadCandidate]:
        if paper.doi:
            query_text = f'DOI:"{self._clean_doi(paper.doi)}"'
        else:
            return []
        data = self._get_json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query_text, "format": "json", "pageSize": 1},
        )
        rows = data.get("resultList", {}).get("result", [])
        if not rows:
            return []
        pmcid = rows[0].get("pmcid")
        if not pmcid:
            return []
        return [DownloadCandidate(
            f"https://europepmc.org/articles/{pmcid}?pdf=render",
            "europe_pmc",
        )]

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._session.get(
            url,
            params=params,
            timeout=int(self.config.get("timeout", {}).get("search", 30)),
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _clean_doi(value: str) -> str:
        return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I)

    @staticmethod
    def _clean_arxiv_id(value: str) -> str:
        clean = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value.strip(), flags=re.I)
        return clean.removesuffix(".pdf")

    @staticmethod
    def _looks_like_pdf_url(url: str) -> bool:
        lowered = url.lower()
        return (
            lowered.endswith(".pdf")
            or "arxiv.org/pdf/" in lowered
            or "/article/download/" in lowered
            or "pdf=render" in lowered
            or "type=pdf" in lowered
        )

    @staticmethod
    def _deduplicate(candidates: Iterable[DownloadCandidate]) -> List[DownloadCandidate]:
        result: List[DownloadCandidate] = []
        seen = set()
        for candidate in candidates:
            if not candidate.url or not candidate.url.startswith(("http://", "https://")):
                continue
            key = candidate.url.rstrip("/")
            if key not in seen:
                seen.add(key)
                result.append(candidate)
        return result

    def close(self) -> None:
        self._session.close()
