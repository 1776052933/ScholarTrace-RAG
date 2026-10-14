"""
crossref_search.py — CrossRef 搜索引擎

通过 CrossRef REST API (https://api.crossref.org/works) 进行论文检索。
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from paper_downloader.src.search_engines.base_search import SearchEngine


class CrossrefSearch(SearchEngine):
    """CrossRef 搜索引擎。

    通过 CrossRef REST API 按标题搜索论文并解析返回的 JSON 数据。

    Usage::

        engine = CrossrefSearch(config={"timeout": {"search": 30}})
        results = engine.search("machine learning", max_results=10)
        info = engine.get_paper_info("10.1038/nature14539")
    """

    ENGINE_NAME = "crossref"

    BASE_URL = "https://api.crossref.org/works"

    # CrossRef "礼貌" 邮箱（API 要求）
    _POLITE_MAILTO = "auto-paper@example.com"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._session = self._build_session()

    # ── HTTP 会话 ─────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """创建带重试和超时配置的 requests Session。"""
        session = requests.Session()

        # 重试策略
        retry_cfg = self.config.get("retry", {})
        retries = Retry(
            total=retry_cfg.get("max_attempts", 3),
            backoff_factor=retry_cfg.get("backoff_factor", 2),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # 代理
        proxies = self._get_proxy_dict()
        if proxies:
            session.proxies.update(proxies)

        return session

    def _get_proxy_dict(self) -> Optional[Dict[str, str]]:
        """从配置构建代理字典。"""
        proxy_cfg = self.config.get("proxy", {})
        if not proxy_cfg.get("enabled", False):
            return None
        proxies = {}
        for proto in ("http", "https"):
            url = proxy_cfg.get(proto, "")
            if url:
                user = proxy_cfg.get("username", "")
                pwd = proxy_cfg.get("password", "")
                if user and pwd and "://" in url:
                    scheme, rest = url.split("://", 1)
                    url = f"{scheme}://{user}:{pwd}@{rest}"
                proxies[proto] = url
        return proxies or None

    # ── 搜索 ──────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        """在 CrossRef 中按标题搜索论文。

        Args:
            query:       搜索关键词或标题。
            max_results: 最大返回数量（每页最多 100）。
            **kwargs:
                min_year:    最早发表年份。
                sort:        排序方式 (relevance / published / published-online)，默认 relevance。
                filter_type: 过滤类型 (journal-article / proceedings-article 等)。

        Returns:
            标准化论文信息列表。
        """
        query = self._clean_query(query)
        self.logger.info("CrossRef 搜索: '%s' (max_results=%d)", query, max_results)

        params: Dict[str, Any] = {
            "query.bibliographic": query,
            "rows": min(max_results, 100),
            "offset": 0,
        }

        # 排序
        sort_map = {
            "relevance":        "relevance",
            "published":        "published",
            "published-online": "published-online",
        }
        params["sort"] = sort_map.get(kwargs.get("sort", "relevance"), "relevance")

        # 类型过滤
        filter_parts = []
        if kwargs.get("filter_type"):
            filter_parts.append(f"type:{kwargs['filter_type']}")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        self._respect_rate_limit()

        try:
            response = self._session.get(
                self.BASE_URL,
                params=params,
                headers={"User-Agent": f"AutoPaper/0.1 (mailto:{self._POLITE_MAILTO})"},
                timeout=self._get_timeout("search"),
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            self.logger.error("CrossRef 搜索超时: '%s'", query)
            return []
        except requests.exceptions.HTTPError as exc:
            self.logger.error("CrossRef HTTP 错误: %s — 响应: %s", exc, exc.response.text[:300] if exc.response else "")
            return []
        except requests.exceptions.RequestException as exc:
            self.logger.error("CrossRef 请求异常: %s", exc)
            return []
        except ValueError as exc:
            self.logger.error("CrossRef JSON 解析失败: %s", exc)
            return []

        items = data.get("message", {}).get("items", [])
        self.logger.info("CrossRef 搜索返回 %d 条 (total-results=%d)",
                         len(items),
                         data.get("message", {}).get("total-results", 0))

        raw_results = [self._parse_crossref_item(item) for item in items]

        # 客户端年份过滤
        min_year = kwargs.get("min_year")
        if min_year:
            raw_results = [
                r for r in raw_results
                if r.get("year") and int(r["year"]) >= int(min_year)
            ]

        return self.normalize_results(raw_results)

    def _parse_crossref_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """解析单个 CrossRef 条目。"""
        # 标题 — 可能为列表
        title_raw = item.get("title", [""])
        title = title_raw[0] if isinstance(title_raw, list) and title_raw else str(title_raw)

        # 作者
        authors_raw = item.get("author", [])
        authors = []
        for a in authors_raw:
            given = a.get("given", "")
            family = a.get("family", "")
            full = f"{given} {family}".strip()
            if full:
                authors.append(full)

        # 年份
        date_parts = item.get("published-print", {}).get("date-parts", [[None]])[0]
        if not date_parts or not date_parts[0]:
            date_parts = item.get("published-online", {}).get("date-parts", [[None]])[0]
        if not date_parts or not date_parts[0]:
            date_parts = item.get("issued", {}).get("date-parts", [[None]])[0]
        year = str(date_parts[0]) if date_parts and date_parts[0] else None

        # DOI
        doi = item.get("DOI")

        # URL — 优先用 DOI 构建永久链接
        url = f"https://doi.org/{doi}" if doi else item.get("URL", "")

        # PDF 链接
        pdf_url = None
        links = item.get("link", [])
        for link in links:
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break
        if not pdf_url and links:
            # 尝试第一个链接
            pdf_url = links[0].get("URL")

        # 引用计数
        citation_count = item.get("is-referenced-by-count")

        # 期刊
        journal = None
        container = item.get("container-title", [])
        if isinstance(container, list) and container:
            journal = container[0]
        elif isinstance(container, str):
            journal = container

        return {
            "title":          title,
            "authors":        authors,
            "year":           year,
            "abstract":       item.get("abstract"),
            "doi":            doi,
            "arxiv_id":       None,
            "pdf_url":        pdf_url,
            "url":            url,
            "citation_count": citation_count,
            "journal":        journal,
        }

    # ── 获取单篇信息 ──────────────────────────────────────────────

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """根据 DOI 获取论文详细信息。

        Args:
            identifier: DOI（如 "10.1038/nature14539"）。
            **kwargs:   未使用。

        Returns:
            标准化论文信息，未找到返回 None。
        """
        # 清理 DOI 前缀
        doi = identifier.strip()
        doi = re.sub(r"^https?://doi\.org/", "", doi)

        self.logger.info("获取 CrossRef 论文信息: %s", doi)
        self._respect_rate_limit()

        url = f"{self.BASE_URL}/{quote(doi, safe='')}"
        try:
            response = self._session.get(
                url,
                headers={"User-Agent": f"AutoPaper/0.1 (mailto:{self._POLITE_MAILTO})"},
                timeout=self._get_timeout("search"),
            )
            response.raise_for_status()
            item = response.json().get("message", {})
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self.logger.warning("CrossRef 未找到论文: %s", doi)
            else:
                self.logger.error("CrossRef HTTP 错误 (%s): %s", doi, exc)
            return None
        except requests.exceptions.RequestException as exc:
            self.logger.error("CrossRef 请求异常 (%s): %s", doi, exc)
            return None
        except ValueError as exc:
            self.logger.error("CrossRef JSON 解析失败 (%s): %s", doi, exc)
            return None

        if not item:
            return None

        paper = self._parse_crossref_item(item)
        return self._normalize_single(paper, self.ENGINE_NAME)

    # ── 工具 ──────────────────────────────────────────────────────

    def _get_timeout(self, key: str) -> int:
        """从配置获取超时值。"""
        return int(self.config.get("timeout", {}).get(key, 30))

    def _respect_rate_limit(self) -> None:
        """根据配置延迟以遵守 API 速率限制。"""
        delay = float(self.config.get("concurrency", {}).get("request_delay", 1.0))
        if delay > 0:
            time.sleep(delay)

    def close(self) -> None:
        """关闭 HTTP 会话。"""
        self._session.close()
