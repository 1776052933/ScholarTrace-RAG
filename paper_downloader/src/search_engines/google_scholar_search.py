"""
google_scholar_search.py — Google Scholar 搜索引擎

基于 ``scholarly`` 库实现 Google Scholar 检索，
包含反爬处理（代理旋转、随机延迟、User-Agent 轮换）。
"""

import logging
import random
import re
import time
from typing import Any, Dict, List, Optional

try:
    from scholarly import scholarly, ProxyGenerator
    SCHOLARLY_AVAILABLE = True
except ImportError:
    scholarly = None  # type: ignore[assignment]
    ProxyGenerator = None  # type: ignore[assignment]
    SCHOLARLY_AVAILABLE = False

from paper_downloader.src.search_engines.base_search import SearchEngine

# ── User-Agent 池 ────────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
]


class GoogleScholarSearch(SearchEngine):
    """Google Scholar 搜索引擎。

    使用 ``scholarly`` 库爬取 Google Scholar 搜索结果。
    自动处理反爬机制：代理轮换、随机延迟、UA 切换。

    注意:
        Google Scholar 有严格的反爬机制。如遇到验证码或 429，
        请配置有效的代理或使用 scholarly 的免费代理池。

    Usage::

        engine = GoogleScholarSearch(config={
            "proxy": {"enabled": True, "http": "http://proxy:8080"},
            "concurrency": {"request_delay": 10.0},
        })
        results = engine.search("deep learning", max_results=5)
    """

    ENGINE_NAME = "google_scholar"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        if not SCHOLARLY_AVAILABLE:
            self.logger.warning("scholarly 未安装，GoogleScholarSearch 不可用。请运行: pip install scholarly")
            self._available = False
            return

        self._available = True
        self._setup_scholarly()

    def _setup_scholarly(self) -> None:
        """配置 scholarly 的代理和会话参数。"""
        proxy_cfg = self.config.get("proxy", {})

        if proxy_cfg.get("enabled", False):
            pg = ProxyGenerator()
            http_proxy = proxy_cfg.get("http", "")
            https_proxy = proxy_cfg.get("https", "")

            proxy_url = https_proxy or http_proxy
            if proxy_url:
                user = proxy_cfg.get("username", "")
                pwd = proxy_cfg.get("password", "")
                if user and pwd:
                    if "://" in proxy_url:
                        scheme, rest = proxy_url.split("://", 1)
                        proxy_url = f"{scheme}://{user}:{pwd}@{rest}"

                success = pg.SingleProxy(http=proxy_url, https=proxy_url)
                if success:
                    scholarly.use_proxy(pg)
                    self.logger.info("已配置 Google Scholar 代理: %s", proxy_url.split("@")[-1])
                else:
                    self.logger.warning("代理配置失败，使用直连")
            else:
                # 使用 scholarly 内置的免费代理池
                try:
                    pg = ProxyGenerator()
                    pg.FreeProxies()
                    scholarly.use_proxy(pg)
                    self.logger.info("已启用 scholarly 免费代理池")
                except Exception as exc:
                    self.logger.warning("免费代理池加载失败: %s", exc)

    # ── 搜索 ──────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        """在 Google Scholar 中搜索论文。

        Args:
            query:       搜索关键词或标题。
            max_results: 最大返回数量（注意：过大易触发验证码）。
            **kwargs:
                min_year:  最早发表年份。
                patente:   是否包含专利，默认 False。
                citations: 是否包含引用，默认 True。

        Returns:
            标准化论文信息列表。
        """
        if not self._available:
            self.logger.error("GoogleScholarSearch 不可用（scholarly 未安装）")
            return []

        query = self._clean_query(query)
        self.logger.info("Google Scholar 搜索: '%s' (max_results=%d)", query, max_results)

        # 限制单次搜索最多 10 条以降低被封风险
        # 可通过多次查询获取更多结果
        effective_max = min(max_results, kwargs.get("_batch_size", 10))

        self._respect_rate_limit()

        try:
            # 使用 scholarly.search_pubs 进行搜索
            search_query = scholarly.search_pubs(
                query,
                patents=kwargs.get("patents", False),
                citations=kwargs.get("citations", True),
            )

            raw_results = []
            for i, pub in enumerate(search_query):
                if i >= effective_max:
                    break
                try:
                    # 逐条填充详细信息（会发起额外请求）
                    pub = scholarly.fill(pub, sections=["bib", "pub_url"])
                    parsed = self._parse_scholarly_pub(pub)
                    raw_results.append(parsed)
                    self.logger.debug("GS 结果 %d: %s", i + 1, parsed.get("title", "")[:60])
                except Exception as exc:
                    self.logger.warning("解析 GS 结果 %d 失败: %s", i + 1, exc)
                    continue

                # 每条结果之间增加延迟
                time.sleep(random.uniform(2.0, 5.0))

        except StopIteration:
            self.logger.info("Google Scholar 搜索完成（结果已耗尽）")
        except Exception as exc:
            error_msg = str(exc).lower()
            if "captcha" in error_msg or "429" in error_msg or "too many" in error_msg:
                self.logger.error(
                    "Google Scholar 触发反爬机制。请: "
                    "1) 增大 request_delay 2) 配置有效代理 3) 减少 max_results"
                )
            else:
                self.logger.error("Google Scholar 搜索异常: %s", exc, exc_info=True)
            # 返回已收集的结果，不中断
            if raw_results:
                self.logger.info("返回异常前已收集的 %d 条结果", len(raw_results))

        self.logger.info("Google Scholar 搜索完成，共收集 %d 条", len(raw_results))

        # 年份过滤
        min_year = kwargs.get("min_year")
        if min_year and raw_results:
            raw_results = [
                r for r in raw_results
                if r.get("year") and int(r["year"]) >= int(min_year)
            ]

        return self.normalize_results(raw_results)

    def _parse_scholarly_pub(self, pub: Any) -> Dict[str, Any]:
        """解析 scholarly 返回的 Publication 对象。

        Args:
            pub: scholarly 返回的填充后的 Publication 对象。

        Returns:
            中间格式的论文 dict。
        """
        bib = getattr(pub, "bib", {}) or {}

        # 标题
        title = bib.get("title", "")

        # 作者
        author_str = bib.get("author", "") or ""
        authors: List[str] = []
        if author_str:
            authors = [a.strip() for a in author_str.split(" and ")]

        # 年份
        year_raw = bib.get("pub_year") or ""
        year = str(year_raw) if year_raw else None

        # 摘要
        abstract = bib.get("abstract", "")

        # DOI
        doi = bib.get("doi") or None

        # arXiv ID（如果在摘要或 URL 中出现）
        arxiv_id = self._extract_arxiv_from_text(abstract or "")
        if not arxiv_id:
            eprint_url = getattr(pub, "eprint_url", "") or bib.get("eprint", "")
            arxiv_id = self._extract_arxiv_from_text(str(eprint_url))

        # PDF URL
        pdf_url = getattr(pub, "eprint_url", "") or None

        # URL
        url = getattr(pub, "pub_url", "") or bib.get("url", "") or ""

        # 引用计数
        citation_count = bib.get("num_citations") or None
        if citation_count is not None:
            try:
                citation_count = int(citation_count)
            except (ValueError, TypeError):
                citation_count = None

        # 期刊
        journal = bib.get("journal", "") or bib.get("venue", "") or None

        return {
            "title":          title,
            "authors":        authors,
            "year":           year,
            "abstract":       abstract,
            "doi":            doi,
            "arxiv_id":       arxiv_id,
            "pdf_url":        pdf_url,
            "url":            url,
            "citation_count": citation_count,
            "journal":        journal,
        }

    # ── 获取单篇信息 ──────────────────────────────────────────────

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """根据论文标题或 URL 获取 Google Scholar 详细信息。

        Args:
            identifier: 论文标题或 Google Scholar URL。
            **kwargs:   未使用。

        Returns:
            标准化论文信息，未找到返回 None。

        注意:
            此方法通过搜索实现，效率较低。
        """
        if not self._available:
            return None

        # 检查是否为 URL
        if identifier.startswith("http"):
            self.logger.info("通过 URL 获取 GS 论文信息: %s", identifier[:80])
            self._respect_rate_limit()
            try:
                pub = scholarly.search_pubs_url(identifier)
                pub = scholarly.fill(pub, sections=["bib", "pub_url"])
                paper = self._parse_scholarly_pub(pub)
                return self._normalize_single(paper, self.ENGINE_NAME)
            except Exception as exc:
                self.logger.error("通过 URL 获取 GS 信息失败: %s", exc)
                return None

        # 按标题搜索
        self.logger.info("按标题搜索 GS 论文信息: %s", identifier[:80])
        results = self.search(identifier, max_results=1, **kwargs)
        return results[0] if results else None

    # ── 工具 ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_arxiv_from_text(text: str) -> Optional[str]:
        """从文本中提取 arXiv ID。"""
        if not text:
            return None
        pattern = re.compile(r"(?:arxiv\.org/abs/|arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
        match = pattern.search(text)
        return match.group(1) if match else None

    def _respect_rate_limit(self) -> None:
        """随机延迟以模拟人类行为，降低封禁风险。

        Google Scholar 建议每次请求间隔 ≥10 秒。
        """
        base_delay = float(self.config.get("concurrency", {}).get("request_delay", 10.0))
        # 添加 ±30% 随机抖动
        jitter = base_delay * random.uniform(-0.3, 0.3)
        delay = max(base_delay + jitter, 5.0)
        self.logger.debug("GS 延迟 %.1f 秒", delay)
        time.sleep(delay)

    @staticmethod
    def _get_random_ua() -> str:
        """返回随机 User-Agent 字符串。"""
        return random.choice(_USER_AGENTS)
