"""
arxiv_search.py — arXiv 搜索引擎

基于 ``arxiv`` 官方 Python 客户端，提供论文搜索与信息获取。
"""

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    arxiv = None  # type: ignore[assignment]
    ARXIV_AVAILABLE = False

from paper_downloader.src.search_engines.base_search import SearchEngine


class ArxivSearch(SearchEngine):
    """arXiv 搜索引擎。

    使用 arXiv 官方 API 进行论文检索，支持按标题/关键词搜索
    和按 arXiv ID 获取详细信息。

    Usage::

        engine = ArxivSearch(config={"max_results": 10})
        results = engine.search("attention is all you need")
        info = engine.get_paper_info("1706.03762")
    """

    ENGINE_NAME = "arxiv"
    API_URL = "https://export.arxiv.org/api/query"
    _HTTP_LOCK = threading.Lock()
    _last_http_request = 0.0

    # arXiv ID 格式: 四位年份 + 五位数，可选版本号，或旧格式
    _ARXIV_ID_PATTERN = re.compile(
        r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[\w\-]+/\d{7}(?:v\d+)?)",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # arxiv 包是优化路径而非硬依赖；缺失时使用官方 Atom HTTP API。
        self._available = True
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "AutoPaper/0.2 (academic paper downloader; mailto:auto-paper@example.com)",
        })
        if not ARXIV_AVAILABLE:
            self.logger.info("arxiv 包未安装，将使用 arXiv Atom HTTP API")

    # ── 搜索 ──────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20, **kwargs: Any) -> List[Dict[str, Any]]:
        """在 arXiv 中按标题/关键词搜索论文。

        Args:
            query:       搜索关键词。
            max_results: 最大返回数量。
            **kwargs:
                sort_by:  排序方式 (relevance / lastUpdatedDate / submittedDate)，默认 relevance。
                min_year: 最早发表年份（用于客户端过滤）。

        Returns:
            标准化论文信息列表。
        """
        query = self._clean_query(query)
        self.logger.info("arXiv 搜索: '%s' (max_results=%d)", query, max_results)

        # 延迟以尊重速率限制
        self._respect_rate_limit()

        sort_choice = kwargs.get("sort_by", "relevance")

        if arxiv is None:
            raw_results = self._search_via_http(query, max_results, sort_choice)
        else:
            sort_map = {
                "relevance":        arxiv.SortCriterion.Relevance,
                "lastUpdatedDate":  arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate":    arxiv.SortCriterion.SubmittedDate,
            }
            sort_by = sort_map.get(sort_choice, arxiv.SortCriterion.Relevance)
            try:
                client = arxiv.Client()
                search_query = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=sort_by,
                )
                raw_results = []
                for result in client.results(search_query):
                    raw_results.append(self._parse_arxiv_result(result))

            except arxiv.UnexpectedEmptyPageError:
                self.logger.warning("arXiv API 返回空页，可能已达结果末尾")
                raw_results = []
            except arxiv.HTTPError as exc:
                self.logger.error("arXiv HTTP 错误: %s", exc)
                return []
            except Exception as exc:
                self.logger.error("arXiv 搜索异常: %s", exc, exc_info=True)
                return []

        self.logger.info("arXiv 搜索原始命中 %d 条", len(raw_results))

        # 客户端过滤年份（arXiv API 不支持直接按年份过滤）
        min_year = kwargs.get("min_year")
        if min_year:
            raw_results = [
                r for r in raw_results
                if r.get("year") and int(r["year"]) >= int(min_year)
            ]
            self.logger.info("年份过滤 (>=%s) 后剩余 %d 条", min_year, len(raw_results))

        return self.normalize_results(raw_results)

    def _search_via_http(
        self,
        query: str,
        max_results: int,
        sort_by: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """不依赖第三方 arxiv 包，直接调用官方 Atom API。"""
        sort_map = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }
        params = {
            "search_query": f'ti:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_map.get(sort_by, "relevance"),
            "sortOrder": "descending",
        }
        try:
            response = self._http_get(params)
            response.raise_for_status()
            return self._parse_atom_feed(response.content)
        except (requests.RequestException, ET.ParseError) as exc:
            self.logger.error("arXiv Atom API 请求失败: %s", exc)
            return []

    @staticmethod
    def _parse_atom_feed(content: bytes) -> List[Dict[str, Any]]:
        """把 arXiv Atom feed 转为搜索引擎的中间字典格式。"""
        root = ET.fromstring(content)
        atom = "{http://www.w3.org/2005/Atom}"
        arxiv_ns = "{http://arxiv.org/schemas/atom}"
        results: List[Dict[str, Any]] = []

        for entry in root.findall(f"{atom}entry"):
            entry_id = (entry.findtext(f"{atom}id") or "").strip()
            arxiv_id = ArxivSearch._extract_arxiv_id(entry_id)
            pdf_url = ""
            for link in entry.findall(f"{atom}link"):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            published = (entry.findtext(f"{atom}published") or "").strip()
            results.append({
                "title": " ".join((entry.findtext(f"{atom}title") or "").split()),
                "authors": [
                    (author.findtext(f"{atom}name") or "").strip()
                    for author in entry.findall(f"{atom}author")
                    if (author.findtext(f"{atom}name") or "").strip()
                ],
                "year": published[:4] or None,
                "abstract": " ".join((entry.findtext(f"{atom}summary") or "").split()),
                "doi": entry.findtext(f"{arxiv_ns}doi"),
                "arxiv_id": arxiv_id,
                "pdf_url": pdf_url,
                "url": entry_id,
                "citation_count": None,
                "journal": entry.findtext(f"{arxiv_ns}journal_ref"),
            })
        return results

    def _parse_arxiv_result(self, result: Any) -> Dict[str, Any]:
        """解析单个 arXiv 搜索结果，转为中间格式。"""
        # 提取作者姓名字符串列表
        authors = [a.name for a in result.authors] if result.authors else []

        # 提取 DOI（arXiv 结果中的 doi 可能为 None）
        doi = result.doi if hasattr(result, "doi") and result.doi else None

        # arXiv ID — 从 entry_id URL 中提取
        arxiv_id = self._extract_arxiv_id(result.entry_id) if result.entry_id else None

        # 年份
        year = str(result.published.year) if result.published else None

        # 期刊引用
        journal = result.journal_ref if hasattr(result, "journal_ref") and result.journal_ref else None

        # 注释中有时包含 DOI 或更多信息
        comment = result.comment if hasattr(result, "comment") and result.comment else ""

        # 尝试从 comment 中提取 DOI（如果主字段没有的话）
        if not doi and comment:
            doi_match = re.search(r"doi[:\s]*(\S+)", comment, re.IGNORECASE)
            if doi_match:
                doi = doi_match.group(1).rstrip(".,;")

        return {
            "title":          result.title or "",
            "authors":        authors,
            "year":           year,
            "abstract":       result.summary or "",
            "doi":            doi,
            "arxiv_id":       arxiv_id,
            "pdf_url":        result.pdf_url or "",
            "url":            result.entry_id or "",
            "citation_count": None,  # arXiv 不提供引用计数
            "journal":        journal,
        }

    # ── 获取单篇信息 ──────────────────────────────────────────────

    def get_paper_info(self, identifier: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """根据 arXiv ID 获取论文详细信息。

        Args:
            identifier: arXiv ID (如 "1706.03762" 或 "arxiv:1706.03762")。
            **kwargs:   未使用。

        Returns:
            标准化论文信息，未找到返回 None。
        """
        arxiv_id = self._extract_arxiv_id(identifier)
        if not arxiv_id:
            self.logger.warning("无效的 arXiv 标识符: %s", identifier)
            return None

        self.logger.info("获取 arXiv 论文信息: %s", arxiv_id)
        self._respect_rate_limit()

        if arxiv is None:
            try:
                response = self._http_get({"id_list": arxiv_id, "max_results": 1})
                response.raise_for_status()
                results = self._parse_atom_feed(response.content)
                return self._normalize_single(results[0], self.ENGINE_NAME) if results else None
            except (requests.RequestException, ET.ParseError) as exc:
                self.logger.error("获取 arXiv 论文异常 (%s): %s", arxiv_id, exc)
                return None

        try:
            client = arxiv.Client()
            search_query = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search_query))
            paper = self._parse_arxiv_result(result)
            return self._normalize_single(paper, self.ENGINE_NAME)
        except StopIteration:
            self.logger.warning("arXiv 未找到论文: %s", arxiv_id)
            return None
        except arxiv.HTTPError as exc:
            self.logger.error("arXiv HTTP 错误 (%s): %s", arxiv_id, exc)
            return None
        except Exception as exc:
            self.logger.error("获取 arXiv 论文异常 (%s): %s", arxiv_id, exc)
            return None

    # ── ID 提取 ───────────────────────────────────────────────────

    @classmethod
    def _extract_arxiv_id(cls, text: str) -> Optional[str]:
        """从文本中提取 arXiv ID。

        Args:
            text: 可能包含 arXiv ID 的文本或 URL。

        Returns:
            提取到的 arXiv ID，失败返回 None。
        """
        if not text:
            return None
        match = cls._ARXIV_ID_PATTERN.search(text)
        return match.group(1) if match else None

    # ── 速率限制 ──────────────────────────────────────────────────

    def _respect_rate_limit(self) -> None:
        """根据配置延迟以遵守 arXiv API 限制（建议 ≥3 秒）。"""
        # 官方 Python 客户端自行处理节流；HTTP 回退由 _http_get 串行节流。
        if arxiv is None:
            return
        delay = float(self.config.get("concurrency", {}).get("request_delay", 3.0))
        # arXiv 官方建议至少 3 秒间隔
        delay = max(delay, 3.0)
        time.sleep(delay)

    def _http_get(self, params: Dict[str, Any]) -> requests.Response:
        """串行访问 arXiv API，确保任意线程间至少间隔配置的秒数。"""
        delay = max(float(self.config.get("concurrency", {}).get("request_delay", 3.0)), 3.0)
        with self._HTTP_LOCK:
            elapsed = time.monotonic() - self.__class__._last_http_request
            if elapsed < delay:
                time.sleep(delay - elapsed)
            response = self._session.get(
                self.API_URL,
                params=params,
                timeout=int(self.config.get("timeout", {}).get("search", 30)),
            )
            self.__class__._last_http_request = time.monotonic()
            return response
