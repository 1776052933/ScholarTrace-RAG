"""
search_factory.py — 搜索引擎工厂

根据配置动态选择搜索引擎，支持多引擎并行搜索、结果合并与去重。
可选集成 CacheManager 实现搜索结果透明缓存。
"""

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Type

from paper_downloader.src.search_engines.base_search import SearchEngine
from paper_downloader.src.search_engines.arxiv_search import ArxivSearch
from paper_downloader.src.search_engines.crossref_search import CrossrefSearch
from paper_downloader.src.search_engines.google_scholar_search import GoogleScholarSearch
from paper_downloader.src.search_engines.openalex_search import OpenAlexSearch
from paper_downloader.src.search_engines.semantic_scholar_search import SemanticScholarSearch

# 尝试导入 rapidfuzz（项目已有依赖），不可用时回退到简单字符串匹配
try:
    from rapidfuzz import fuzz
    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False


# ── 引擎注册表 ───────────────────────────────────────────────────

_ENGINE_REGISTRY: Dict[str, Type[SearchEngine]] = {
    "arxiv":           ArxivSearch,
    "crossref":        CrossrefSearch,
    "google_scholar":  GoogleScholarSearch,
    "google scholar":  GoogleScholarSearch,
    "openalex":         OpenAlexSearch,
    "semantic_scholar": SemanticScholarSearch,
    "semantic scholar": SemanticScholarSearch,
}

# 引擎别名
_ENGINE_ALIASES = {
    "gs": "google_scholar",
    "gsr": "google_scholar",
    "s2": "semantic_scholar",
}


class SearchFactory:
    """搜索引擎工厂。

    根据配置实例化一个或多个搜索引擎，提供并行搜索与结果合并去重。

    Usage::

        factory = SearchFactory(config)
        results = factory.search_all("attention is all you need", max_results=10)

        # 指定引擎
        results = factory.search_all("transformers", engines=["arxiv", "crossref"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工厂。

        Args:
            config: 配置字典，至少包含 ``search.engines`` 列表。
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # 从配置读取要启用的引擎
        engine_names = self.config.get("search", {}).get("engines", ["arxiv"])
        engine_names = [self._resolve_alias(name) for name in engine_names]

        self.engines: Dict[str, SearchEngine] = {}
        self._cache_manager: Optional[Any] = None  # type: ignore[assignment]
        self._cache_enabled = bool(self.config.get("cache", {}).get("enabled", False))
        self._init_engines(engine_names)
        self.logger.info("SearchFactory 初始化完成，已加载引擎: %s", list(self.engines.keys()))

    def _resolve_alias(self, name: str) -> str:
        """解析引擎别名。"""
        return _ENGINE_ALIASES.get(name.lower(), name.lower())

    def _init_engines(self, engine_names: List[str]) -> None:
        """根据名称列表实例化搜索引擎。"""
        for name in engine_names:
            engine_cls = _ENGINE_REGISTRY.get(name)
            if engine_cls is None:
                self.logger.warning("未知的搜索引擎 '%s'，已跳过。可用: %s",
                                    name, list(_ENGINE_REGISTRY.keys()))
                continue
            try:
                self.engines[name] = engine_cls(self.config)
            except Exception as exc:
                self.logger.error("实例化 '%s' 引擎失败: %s", name, exc)

    # ── 多引擎搜索 ────────────────────────────────────────────────

    def search_all(
        self,
        query: str,
        max_results: int = 20,
        engines: Optional[List[str]] = None,
        deduplicate: bool = True,
        sort_by: str = "relevance",
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """并行搜索所有引擎，合并并去重结果。

        Args:
            query:       搜索查询字符串。
            max_results: 每个引擎的最大返回数量。
            engines:     指定使用的引擎名称列表。
                         None 表示使用所有已初始化的引擎。
            deduplicate: 是否对跨引擎结果去重。默认 True。
            sort_by:     排序方式 (relevance / citations / year)。
            **kwargs:    传递给各引擎 search() 的额外参数。

        Returns:
            合并去重后的标准化论文列表。
        """
        target_engines = self._resolve_engines(engines)
        if not target_engines:
            self.logger.error("没有可用的搜索引擎")
            return []

        self.logger.info("并行搜索 %d 个引擎: %s", len(target_engines), list(target_engines.keys()))
        self.logger.info("查询: '%s', 每引擎 max_results=%d", query, max_results)

        # 并行搜索
        # 查缓存
        cache_key: Optional[str] = None
        if self._cache_enabled and self._cache_manager is not None:
            cache_key = self._make_cache_key(query, max_results, engines, sort_by, **kwargs)
            cached = self._cache_manager.get(cache_key)
            if cached is not None:
                self.logger.info("缓存命中: '%s' (%d 条)", query, len(cached))
                return cached

        all_results = self._parallel_search(target_engines, query, max_results, **kwargs)

        self.logger.info("原始合并结果: %d 条", len(all_results))

        # 去重
        if deduplicate and len(all_results) > 1:
            before = len(all_results)
            all_results = self._deduplicate(all_results)
            self.logger.info("去重: %d → %d 条 (移除 %d 条重复)", before, len(all_results), before - len(all_results))

        # 排序
        all_results = self._sort_results(all_results, sort_by)

        # 截断
        total_max = self.config.get("search", {}).get("max_results", max_results)
        if len(all_results) > total_max:
            self.logger.info("截断至 %d 条", total_max)
            all_results = all_results[:total_max]

        # 写入缓存
        if cache_key and self._cache_manager is not None and all_results:
            ttl = int(self.config.get("cache", {}).get("ttl", 86400))
            self._cache_manager.set(cache_key, all_results, ttl=ttl)

        return all_results

    def set_cache(self, cache_manager: Any) -> None:
        """注入缓存管理器，启用搜索结果缓存。

        设置后 search_all() 会先查缓存再搜索，搜索结果自动缓存。

        Args:
            cache_manager: CacheManager 实例，None 禁用缓存。
        """
        self._cache_manager = cache_manager
        self._cache_enabled = cache_manager is not None
        if cache_manager:
            self.logger.info("搜索结果缓存已启用")

    def enable_cache(self) -> None:
        """启用缓存（需要在 set_cache 之后调用）。"""
        self._cache_enabled = self._cache_manager is not None

    def disable_cache(self) -> None:
        """临时禁用缓存（不清除缓存管理器）。"""
        self._cache_enabled = False

    @staticmethod
    def _make_cache_key(query: str, max_results: int,
                        engines: Optional[List[str]], sort_by: str,
                        **kwargs: Any) -> str:
        """生成搜索缓存键。"""
        payload = {
            "q": query.strip().lower(),
            "n": max_results,
            "e": sorted(engines) if engines else "all",
            "s": sort_by,
        }
        if kwargs:
            payload["kw"] = {k: v for k, v in sorted(kwargs.items())}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        h = hashlib.md5(raw.encode()).hexdigest()[:16]
        return f"search:{h}"

    def _resolve_engines(
        self, engine_names: Optional[List[str]] = None,
    ) -> Dict[str, SearchEngine]:
        """解析要使用的引擎。"""
        if engine_names is None:
            return self.engines

        resolved = {}
        for name in engine_names:
            name = self._resolve_alias(name)
            engine = self.engines.get(name)
            if engine:
                resolved[name] = engine
            else:
                self.logger.warning("请求的引擎 '%s' 未初始化或不可用", name)
        return resolved

    def _parallel_search(
        self,
        engines: Dict[str, SearchEngine],
        query: str,
        max_results: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """使用线程池并行调用多个引擎的 search()。

        Args:
            engines:     参与搜索的引擎字典。
            query:       搜索查询。
            max_results: 每个引擎的最大返回数。
            **kwargs:    引擎额外参数。

        Returns:
            所有引擎结果的合并列表。
        """
        all_results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            futures = {}
            for name, engine in engines.items():
                future = executor.submit(engine.search, query, max_results, **kwargs)
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        self.logger.info("引擎 '%s' 返回 %d 条结果", name, len(results))
                    else:
                        self.logger.warning("引擎 '%s' 未返回结果", name)
                except Exception as exc:
                    self.logger.error("引擎 '%s' 搜索异常: %s", name, exc)

        return all_results

    # ── 去重逻辑 ──────────────────────────────────────────────────

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对跨引擎结果进行去重合并。

        策略:
            1. 按 DOI 精确去重（最高优先级）。
            2. 按规范化标题模糊去重（相似度 > 阈值）。
            3. 保留信息更完整的结果。

        Args:
            results: 待去重的论文列表。

        Returns:
            去重后的论文列表。
        """
        if not results:
            return results

        deduped: List[Dict[str, Any]] = []
        for paper in results:
            matched_idx = self._find_duplicate(paper, deduped)
            if matched_idx is not None:
                # 合并：用更完整的字段覆盖
                deduped[matched_idx] = self._merge_papers(deduped[matched_idx], paper)
            else:
                deduped.append(paper)

        return deduped

    def _find_duplicate(
        self, paper: Dict[str, Any], existing: List[Dict[str, Any]],
    ) -> Optional[int]:
        """在已有结果中查找与 paper 相同的论文。

        Args:
            paper:    待检查的论文。
            existing: 已去重的论文列表。

        Returns:
            匹配到的索引，未找到返回 None。
        """
        paper_doi = paper.get("doi")
        paper_title = self._norm_title(paper.get("title", ""))

        for i, ex in enumerate(existing):
            # 1) DOI 精确匹配
            ex_doi = ex.get("doi")
            if paper_doi and ex_doi and paper_doi.lower() == ex_doi.lower():
                return i

            # 2) arXiv ID 匹配
            paper_arxiv = paper.get("arxiv_id")
            ex_arxiv = ex.get("arxiv_id")
            if paper_arxiv and ex_arxiv and paper_arxiv == ex_arxiv:
                return i

            # 3) 标题模糊匹配
            ex_title = self._norm_title(ex.get("title", ""))
            if paper_title and ex_title:
                if _HAS_FUZZ:
                    similarity = fuzz.ratio(paper_title, ex_title)
                    if similarity >= 90:
                        return i
                else:
                    # 简单回退：规范化后完全相同
                    if paper_title == ex_title:
                        return i

        return None

    @staticmethod
    def _merge_papers(
        primary: Dict[str, Any], secondary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """合并两篇识别为同一论文的记录。

        优先保留 primary 中的非空值，
        用 secondary 中的非空值填补 primary 的缺失字段。

        Args:
            primary:   主记录（保留）。
            secondary: 从记录（填补）。

        Returns:
            合并后的记录。
        """
        # 可合并的字段列表
        mergeable_scalar = [
            "title", "year", "abstract", "doi", "arxiv_id",
            "pdf_url", "url", "journal",
        ]

        for field in mergeable_scalar:
            if not primary.get(field) and secondary.get(field):
                primary[field] = secondary[field]

        # 作者：合并去重
        primary_authors = primary.get("authors", [])
        secondary_authors = secondary.get("authors", [])
        seen = {a.lower() for a in primary_authors}
        for a in secondary_authors:
            if a.lower() not in seen:
                primary_authors.append(a)
                seen.add(a.lower())
        primary["authors"] = primary_authors

        # 引用计数：取最大值
        p_count = primary.get("citation_count") or 0
        s_count = secondary.get("citation_count") or 0
        primary["citation_count"] = max(p_count, s_count)

        # 来源：合并标记
        p_source = primary.get("source", "")
        s_source = secondary.get("source", "")
        if p_source and s_source and s_source not in p_source:
            primary["source"] = f"{p_source},{s_source}"

        return primary

    # ── 排序 ──────────────────────────────────────────────────────

    def _sort_results(
        self, results: List[Dict[str, Any]], sort_by: str,
    ) -> List[Dict[str, Any]]:
        """对结果列表排序。

        Args:
            results: 论文列表。
            sort_by: 排序方式 (relevance / citations / year)。

        Returns:
            排序后的列表（降序）。
        """
        if sort_by == "citations":

            def key(p: Dict[str, Any]) -> int:
                c = p.get("citation_count")
                return int(c) if c is not None else -1

            results.sort(key=key, reverse=True)
        elif sort_by == "year":

            def key_year(p: Dict[str, Any]) -> int:
                y = p.get("year")
                try:
                    return int(y)
                except (ValueError, TypeError):
                    return 0

            results.sort(key=key_year, reverse=True)
        # relevance: 保持引擎返回的原始顺序

        return results

    # ── 标题标准化 ───────────────────────────────────────────────

    @staticmethod
    def _norm_title(title: str) -> str:
        """规范化标题用于比较。

        - 转小写
        - 移除标点和多余空格
        - 保留字母数字和空格

        Args:
            title: 原始标题。

        Returns:
            规范化后的标题字符串。
        """
        if not title:
            return ""
        norm = title.lower()
        norm = re.sub(r"[^\w\s]", "", norm)
        norm = re.sub(r"\s+", " ", norm).strip()
        return norm

    # ── 单引擎直通 ────────────────────────────────────────────────

    def get_engine(self, name: str) -> Optional[SearchEngine]:
        """获取已初始化的指定引擎。

        Args:
            name: 引擎名称 (arxiv / crossref / google_scholar)。

        Returns:
            SearchEngine 实例或 None。
        """
        name = self._resolve_alias(name)
        return self.engines.get(name)

    def search_single(
        self,
        query: str,
        engine: str,
        max_results: int = 20,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """使用单个指定引擎搜索。

        Args:
            query:       搜索查询。
            engine:      引擎名称。
            max_results: 最大返回数。
            **kwargs:    引擎特定参数。

        Returns:
            标准化论文列表。
        """
        eng = self.get_engine(engine)
        if eng is None:
            self.logger.error("引擎不可用: %s", engine)
            return []
        return eng.search(query, max_results, **kwargs)

    # ── 元信息 ────────────────────────────────────────────────────

    @property
    def available_engines(self) -> List[str]:
        """返回已加载的引擎名称列表。"""
        return list(self.engines.keys())

    @classmethod
    def list_registered_engines(cls) -> List[str]:
        """返回所有已注册的引擎名称（不考虑当前配置）。"""
        return list(_ENGINE_REGISTRY.keys())
