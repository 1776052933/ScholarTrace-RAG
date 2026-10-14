"""
paper_downloader.py — 论文自动下载器主模块

提供论文搜索与 PDF 下载的核心抽象接口。
各搜索引擎（arXiv、Google Scholar、CrossRef）需继承并实现具体逻辑。
"""

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class PaperDownloader(ABC):
    """论文自动下载器抽象基类。

    定义了搜索与下载论文的标准接口。
    具体搜索引擎实现需覆盖 ``search_by_title()`` 和 ``download_paper()``。

    Usage::

        class ArxivDownloader(PaperDownloader):
            def search_by_title(self, title, **kwargs):
                ...
            def download_paper(self, identifier, save_path=None, **kwargs):
                ...

        downloader = ArxivDownloader()
        results = downloader.main("Attention Is All You Need")
    """

    # 文件名非法字符（与 papers/paper_manager.py 保持一致）
    _ILLEGAL_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

    def __init__(self, config_path: Optional[str] = None):
        """初始化下载器。

        Args:
            config_path: YAML 配置文件路径。
                         默认为 paper_downloader/config/config.yaml。
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("%s 初始化完成", self.__class__.__name__)

    # ── 配置加载 ──────────────────────────────────────────────────

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载 YAML 配置文件。

        Args:
            config_path: 配置文件路径，默认回退到同级 config/ 下的 config.yaml。

        Returns:
            解析后的配置字典。
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config",
                "config.yaml",
            )
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """配置日志系统（同时输出到文件和控制台）。"""
        log_cfg = self.config.get("logging", {})
        log_path = log_cfg.get("file", "./logs/paper_downloader.log")

        # 确保日志目录存在
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_cfg.get("level", "INFO")),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(log_path, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    # ── 抽象接口 ──────────────────────────────────────────────────

    @abstractmethod
    def search_by_title(self, title: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """按标题搜索论文。

        Args:
            title:   论文标题或关键词。
            **kwargs: 额外搜索参数（如 max_results、year 等）。

        Returns:
            匹配的论文信息列表，每项至少包含:
            - title   (str): 论文标题
            - authors (list[str]): 作者列表
            - year    (str|int): 发表年份
            - doi     (str|None): DOI 标识符
            - url     (str): 论文在线地址
            - source  (str): 数据来源 (arxiv/google_scholar/crossref)
        """
        ...

    @abstractmethod
    def download_paper(
        self,
        identifier: str,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """下载论文 PDF 到本地。

        Args:
            identifier: 论文标识符（DOI / arXiv ID / URL）。
            save_path:  保存路径；为 None 时使用配置中的默认路径。
            **kwargs:   额外参数（如自定义文件名等）。

        Returns:
            下载成功返回文件绝对路径，失败返回 None。
        """
        ...

    # ── 通用工具 ──────────────────────────────────────────────────

    def _ensure_save_dir(self, path: Optional[str] = None) -> Path:
        """确保保存目录存在并返回其 Path 对象。

        Args:
            path: 目标目录路径，默认取自配置。

        Returns:
            已存在的目录 Path。
        """
        if path is None:
            path = self.config.get("download", {}).get("path", "./papers")
        save_dir = Path(path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _sanitize_filename(self, filename: str, max_len: int = 200) -> str:
        """清理文件名中的非法字符。

        Args:
            filename: 原始文件名。
            max_len:  最大长度限制。

        Returns:
            合法的文件名。
        """
        sanitized = re.sub(self._ILLEGAL_CHARS, "_", filename)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = re.sub(r"_(\.\w+)$", r"\1", sanitized)  # "name_.pdf" → "name.pdf"
        sanitized = sanitized.strip("_ .")
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
        return sanitized

    def _build_filename(
        self,
        title: str,
        authors: str | List[str] = "",
        year: str = "",
    ) -> str:
        """根据模板生成规范的 PDF 文件名。

        Args:
            title:   论文标题。
            authors: 作者信息。
            year:    发表年份。

        Returns:
            生成的文件名（不含扩展名）。
        """
        template = self.config.get("download", {}).get(
            "filename_template", "{first_author}_{year}_{title}"
        )

        # 提取第一作者姓氏
        first_author = "unknown"
        if authors:
            if isinstance(authors, list):
                first_name = authors[0] if authors else "unknown"
            else:
                first_name = str(authors).split(",")[0].strip()
            parts = first_name.split()
            first_author = parts[-1] if parts else first_name

        year_str = str(year) if year else "nodate"
        title_part = (title[:80] + "..") if len(title) > 80 else title

        filename = (
            template.replace("{first_author}", first_author)
            .replace("{year}", year_str)
            .replace("{title}", title_part)
        )
        return self._sanitize_filename(filename)

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """根据配置构建 requests 兼容的代理字典。

        Returns:
            {'http': ..., 'https': ...} 或 None（未启用代理时）。
        """
        proxy_cfg = self.config.get("proxy", {})
        if not proxy_cfg.get("enabled", False):
            return None

        proxies = {}
        for proto in ("http", "https"):
            url = proxy_cfg.get(proto, "")
            if url:
                # 拼接认证信息
                user = proxy_cfg.get("username", "")
                pwd = proxy_cfg.get("password", "")
                if user and pwd:
                    # 在 URL 中插入认证
                    if "://" in url:
                        scheme, rest = url.split("://", 1)
                        url = f"{scheme}://{user}:{pwd}@{rest}"
                proxies[proto] = url
        return proxies or None

    def _respect_rate_limit(self) -> None:
        """根据配置中的 request_delay 休眠，控制请求频率。"""
        delay = float(self.config.get("concurrency", {}).get("request_delay", 1.0))
        if delay > 0:
            time.sleep(delay)

    # ── 并发下载 ──────────────────────────────────────────────────

    def _download_many(
        self,
        papers: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """使用线程池并发下载多篇论文。

        Args:
            papers:   包含标识符的论文 dict 列表。
            save_dir: 保存目录。

        Returns:
            更新后的 papers 列表（新增 ``pdf_path`` 字段）。
        """
        max_workers = int(self.config.get("concurrency", {}).get("max_downloads", 3))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, paper in enumerate(papers):
                identifier = self._extract_identifier(paper)
                if not identifier:
                    self.logger.warning("论文缺少标识符，跳过: %s", paper.get("title", "N/A"))
                    continue
                future = executor.submit(self.download_paper, identifier, save_dir)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    pdf_path = future.result()
                    papers[idx]["pdf_path"] = pdf_path
                    if pdf_path:
                        self.logger.info("下载成功 [%d/%d]: %s", idx + 1, len(papers), papers[idx].get("title"))
                    else:
                        self.logger.warning("下载失败 [%d/%d]: %s", idx + 1, len(papers), papers[idx].get("title"))
                except Exception as exc:
                    self.logger.error("下载异常 [%d/%d]: %s — %s", idx + 1, len(papers), papers[idx].get("title"), exc)
                    papers[idx]["pdf_path"] = None

        return papers

    @staticmethod
    def _extract_identifier(paper: Dict[str, Any]) -> Optional[str]:
        """从论文 dict 中提取最优先的标识符。

        优先级: DOI > arXiv ID > URL
        """
        return paper.get("doi") or paper.get("arxiv_id") or paper.get("url")

    # ── 入口 ──────────────────────────────────────────────────────

    def main(
        self,
        query: str,
        download: bool = True,
        max_results: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """主入口：搜索并（可选）下载论文。

        串联 ``search_by_title`` → ``download_paper`` 的完整流水线。

        Args:
            query:       搜索关键词或论文标题。
            download:    是否自动下载搜索到的论文。默认 True。
            max_results: 限制下载篇数，None 表示不限。
            **kwargs:    传递给 ``search_by_title`` 和 ``download_paper``。

        Returns:
            论文结果列表，每项包含搜索字段 + ``pdf_path``（如有下载）。
        """
        self.logger.info("=" * 60)
        self.logger.info("开始处理查询: %s", query)

        # 1. 搜索
        papers = self.search_by_title(query, **kwargs)
        self.logger.info("搜索完成，共命中 %d 篇论文", len(papers))

        if not papers:
            self.logger.warning("未找到匹配的论文")
            return []

        # 2. 截断
        if max_results is not None and max_results < len(papers):
            papers = papers[:max_results]
            self.logger.info("截断至前 %d 篇", max_results)

        # 3. 下载
        if download:
            save_dir = kwargs.pop("save_path", None)
            papers = self._download_many(papers, save_dir)

        self.logger.info("流水线完成，共处理 %d 篇论文", len(papers))
        return papers
