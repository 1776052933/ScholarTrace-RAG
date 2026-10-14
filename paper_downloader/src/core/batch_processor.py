"""
batch_processor.py — 批量处理器

从多种文件格式读取论文标题列表，执行批量下载，
并生成下载报告（JSON / CSV）。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.core.progress_tracker import ProgressTracker
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import ValidationError


class BatchProcessor:
    """批量论文下载处理器。

    支持从 TXT / CSV / JSON / BibTeX 文件读取论文列表，
    执行批量下载并生成报告。

    Usage::

        bp = BatchProcessor(PaperDownloader())

        # 从文件加载标题
        titles = bp.from_file("titles.txt")

        # 从 CSV 文件指定列加载
        titles = bp.from_csv("papers.csv", column="title")

        # 从 BibTeX 解析
        titles = bp.from_bibtex("references.bib")

        # 执行批量下载并生成报告
        results = bp.process_batch(titles, output_dir="./downloads")
        bp.generate_report(results, output_dir="./downloads")
    """

    def __init__(
        self,
        downloader: Optional[PaperDownloader] = None,
        **kwargs: Any,
    ):
        """初始化批量处理器。

        Args:
            downloader: PaperDownloader 实例。None 则自动创建。
            **kwargs:   传递给 PaperDownloader 的构造参数。
        """
        self._downloader = downloader or PaperDownloader(**kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── 文件加载 ──────────────────────────────────────────────────

    def from_file(self, file_path: str) -> List[str]:
        """从文件读取论文标题列表。

        自动根据扩展名分发:
            - .txt  → 每行一个标题
            - .csv  → 尝试自动检测标题列
            - .json → 解析 JSON 数组
            - .bib  → BibTeX 解析

        Args:
            file_path: 文件路径。

        Returns:
            标题字符串列表。

        Raises:
            ValidationError: 文件不存在或格式不支持。
        """
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"文件不存在: {file_path}", file_path=str(path))

        ext = path.suffix.lower()
        if ext in (".txt", ".text", ""):
            return self.from_txt(file_path)
        elif ext == ".csv":
            return self.from_csv(file_path)
        elif ext == ".json":
            return self.from_json(file_path)
        elif ext == ".bib":
            papers = self.from_bibtex(file_path)
            return [p.title for p in papers if p.title]
        else:
            raise ValidationError(
                f"不支持的文件格式: {ext}",
                file_path=str(path),
                reason=f"supported: .txt, .csv, .json, .bib",
            )

    def from_txt(self, file_path: str) -> List[str]:
        """从 TXT 文件读取标题（每行一个，忽略空行和 # 注释）。

        Args:
            file_path: TXT 文件路径。

        Returns:
            标题列表。
        """
        path = Path(file_path)
        titles: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    titles.append(line)
        self.logger.info("从 TXT 加载 %d 个标题: %s", len(titles), path.name)
        return titles

    def from_csv(self, file_path: str, column: Optional[str] = None) -> List[str]:
        """从 CSV 文件的指定列读取标题。

        Args:
            file_path: CSV 文件路径。
            column:    列名。None 时自动检测（优先
                       title / Title / paper_title / name 列）。

        Returns:
            标题列表。
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValidationError("CSV 文件没有表头", file_path=str(path))

            if column is None:
                column = self._auto_detect_title_column(reader.fieldnames)

            if column not in reader.fieldnames:
                raise ValidationError(
                    f"CSV 缺少列: '{column}'",
                    file_path=str(path),
                    reason=f"available columns: {reader.fieldnames}",
                )

            titles = [row[column].strip() for row in reader if row.get(column, "").strip()]

        self.logger.info("从 CSV 加载 %d 个标题 (列 '%s'): %s", len(titles), column, path.name)
        return titles

    @staticmethod
    def _auto_detect_title_column(fieldnames: List[str]) -> str:
        """自动检测标题列名。"""
        candidates = ["title", "Title", "TITLE", "paper_title", "name", "paper"]
        for c in candidates:
            if c in fieldnames:
                return c
        # 回退：包含 "title" 的第一列
        for fn in fieldnames:
            if "title" in fn.lower():
                return fn
        # 最后一招：第一列
        return fieldnames[0]

    def from_json(self, file_path: str) -> List[str]:
        """从 JSON 文件读取标题。

        支持格式:
            - 字符串数组: ["title1", "title2"]
            - 对象数组:   [{"title": "..."}, ...]

        Args:
            file_path: JSON 文件路径。

        Returns:
            标题列表。
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        titles: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    titles.append(item.strip())
                elif isinstance(item, dict):
                    t = item.get("title") or item.get("Title") or item.get("paper_title") or ""
                    if t.strip():
                        titles.append(t.strip())
        elif isinstance(data, dict):
            # 可能是 {"papers": [{"title": ...}]} 格式
            items = data.get("papers") or data.get("items") or []
            for item in items:
                if isinstance(item, dict):
                    t = item.get("title", "")
                    if t.strip():
                        titles.append(t.strip())

        self.logger.info("从 JSON 加载 %d 个标题: %s", len(titles), path.name)
        return titles

    def from_bibtex(self, file_path: str) -> List[Paper]:
        """从 BibTeX 文件解析论文信息。

        解析 @article 和 @inproceedings 条目，
        提取 title / author / year / doi / journal 字段。

        Args:
            file_path: BibTeX 文件路径。

        Returns:
            解析出的 Paper 对象列表。
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        papers = self._parse_bibtex_entries(content)
        self.logger.info("从 BibTeX 解析 %d 篇论文: %s", len(papers), path.name)
        return papers

    @staticmethod
    def _parse_bibtex_entries(content: str) -> List[Paper]:
        """解析 BibTeX 条目字符串。

        Args:
            content: BibTeX 文件内容。

        Returns:
            Paper 列表。
        """
        papers: List[Paper] = []

        # 匹配每个 @type{key, fields} 条目
        entry_pattern = re.compile(
            r"@(\w+)\s*\{\s*([^,]*),\s*(.*?)\}\s*$",
            re.MULTILINE | re.DOTALL,
        )

        for match in entry_pattern.finditer(content):
            entry_type = match.group(1).lower()
            if entry_type not in ("article", "inproceedings", "misc", "techreport"):
                continue

            fields_str = match.group(3)
            fields = BatchProcessor._parse_bibtex_fields(fields_str)

            title = fields.get("title", "")
            author_str = fields.get("author", "")
            authors = [
                a.strip() for a in re.split(r"\s+and\s+", author_str)
                if a.strip()
            ]

            papers.append(Paper(
                title=title,
                authors=authors,
                year=fields.get("year"),
                doi=fields.get("doi"),
                journal=fields.get("journal") or fields.get("booktitle"),
                url=fields.get("url"),
            ))

        return papers

    @staticmethod
    def _parse_bibtex_fields(fields_str: str) -> Dict[str, str]:
        """解析 BibTeX 字段列表为字典。"""
        result: Dict[str, str] = {}
        # 匹配 field = {value} 或 field = "value"
        field_pattern = re.compile(
            r'(\w+)\s*=\s*[{"]([^}"]*)[}"]',
            re.MULTILINE,
        )
        for fm in field_pattern.finditer(fields_str):
            key = fm.group(1).lower()
            value = fm.group(2).strip()
            # 清理 LaTeX 特殊字符
            value = re.sub(r"\{|\\", "", value)
            value = re.sub(r"\s+", " ", value)
            result[key] = value
        return result

    # ── 批量处理 ──────────────────────────────────────────────────

    def process_batch(
        self,
        titles: List[str],
        output_dir: Optional[str] = None,
        max_results: int = 3,
        engines: Optional[List[str]] = None,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> List[Paper]:
        """执行批量下载。

        Args:
            titles:        论文标题列表。
            output_dir:    输出目录。
            max_results:   搜索候选数。
            engines:       搜索引擎列表。
            skip_existing: 跳过已存在输出目录中的 PDF。默认 True。
            **kwargs:      传递给 PaperDownloader.batch_download() 的额外参数。

        Returns:
            Paper 对象列表。
        """
        if not titles:
            self.logger.warning("标题列表为空")
            return []

        # 跳过已有 PDF 的论文
        effective_titles = titles
        pre_skipped = 0
        if skip_existing and output_dir:
            out = Path(output_dir)
            effective_titles = []
            for title in titles:
                # 检查目录中是否有匹配的 PDF
                found = False
                if out.exists():
                    for pdf in out.glob("*.pdf"):
                        if pdf.stem in title or title[:30] in pdf.stem:
                            found = True
                            break
                if found:
                    pre_skipped += 1
                else:
                    effective_titles.append(title)
            if pre_skipped > 0:
                self.logger.info("预跳过 %d 篇已有 PDF 的论文", pre_skipped)

        self.logger.info("批量处理 %d 篇论文 (跳过 %d 篇)",
                         len(effective_titles), pre_skipped)

        results = self._downloader.batch_download(
            titles=effective_titles,
            output_dir=output_dir,
            max_results=max_results,
            engines=engines,
            **kwargs,
        )

        return results

    # ── 报告生成 ──────────────────────────────────────────────────

    def generate_report(
        self,
        papers: List[Paper],
        output_dir: str = ".",
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """生成下载报告。

        Args:
            papers:     Paper 对象列表。
            output_dir: 报告输出目录。
            formats:    报告格式列表，默认 ["json", "csv"]。

        Returns:
            {format: file_path} 映射。
        """
        formats = formats or ["json", "csv"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_files: Dict[str, str] = {}

        for fmt in formats:
            if fmt == "json":
                path = os.path.join(output_dir, f"download_report_{timestamp}.json")
                self._write_json_report(papers, path)
                report_files["json"] = path
            elif fmt == "csv":
                path = os.path.join(output_dir, f"download_report_{timestamp}.csv")
                self._write_csv_report(papers, path)
                report_files["csv"] = path

        self.logger.info("报告已生成: %s", report_files)
        return report_files

    def _write_json_report(self, papers: List[Paper], path: str) -> None:
        """写入 JSON 格式报告。"""
        succeeded = sum(1 for p in papers if p.has_pdf)
        failed = len(papers) - succeeded

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total": len(papers),
                "succeeded": succeeded,
                "failed": failed,
            },
            "papers": [p.to_dict() for p in papers],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def _write_csv_report(self, papers: List[Paper], path: str) -> None:
        """写入 CSV 格式报告。"""
        columns = [
            "title", "first_author", "year", "doi", "arxiv_id",
            "source", "citation_count", "pdf_path", "file_size", "status",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for p in papers:
                writer.writerow({
                    "title": p.title,
                    "first_author": p.first_author,
                    "year": p.year or "",
                    "doi": p.doi or "",
                    "arxiv_id": p.arxiv_id or "",
                    "source": p.source or "",
                    "citation_count": p.citation_count or "",
                    "pdf_path": p.pdf_path or "",
                    "file_size": p.file_size,
                    "status": "success" if p.has_pdf else "failed",
                })
