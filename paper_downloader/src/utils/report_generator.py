"""
report_generator.py — 报告生成器

从下载结果生成多种格式的报告：JSON、CSV、Markdown。
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_downloader.src.models.paper import Paper


class ReportGenerator:
    """下载报告生成器。

    从 Paper 列表生成 JSON / CSV / Markdown 格式的下载摘要报告。

    Usage::

        papers = downloader.batch_download([...])
        gen = ReportGenerator()
        gen.export_json(papers, "report.json")
        gen.export_csv(papers, "report.csv")
        print(gen.generate_summary(papers))
    """

    def __init__(
        self,
        output_dir: str = ".",
        timestamp_format: str = "%Y%m%d_%H%M%S",
    ):
        """初始化。

        Args:
            output_dir:       默认输出目录。
            timestamp_format: 文件时间戳格式。
        """
        self.output_dir = output_dir
        self.timestamp_format = timestamp_format

    # ── 摘要 ──────────────────────────────────────────────────────

    def generate_summary(self, papers: List[Paper]) -> Dict[str, Any]:
        """生成下载摘要统计。

        Args:
            papers: Paper 对象列表。

        Returns:
            摘要字典。
        """
        total = len(papers)
        succeeded = sum(1 for p in papers if p.has_pdf)
        failed = total - succeeded
        total_bytes = sum(p.file_size for p in papers if p.file_size > 0)

        # 按来源统计
        by_source: Dict[str, int] = {}
        for p in papers:
            src = p.source or "unknown"
            by_source[src] = by_source.get(src, 0) + 1

        # 按年份统计
        by_year: Dict[str, int] = {}
        for p in papers:
            y = p.year or "unknown"
            by_year[y] = by_year.get(y, 0) + 1

        return {
            "generated_at": datetime.now().isoformat(),
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": round(succeeded / max(total, 1) * 100, 1),
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "by_source": by_source,
            "by_year": by_year,
            "papers": [self._paper_summary(p) for p in papers],
        }

    def _paper_summary(self, paper: Paper) -> Dict[str, Any]:
        """单篇论文摘要。"""
        return {
            "title": paper.title,
            "authors": "; ".join(paper.authors[:3]),
            "year": paper.year,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "source": paper.source,
            "status": "success" if paper.has_pdf else "failed",
            "pdf_path": paper.pdf_path,
            "file_size": paper.file_size,
        }

    # ── 导出 JSON ─────────────────────────────────────────────────

    def export_json(
        self,
        papers: List[Paper],
        file_path: Optional[str] = None,
    ) -> str:
        """导出为 JSON 文件。

        Args:
            papers:    Paper 列表。
            file_path: 文件路径，None 生成默认文件名。

        Returns:
            文件路径。
        """
        path = file_path or self._make_path("download_report", ".json")
        summary = self.generate_summary(papers)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return path

    def export_json_string(self, papers: List[Paper], indent: int = 2) -> str:
        """导出为 JSON 字符串。"""
        return json.dumps(
            self.generate_summary(papers),
            ensure_ascii=False,
            indent=indent,
        )

    # ── 导出 CSV ──────────────────────────────────────────────────

    def export_csv(
        self,
        papers: List[Paper],
        file_path: Optional[str] = None,
    ) -> str:
        """导出为 CSV 文件。

        Args:
            papers:    Paper 列表。
            file_path: 文件路径，None 生成默认文件名。

        Returns:
            文件路径。
        """
        path = file_path or self._make_path("download_report", ".csv")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "title", "first_author", "year", "doi", "arxiv_id",
            "source", "citation_count", "journal",
            "pdf_path", "file_size", "status",
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
                    "journal": p.journal or "",
                    "pdf_path": p.pdf_path or "",
                    "file_size": p.file_size,
                    "status": "success" if p.has_pdf else "failed",
                })

        return path

    def export_csv_string(self, papers: List[Paper]) -> str:
        """导出为 CSV 字符串。"""
        output = io.StringIO()
        columns = ["title", "first_author", "year", "doi", "status"]
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for p in papers:
            writer.writerow({
                "title": p.title,
                "first_author": p.first_author,
                "year": p.year or "",
                "doi": p.doi or "",
                "status": "success" if p.has_pdf else "failed",
            })
        return output.getvalue()

    # ── Markdown 报告 ─────────────────────────────────────────────

    def generate_markdown(self, papers: List[Paper]) -> str:
        """生成 Markdown 格式的下载报告。

        Args:
            papers: Paper 列表。

        Returns:
            Markdown 字符串。
        """
        summary = self.generate_summary(papers)
        lines: List[str] = []

        # 标题
        lines.append("# 论文下载报告")
        lines.append("")
        lines.append(f"**生成时间**: {summary['generated_at']}")
        lines.append("")

        # 总览
        lines.append("## 总览")
        lines.append("")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 总计 | {summary['total']} |")
        lines.append(f"| 成功 | {summary['succeeded']} |")
        lines.append(f"| 失败 | {summary['failed']} |")
        lines.append(f"| 成功率 | {summary['success_rate']}% |")
        lines.append(f"| 总大小 | {summary['total_size_mb']} MB |")
        lines.append("")

        # 按来源
        if summary["by_source"]:
            lines.append("## 按数据来源")
            lines.append("")
            lines.append("| 来源 | 数量 |")
            lines.append("|------|------|")
            for src, count in sorted(summary["by_source"].items(), key=lambda x: -x[1]):
                lines.append(f"| {src} | {count} |")
            lines.append("")

        # 按年份
        if summary["by_year"]:
            lines.append("## 按年份")
            lines.append("")
            lines.append("| 年份 | 数量 |")
            lines.append("|------|------|")
            for year, count in sorted(summary["by_year"].items(), key=lambda x: -x[1]):
                lines.append(f"| {year} | {count} |")
            lines.append("")

        # 论文列表
        lines.append("## 论文列表")
        lines.append("")
        lines.append("| # | 标题 | 作者 | 年份 | 状态 |")
        lines.append("|---|------|------|------|------|")
        for i, p in enumerate(papers, 1):
            status = "✅" if p.has_pdf else "❌"
            title = p.title[:60] + ("..." if len(p.title) > 60 else "")
            author = p.first_author
            year = p.year or "-"
            lines.append(f"| {i} | {title} | {author} | {year} | {status} |")

        lines.append("")
        return "\n".join(lines)

    def export_markdown(
        self,
        papers: List[Paper],
        file_path: Optional[str] = None,
    ) -> str:
        """导出 Markdown 报告到文件。

        Args:
            papers:    Paper 列表。
            file_path: 文件路径。

        Returns:
            文件路径。
        """
        path = file_path or self._make_path("download_report", ".md")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_markdown(papers)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # ── JSON Lines (JSONL) ────────────────────────────────────────

    def export_jsonl(
        self,
        papers: List[Paper],
        file_path: Optional[str] = None,
    ) -> str:
        """导出为 JSONL 格式（每行一个 JSON 对象，便于流式处理）。

        Args:
            papers:    Paper 列表。
            file_path: 文件路径。

        Returns:
            文件路径。
        """
        path = file_path or self._make_path("download_records", ".jsonl")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(self._paper_summary(p), ensure_ascii=False) + "\n")
        return path

    # ── BibTeX 导出 ───────────────────────────────────────────────

    def export_bibtex(
        self,
        papers: List[Paper],
        file_path: Optional[str] = None,
    ) -> str:
        """导出为 BibTeX 文件。

        Args:
            papers:    Paper 列表（仅导出已下载成功的）。
            file_path: 文件路径。

        Returns:
            文件路径。
        """
        path = file_path or self._make_path("references", ".bib")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        succeeded = [p for p in papers if p.title]
        entries = [p.to_bibtex() for p in succeeded if p.title]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(entries) + "\n")

        return path

    # ── 工具 ──────────────────────────────────────────────────────

    def _make_path(self, prefix: str, ext: str) -> str:
        """生成带时间戳的文件路径。"""
        ts = datetime.now().strftime(self.timestamp_format)
        return os.path.join(self.output_dir, f"{prefix}_{ts}{ext}")
