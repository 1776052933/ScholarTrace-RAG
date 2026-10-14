"""
pdf_processor.py — PDF 处理工具

提供 PDF 元数据提取、按规范重命名、损坏检测等功能。
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_downloader.src.downloaders.base_downloader import Downloader

# PyPDF2 可选导入
try:
    from PyPDF2 import PdfReader, PdfWriter
    PYPDF2_AVAILABLE = True
except ImportError:
    PdfReader = None  # type: ignore[assignment]
    PdfWriter = None  # type: ignore[assignment]
    PYPDF2_AVAILABLE = False

# 文件名非法字符（与 paper_manager.py 一致）
_ILLEGAL_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


class PDFProcessor:
    """PDF 文件处理工具集。

    提供 PDF 元数据提取、合规重命名、损坏检测等静态方法。

    Usage::

        meta = PDFProcessor.extract_metadata("paper.pdf")
        new_path = PDFProcessor.rename_pdf("paper.pdf", "Attention Is All You Need", "Vaswani et al.")
        ok = PDFProcessor.check_corrupted("paper.pdf")
    """

    @staticmethod
    def extract_metadata(pdf_path: str | Path) -> Dict[str, Any]:
        """提取 PDF 元数据。

        使用 PyPDF2 读取 PDF 信息字典，并补充文件系统信息。

        Args:
            pdf_path: PDF 文件路径。

        Returns:
            元数据 dict，包含:
                - title, author, subject, creator, producer
                - pages         : 页数
                - file_size     : 字节数
                - created_date  : 文件创建时间
                - modified_date : 文件修改时间
                - is_encrypted  : 是否加密
                - is_valid_pdf  : 是否有效 PDF
        """
        path = Path(pdf_path)
        meta: Dict[str, Any] = {
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "file_size": path.stat().st_size if path.exists() else 0,
            "created_date": datetime.fromtimestamp(path.stat().st_ctime).isoformat() if path.exists() else None,
            "modified_date": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
            "is_valid_pdf": Downloader.validate_pdf(path),
            "pages": 0,
            "is_encrypted": False,
            "title": None,
            "author": None,
            "subject": None,
            "creator": None,
            "producer": None,
        }

        if not path.exists() or not meta["is_valid_pdf"]:
            return meta

        # PyPDF2 元数据
        if PYPDF2_AVAILABLE:
            try:
                reader = PdfReader(str(path))
                meta["pages"] = len(reader.pages)
                meta["is_encrypted"] = reader.is_encrypted

                info = reader.metadata
                if info:
                    meta["title"] = PDFProcessor._safe_meta(info, "/Title")
                    meta["author"] = PDFProcessor._safe_meta(info, "/Author")
                    meta["subject"] = PDFProcessor._safe_meta(info, "/Subject")
                    meta["creator"] = PDFProcessor._safe_meta(info, "/Creator")
                    meta["producer"] = PDFProcessor._safe_meta(info, "/Producer")
            except Exception as exc:
                logger = logging.getLogger("PDFProcessor")
                logger.warning("PyPDF2 读取元数据失败: %s — %s", pdf_path, exc)

        return meta

    @staticmethod
    def _safe_meta(info: Any, key: str) -> Optional[str]:
        """安全地从 PDF 元数据字典中取出字符串值。"""
        try:
            val = info.get(key)
            if val:
                return str(val).strip() or None
        except Exception:
            pass
        return None

    # ── 重命名 ────────────────────────────────────────────────────

    @staticmethod
    def rename_pdf(
        pdf_path: str | Path,
        title: str = "",
        authors: str = "",
        year: str = "",
        template: str = "{first_author}_{year}_{title}",
        keep_original: bool = True,
    ) -> Optional[Path]:
        """按论文元数据重命名 PDF 文件。

        Args:
            pdf_path:   原始 PDF 路径。
            title:      论文标题。
            authors:    作者字符串。
            year:       发表年份。
            template:   命名模板，可用变量: {first_author}, {year}, {title}。
            keep_original: 是否保留原文件。True=复制，False=移动。

        Returns:
            重命名后的文件 Path，失败返回 None。
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger = logging.getLogger("PDFProcessor")
            logger.error("文件不存在: %s", pdf_path)
            return None

        # 提取第一作者姓氏
        first_author = "unknown"
        if authors:
            first_name = authors.split(",")[0].strip() if "," in authors else authors.split(";")[0].strip()
            parts = first_name.split()
            first_author = parts[-1] if parts else first_name

        year_str = str(year) if year else "nodate"
        title_part = title[:80] if title else "untitled"

        new_stem = (
            template.replace("{first_author}", first_author)
            .replace("{year}", year_str)
            .replace("{title}", title_part)
        )
        new_stem = PDFProcessor._sanitize_name(new_stem)
        new_name = f"{new_stem}.pdf"
        new_path = pdf_path.parent / new_name

        # 处理重名
        if new_path.exists() and new_path != pdf_path:
            counter = 1
            while (pdf_path.parent / f"{new_stem}_{counter}.pdf").exists():
                counter += 1
            new_path = pdf_path.parent / f"{new_stem}_{counter}.pdf"

        try:
            if keep_original:
                import shutil
                shutil.copy2(pdf_path, new_path)
            else:
                pdf_path.rename(new_path)
            return new_path
        except OSError as exc:
            logger = logging.getLogger("PDFProcessor")
            logger.error("重命名失败: %s → %s — %s", pdf_path, new_path, exc)
            return None

    @staticmethod
    def _sanitize_name(name: str, max_len: int = 200) -> str:
        """清理文件名中的非法字符。"""
        sanitized = re.sub(_ILLEGAL_CHARS, "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = re.sub(r"_(\.\w+)$", r"\1", sanitized)
        sanitized = sanitized.strip("_ .")
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
        return sanitized

    # ── 损坏检测 ──────────────────────────────────────────────────

    @staticmethod
    def check_corrupted(pdf_path: str | Path) -> bool:
        """检查 PDF 文件是否损坏。

        综合检查:
            1. 文件头魔数 (%PDF-) 和尾标记 (%%EOF)。
            2. PyPDF2 能否正常解析所有页面。
            3. 文件交叉引用表 (xref) 是否可读。

        Args:
            pdf_path: PDF 文件路径。

        Returns:
            True 表示 PDF 已损坏。
        """
        path = Path(pdf_path)
        logger = logging.getLogger("PDFProcessor")

        if not path.exists():
            logger.warning("文件不存在，无法检查: %s", pdf_path)
            return True

        # 文件大小检查
        if path.stat().st_size == 0:
            logger.warning("PDF 文件为空: %s", pdf_path)
            return True

        # 基本结构检查
        if not Downloader.validate_pdf_deep(path):
            logger.warning("PDF 结构不完整（缺少头部或尾部）: %s", pdf_path)
            return True

        # PyPDF2 深度检查
        if PYPDF2_AVAILABLE:
            try:
                reader = PdfReader(str(path))
                num_pages = len(reader.pages)
                if num_pages == 0:
                    logger.warning("PDF 页数为 0: %s", pdf_path)
                    return True
                # 尝试读取每页（会触发 xref 解析）
                for i, page in enumerate(reader.pages):
                    _ = page.extract_text()  # 触发页面解析
                    if i >= 5:  # 只检查前 5 页以节省时间
                        break
            except Exception as exc:
                logger.warning("PDF 解析失败，可能已损坏: %s — %s", pdf_path, exc)
                return True

        return False  # 未发现损坏

    # ── 页面信息 ──────────────────────────────────────────────────

    @staticmethod
    def get_page_count(pdf_path: str | Path) -> Optional[int]:
        """获取 PDF 页数。

        Args:
            pdf_path: PDF 文件路径。

        Returns:
            页数，无法读取返回 None。
        """
        if not PYPDF2_AVAILABLE:
            return None
        path = Path(pdf_path)
        if not path.exists() or not Downloader.validate_pdf(path):
            return None
        try:
            reader = PdfReader(str(path))
            return len(reader.pages)
        except Exception as exc:
            logger = logging.getLogger("PDFProcessor")
            logger.warning("获取页数失败: %s — %s", pdf_path, exc)
            return None

    # ── 文本提取 ──────────────────────────────────────────────────

    @staticmethod
    def extract_text(
        pdf_path: str | Path,
        max_pages: int = 3,
    ) -> Optional[str]:
        """提取 PDF 首页文本（通常包含标题和摘要）。

        Args:
            pdf_path:  PDF 文件路径。
            max_pages: 最多提取页数。

        Returns:
            提取的文本，失败返回 None。
        """
        if not PYPDF2_AVAILABLE:
            return None
        path = Path(pdf_path)
        if not path.exists():
            return None
        try:
            reader = PdfReader(str(path))
            texts = []
            for i, page in enumerate(reader.pages):
                if i >= max_pages:
                    break
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n".join(texts) if texts else None
        except Exception as exc:
            logger = logging.getLogger("PDFProcessor")
            logger.warning("提取文本失败: %s — %s", pdf_path, exc)
            return None
