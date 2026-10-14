"""
base_downloader.py — 下载器抽象基类

定义 PDF 下载的统一接口与 PDF 文件校验方法。
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class Downloader(ABC):
    """PDF 下载器抽象基类。

    所有下载器必须实现 ``download()`` 方法。
    ``validate_pdf()`` 提供通用的 PDF 有效性检查。
    """

    DOWNLOADER_NAME: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化下载器。

        Args:
            config: 配置字典（包含 timeout、proxy 等）。
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    # ── 抽象接口 ──────────────────────────────────────────────────

    @abstractmethod
    def download(
        self,
        url: str,
        save_path: str | Path,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Path]:
        """下载 PDF 文件到本地。

        Args:
            url:       PDF 文件的 URL。
            save_path: 保存目录路径。
            filename:  自定义文件名（不含 .pdf 扩展名），None 则从 URL 推断。
            **kwargs:  额外参数。

        Returns:
            下载成功返回完整文件路径的 Path，失败返回 None。
        """
        ...

    # ── PDF 校验 ──────────────────────────────────────────────────

    @staticmethod
    def validate_pdf(file_path: str | Path) -> bool:
        """验证文件是否为有效的 PDF。

        通过检查 PDF 文件头魔数 ``%PDF-`` 来判断。

        Args:
            file_path: PDF 文件路径。

        Returns:
            True 表示有效 PDF 文件。
        """
        path = Path(file_path)
        if not path.exists():
            return False
        if not path.is_file():
            return False
        if path.stat().st_size == 0:
            return False

        try:
            with open(path, "rb") as f:
                header = f.read(5)
            # 有效 PDF 必须以 %PDF- 开头（允许前面有少量 BOM 或空白）
            if header.startswith(b"%PDF-"):
                return True
            # 某些 PDF 前面可能有 BOM
            if len(header) >= 8 and b"%PDF-" in header[:8]:
                return True
            return False
        except (OSError, IOError) as exc:
            logger = logging.getLogger("Downloader")
            logger.warning("PDF 校验时读取文件失败: %s — %s", file_path, exc)
            return False

    @staticmethod
    def validate_pdf_deep(file_path: str | Path) -> bool:
        """深度验证 PDF 文件完整性。

        除文件头外，还检查文件尾的 ``%%EOF`` 标记。

        Args:
            file_path: PDF 文件路径。

        Returns:
            True 表示 PDF 结构完整。
        """
        if not Downloader.validate_pdf(file_path):
            return False

        path = Path(file_path)
        try:
            with open(path, "rb") as f:
                # 检查尾部 — 最后 1024 字节内应有 %%EOF
                f.seek(-min(1024, path.stat().st_size), os.SEEK_END)
                tail = f.read()
            return b"%%EOF" in tail
        except (OSError, IOError) as exc:
            logger = logging.getLogger("Downloader")
            logger.warning("PDF 深度校验失败: %s — %s", file_path, exc)
            return False

    # ── 工具 ──────────────────────────────────────────────────────

    # Windows 文件名非法字符
    _ILLEGAL_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'

    @staticmethod
    def _extract_filename_from_url(url: str) -> str:
        """从 URL 中提取文件名。"""
        from urllib.parse import unquote, urlparse

        path = urlparse(url).path
        filename = unquote(path.rsplit("/", 1)[-1]) if "/" in path else unquote(path)
        if not filename or "." not in filename:
            filename = "paper.pdf"
        return Downloader._safe_filename(filename)

    @staticmethod
    def _safe_filename(name: str, max_len: int = 200) -> str:
        """清理文件名中的非法字符（Windows/macOS/Linux 兼容）。"""
        import re
        sanitized = re.sub(Downloader._ILLEGAL_FILENAME_CHARS, "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = re.sub(r"_(\.\w+)$", r"\1", sanitized)  # "name_.pdf" → "name.pdf"
        sanitized = sanitized.strip("_ .")
        if not sanitized:
            sanitized = "paper.pdf"
        if len(sanitized) > max_len:
            # 保留扩展名
            dot_idx = sanitized.rfind(".")
            if dot_idx > 0:
                ext = sanitized[dot_idx:]
                stem = sanitized[:dot_idx]
                sanitized = stem[:max_len - len(ext)] + ext
            else:
                sanitized = sanitized[:max_len]
        return sanitized

    @staticmethod
    def _sanitize_path(path: str | Path) -> Path:
        """将路径字符串转为绝对 Path 并确保父目录存在。"""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
