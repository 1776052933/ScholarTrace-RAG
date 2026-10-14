"""
arxiv_downloader.py — arXiv 专用下载器

优先使用 arXiv API 获取 PDF，失败时回退到直接 HTTP 下载。

arXiv PDF URL 格式:
    https://arxiv.org/pdf/{arxiv_id}.pdf
    https://arxiv.org/pdf/{arxiv_id}v{version}.pdf
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

from paper_downloader.src.downloaders.base_downloader import Downloader
from paper_downloader.src.downloaders.http_downloader import HTTPDownloader


class ArxivPDFDownloader(Downloader):
    """arXiv 专用 PDF 下载器。

    策略:
        1. 优先通过 arXiv API 获取 PDF URL（更可靠、有重定向）。
        2. 回退到直接拼接 arXiv PDF URL 进行 HTTP 下载。

    Usage::

        dl = ArxivPDFDownloader(config)
        path = dl.download("2401.00001", save_path="./papers")
        # 或直接传 URL
        path = dl.download("https://arxiv.org/abs/2401.00001")
    """

    DOWNLOADER_NAME = "arxiv"

    # arXiv ID / URL 匹配模式
    _ARXIV_ID_RE = re.compile(
        r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?|[\w\-]+/\d{7}(?:v\d+)?)",
        re.IGNORECASE,
    )

    _ARXIV_PDF_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._http_downloader = HTTPDownloader(config)

    # ── 下载 ──────────────────────────────────────────────────────

    def download(
        self,
        url_or_id: str,
        save_path: str | Path = "./papers",
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Path]:
        """下载 arXiv 论文 PDF。

        Args:
            url_or_id: arXiv URL 或纯 ID（如 "2401.00001" 或
                       "https://arxiv.org/abs/2401.00001"）。
            save_path: 保存目录。
            filename:  自定义文件名（不含 .pdf）。
            **kwargs:  透传给 HTTPDownloader.download()。

        Returns:
            下载成功返回文件 Path，失败返回 None。
        """
        arxiv_id = self._extract_arxiv_id(url_or_id)
        if not arxiv_id:
            self.logger.error("无法提取 arXiv ID: %s", url_or_id)
            return None

        self.logger.info("arXiv 下载: %s", arxiv_id)

        # 策略 1: 尝试通过 arXiv API 获取
        if kwargs.get("use_api", True):
            pdf_url = self._get_pdf_url_via_api(arxiv_id)
            if pdf_url:
                result = self._http_downloader.download(pdf_url, save_path, filename, **kwargs)
                if result and self.validate_pdf(result):
                    return result
                self.logger.warning("API 路径下载/校验失败，回退到直接 URL")

        # 策略 2: 直接拼接 PDF URL 下载
        pdf_url = self._ARXIV_PDF_TEMPLATE.format(arxiv_id=arxiv_id)
        result = self._http_downloader.download(pdf_url, save_path, filename, **kwargs)
        if result and self.validate_pdf(result):
            return result

        self.logger.error("arXiv PDF 下载失败（所有路径均已尝试）: %s", arxiv_id)
        return None

    # ── arXiv API ─────────────────────────────────────────────────

    def _get_pdf_url_via_api(self, arxiv_id: str) -> Optional[str]:
        """通过 arXiv API 查询论文并提取 PDF URL。

        Args:
            arxiv_id: arXiv ID。

        Returns:
            PDF URL 或 None。
        """
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search))
            return result.pdf_url if hasattr(result, "pdf_url") and result.pdf_url else None
        except ImportError:
            self.logger.debug("arxiv 包未安装，跳过 API 查询")
            return None
        except StopIteration:
            self.logger.warning("arXiv API 未找到: %s", arxiv_id)
            return None
        except Exception as exc:
            self.logger.warning("arXiv API 查询失败 (%s): %s", arxiv_id, exc)
            return None

    # ── ID 提取 ───────────────────────────────────────────────────

    @classmethod
    def _extract_arxiv_id(cls, text: str) -> Optional[str]:
        """从 URL 或文本中提取 arXiv ID。

        支持格式:
            - 2401.00001
            - 2401.00001v2
            - https://arxiv.org/abs/2401.00001
            - https://arxiv.org/pdf/2401.00001.pdf
            - hep-th/9711200
        """
        if not text:
            return None
        # 先去除 .pdf 后缀
        text = re.sub(r"\.pdf$", "", text.strip(), flags=re.IGNORECASE)
        match = cls._ARXIV_ID_RE.search(text)
        return match.group(1) if match else None

    # ── 资源管理 ──────────────────────────────────────────────────

    def close(self) -> None:
        """关闭内部 HTTP 下载器。"""
        self._http_downloader.close()

    def __enter__(self) -> "ArxivPDFDownloader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
