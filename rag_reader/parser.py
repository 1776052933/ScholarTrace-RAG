"""使用 PyMuPDF 将 PDF 按页转换为文本。"""

import logging
from pathlib import Path
from typing import Dict, List, Union
from .errors import RAGDocumentError


logger = logging.getLogger(__name__)


class PDFParserError(RAGDocumentError):
    """PDF 解析异常基类。"""


class PDFOpenError(PDFParserError):
    """PDF 不存在、损坏、加密或无法打开。"""


class ScannedPDFError(PDFParserError):
    """PDF 仅包含扫描图片，无法直接提取文本。"""


class EmptyPDFTextError(PDFParserError):
    """PDF 没有可提取的文本。"""


class PDFParser:
    """通过 PyMuPDF 提取 PDF 文本并保留页码。"""

    def parse(self, file_path: Union[str, Path]) -> List[Dict[str, Union[int, str]]]:
        """返回 ``[{"page": 1, "text": "..."}]`` 页面结构。"""
        path = Path(file_path).expanduser()
        if not path.exists() or not path.is_file():
            raise PDFOpenError(f"PDF file not found: {path}")
        if path.suffix.casefold() != ".pdf":
            raise PDFOpenError(f"file is not a PDF: {path}")

        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PDFParser requires PyMuPDF: pip install pymupdf") from exc

        logger.info("开始解析 PDF：%s", path)
        try:
            document = fitz.open(str(path))
        except Exception as exc:
            logger.warning("PDF 打开失败：%s", path)
            raise PDFOpenError(f"cannot open PDF: {path}") from exc

        try:
            if document.needs_pass:
                raise PDFOpenError(f"encrypted PDF requires a password: {path}")
            if document.page_count == 0:
                raise EmptyPDFTextError(f"PDF contains no pages: {path}")

            pages: List[Dict[str, Union[int, str]]] = []
            image_count = 0
            blank_pages = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                try:
                    text = page.get_text("text", sort=True).strip()
                except Exception as exc:
                    logger.warning("PDF 第 %d 页文本提取失败：%s", page_index + 1, path)
                    raise PDFParserError(
                        f"failed to extract page {page_index + 1}: {path}"
                    ) from exc
                try:
                    image_count += len(page.get_images(full=True))
                except Exception:
                    logger.debug("无法读取 PDF 第 %d 页图片信息", page_index + 1, exc_info=True)
                if not text:
                    blank_pages.append(page_index + 1)
                pages.append({"page": page_index + 1, "text": text})

            total_text = "".join(str(item["text"]) for item in pages).strip()
            if not total_text:
                if image_count > 0:
                    logger.warning("检测到扫描 PDF，无法提取文本：%s", path)
                    raise ScannedPDFError(f"scanned PDF contains no extractable text: {path}")
                logger.warning("PDF 未提取到任何文本：%s", path)
                raise EmptyPDFTextError(f"PDF contains no extractable text: {path}")
            if blank_pages:
                logger.warning("PDF 存在无文本页面：%s，pages=%s", path, blank_pages)
            logger.info(
                "PDF 解析完成：%s，pages=%d，characters=%d",
                path,
                len(pages),
                sum(len(str(item["text"])) for item in pages),
            )
            return pages
        finally:
            document.close()

    def extract_text(self, file_path: Union[str, Path]) -> str:
        """解析 PDF 并返回按页分隔的完整文本。"""
        pages = self.parse(file_path)
        return "\n\n".join(str(item["text"]) for item in pages if item["text"])
