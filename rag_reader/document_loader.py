"""从 Literature Database 的 PaperResource 加载论文文件。"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from literature_database.schema import PaperResource
from .errors import RAGDocumentError

from .schema import PaperDocument


class DocumentLoadError(RAGDocumentError):
    """论文文件加载失败。"""


class DocumentNotFoundError(DocumentLoadError):
    """论文文件不存在。"""


class DocumentSizeError(DocumentLoadError):
    """论文文件为空或超过允许大小。"""


class DocumentEncodingError(DocumentLoadError):
    """文本文件无法使用支持的编码解码。"""


class UnsupportedDocumentTypeError(DocumentLoadError):
    """论文文件类型不受支持。"""


class DocumentLoader:
    """加载 PDF、TXT 和 Markdown；PDF 仅校验，不解析内容。"""

    _TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
    _SUPPORTED_SUFFIXES = {".pdf", *_TEXT_SUFFIXES}

    def __init__(
        self,
        *,
        max_file_size_mb: float = 100.0,
        encodings: Sequence[str] = ("utf-8-sig", "utf-8", "utf-16", "gb18030"),
    ) -> None:
        if max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be greater than zero")
        if not encodings:
            raise ValueError("encodings must not be empty")
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.encodings: Tuple[str, ...] = tuple(encodings)
        self.detected_encodings: Dict[str, str] = {}

    def load(self, resource: PaperResource) -> PaperDocument:
        """将 PaperResource 转换为 PaperDocument。"""
        if not isinstance(resource, PaperResource):
            raise TypeError("resource must be a literature_database.schema.PaperResource")
        file_path = _path(resource.file_path)
        text_path = _path(resource.text_path)
        if file_path is None and text_path is None:
            raise DocumentNotFoundError(f"paper {resource.paper.paper_id} has no file path")

        if file_path is not None:
            self._validate_file(file_path)
        if text_path is not None and text_path != file_path:
            self._validate_file(text_path)

        content = ""
        content_path = text_path or file_path
        assert content_path is not None
        if content_path.suffix.casefold() in self._TEXT_SUFFIXES:
            content, encoding = self._read_text(content_path)
            self.detected_encodings[str(content_path)] = encoding
        elif content_path.suffix.casefold() == ".pdf":
            # PDF 内容由后续 parser 处理，此处只做文件级校验。
            content = ""
        else:  # _validate_file 通常已经拦截，保留防御性分支。
            raise UnsupportedDocumentTypeError(f"unsupported file type: {content_path.suffix}")

        return PaperDocument(
            paper_id=resource.paper.paper_id,
            title=resource.paper.title,
            file_path=str(file_path) if file_path is not None else None,
            text_path=str(text_path) if text_path is not None else None,
            content=content,
        )

    def load_document(self, resource: PaperResource) -> PaperDocument:
        """``load`` 的语义化别名。"""
        return self.load(resource)

    def _validate_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise DocumentNotFoundError(f"document file not found: {path}")
        suffix = path.suffix.casefold()
        if suffix not in self._SUPPORTED_SUFFIXES:
            raise UnsupportedDocumentTypeError(f"unsupported file type: {suffix or '<none>'}")
        size = path.stat().st_size
        if size <= 0:
            raise DocumentSizeError(f"document file is empty: {path}")
        if size > self.max_file_size_bytes:
            raise DocumentSizeError(
                f"document file exceeds {self.max_file_size_bytes} bytes: {path}"
            )

    def _read_text(self, path: Path) -> tuple[str, str]:
        data = path.read_bytes()
        for encoding in self.encodings:
            try:
                return data.decode(encoding), encoding
            except (UnicodeDecodeError, LookupError):
                continue
        raise DocumentEncodingError(
            f"cannot decode {path} with supported encodings: {', '.join(self.encodings)}"
        )


def _path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser() if value and value.strip() else None
