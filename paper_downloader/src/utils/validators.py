"""
validators.py — 输入验证与清理工具

提供论文标题、DOI、URL 的格式校验和文件名清理函数。
所有函数均为纯函数，无副作用。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


# ── 常量 ──────────────────────────────────────────────────────────

# DOI 正则: 10.XXXX/XXXX...
_DOI_PATTERN = re.compile(
    r"^10\.\d{4,}(?:\.\d+)?/[-._;()/:a-zA-Z0-9]+$",
)

# 合法 URL 协议
_VALID_URL_SCHEMES = {"http", "https", "ftp"}

# 文件名非法字符
_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 标题最小/最大长度
_TITLE_MIN_LEN = 2
_TITLE_MAX_LEN = 1000


# ── 标题验证 ──────────────────────────────────────────────────────

def validate_title(title: str) -> Tuple[bool, Optional[str]]:
    """验证论文标题是否合法。

    检查:
        - 非空
        - 长度在 2~1000 之间
        - 不是纯数字或纯标点
        - 不包含控制字符

    Args:
        title: 论文标题字符串。

    Returns:
        (is_valid, error_message): True 表示合法，
        False 时 error_message 包含原因。

    Example::

        >>> validate_title("Attention Is All You Need")
        (True, None)
        >>> validate_title("")
        (False, "标题不能为空")
    """
    if not title or not title.strip():
        return False, "标题不能为空"

    stripped = title.strip()

    if len(stripped) < _TITLE_MIN_LEN:
        return False, f"标题过短（最少 {_TITLE_MIN_LEN} 个字符）"

    if len(stripped) > _TITLE_MAX_LEN:
        return False, f"标题过长（最多 {_TITLE_MAX_LEN} 个字符）"

    # 检查控制字符
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", stripped):
        return False, "标题包含控制字符"

    # 纯数字/标点检查
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars == 0:
        return False, "标题必须包含至少一个字母"

    return True, None


def validate_title_or_raise(title: str) -> str:
    """验证标题，不合法时抛出 ValueError。

    Args:
        title: 论文标题。

    Returns:
        清理后的标题字符串。

    Raises:
        ValueError: 标题不合法。
    """
    is_valid, error = validate_title(title)
    if not is_valid:
        raise ValueError(f"标题无效: {error}")
    return title.strip()


# ── DOI 验证 ──────────────────────────────────────────────────────

def validate_doi(doi: str) -> Tuple[bool, Optional[str]]:
    """验证 DOI 格式。

    标准 DOI 格式: 10.XXXX/YYYY
    也支持 http://doi.org/ 或 https://doi.org/ 前缀。

    Args:
        doi: DOI 字符串。

    Returns:
        (is_valid, error_message)。

    Example::

        >>> validate_doi("10.1038/nature14539")
        (True, None)
        >>> validate_doi("not-a-doi")
        (False, "DOI 格式无效")
    """
    if not doi or not doi.strip():
        return False, "DOI 不能为空"

    stripped = doi.strip()

    # 移除 URL 前缀
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    if not _DOI_PATTERN.match(stripped):
        return False, f"DOI 格式无效: {stripped[:50]}"

    return True, None


def validate_doi_or_raise(doi: str) -> str:
    """验证 DOI，不合法时抛出 ValueError。

    Returns:
        清理后的 DOI（无前缀）。

    Raises:
        ValueError: DOI 格式无效。
    """
    is_valid, error = validate_doi(doi)
    if not is_valid:
        raise ValueError(f"DOI 无效: {error}")

    # 移除前缀后返回
    stripped = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if stripped.lower().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped


# ── URL 验证 ──────────────────────────────────────────────────────

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """验证 URL 是否有效。

    检查:
        - 非空
        - 协议是 http / https / ftp
        - 包含域名

    Args:
        url: URL 字符串。

    Returns:
        (is_valid, error_message)。

    Example::

        >>> validate_url("https://arxiv.org/pdf/2401.00001.pdf")
        (True, None)
        >>> validate_url("not-a-url")
        (False, "URL 缺少协议")
    """
    if not url or not url.strip():
        return False, "URL 不能为空"

    stripped = url.strip()

    if "://" not in stripped:
        return False, "URL 缺少协议（需 http:// 或 https://）"

    try:
        parsed = urlparse(stripped)
    except Exception:
        return False, "URL 解析失败"

    if parsed.scheme.lower() not in _VALID_URL_SCHEMES:
        return False, f"不支持的协议: {parsed.scheme}"

    if not parsed.netloc:
        return False, "URL 缺少域名"

    return True, None


def validate_url_or_raise(url: str) -> str:
    """验证 URL，不合法时抛出 ValueError。"""
    is_valid, error = validate_url(url)
    if not is_valid:
        raise ValueError(f"URL 无效: {error}")
    return url.strip()


# ── 文件名清理 ────────────────────────────────────────────────────

def sanitize_filename(
    name: str,
    max_len: int = 200,
    replacement: str = "_",
) -> str:
    """清理文件名，移除非法字符。

    Args:
        name:        原始文件名。
        max_len:     最大长度（截断超出部分）。
        replacement: 非法字符的替换字符。

    Returns:
        安全的文件名字符串。

    Example::

        >>> sanitize_filename('paper: "test" <draft>.pdf')
        'paper_ _test_ _draft_.pdf'
    """
    if not name:
        return "untitled"

    # 替换非法字符
    sanitized = _ILLEGAL_FILENAME_CHARS.sub(replacement, name)

    # 合并连续替换符
    sanitized = re.sub(
        re.escape(replacement) + r"{2,}",
        replacement,
        sanitized,
    )

    # 清理扩展名前的替换符: "name_.pdf" → "name.pdf"
    sanitized = re.sub(r"_(\.\w+)$", r"\1", sanitized)

    # 去除首尾空白和替换符
    sanitized = sanitized.strip(f" {replacement}.")

    # 空字符串
    if not sanitized:
        return "untitled"

    # 截断（保持扩展名）
    if len(sanitized) > max_len:
        # 如果有扩展名，保留之
        dot_idx = sanitized.rfind(".")
        if dot_idx > max_len // 2:
            ext = sanitized[dot_idx:]
            stem = sanitized[: dot_idx]
            stem = stem[: max_len - len(ext)]
            sanitized = stem + ext
        else:
            sanitized = sanitized[:max_len]

    return sanitized


# ── 通用检查 ──────────────────────────────────────────────────────

def is_probable_title(text: str) -> bool:
    """启发式判断文本是否更像论文标题（而非关键词查询）。

    特征:
        - 包含大写字母开头的单词
        - 包含冒号或破折号
        - 长度适中

    Args:
        text: 待判断文本。

    Returns:
        True 表示可能是论文标题。
    """
    if not text or len(text) < 5:
        return False

    words = text.split()
    if len(words) < 2:
        return False

    # 有冒号分隔（标题: 副标题）的典型模式
    if ":" in text and len(words) > 3:
        return True

    # 有大写单词
    capital_words = sum(1 for w in words if w and w[0].isupper())
    if capital_words >= len(words) * 0.5:
        return True

    return False
