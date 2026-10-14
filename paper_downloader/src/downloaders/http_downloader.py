"""
http_downloader.py — 通用 HTTP 下载器

基于 ``requests`` 实现，支持:
    - 断点续传（HTTP Range 请求）
    - tqdm 进度条
    - 自动跟随重定向
    - 代理支持
    - 下载超时与重试
"""

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _tqdm = None  # type: ignore[assignment]
    _TQDM_AVAILABLE = False

from paper_downloader.src.downloaders.base_downloader import Downloader


# 默认 User-Agent
_DEFAULT_UA = "AutoPaper/0.1 (academic research tool; mailto:auto-paper@example.com)"

# 常见 PDF Content-Type
_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}


class HTTPDownloader(Downloader):
    """通用 HTTP PDF 下载器。

    支持断点续传、进度条、代理、自动重试。

    Usage::

        dl = HTTPDownloader(config)
        path = dl.download("https://example.org/paper.pdf", save_path="./papers")
    """

    DOWNLOADER_NAME = "http"

    # 分块下载大小
    CHUNK_SIZE = 8192

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._session = self._build_session()

    # ── HTTP 会话 ─────────────────────────────────────────────────

    def _build_session(self) -> requests.Session:
        """创建配置好的 requests Session（带重试和代理）。"""
        session = requests.Session()

        # 重试策略
        retry_cfg = self.config.get("retry", {})
        retries = Retry(
            total=retry_cfg.get("max_attempts", 3),
            backoff_factor=retry_cfg.get("backoff_factor", 2),
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # 代理
        proxies = self._get_proxies()
        if proxies:
            session.proxies.update(proxies)

        # 默认头
        session.headers.update({"User-Agent": _DEFAULT_UA})

        return session

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """从配置构建代理字典。"""
        proxy_cfg = self.config.get("proxy", {})
        if not proxy_cfg.get("enabled", False):
            return None
        proxies = {}
        for proto in ("http", "https"):
            url = proxy_cfg.get(proto, "")
            if url:
                user = proxy_cfg.get("username", "")
                pwd = proxy_cfg.get("password", "")
                if user and pwd and "://" in url:
                    scheme, rest = url.split("://", 1)
                    url = f"{scheme}://{user}:{pwd}@{rest}"
                proxies[proto] = url
        return proxies or None

    # ── 下载主逻辑 ────────────────────────────────────────────────

    def download(
        self,
        url: str,
        save_path: str | Path = "./papers",
        filename: Optional[str] = None,
        resume: bool = True,
        **kwargs: Any,
    ) -> Optional[Path]:
        """下载 PDF 文件。

        Args:
            url:       PDF URL。
            save_path: 保存目录。
            filename:  自定义文件名（不含扩展名）。
            resume:    是否启用断点续传。默认 True。
            **kwargs:
                timeout:     下载超时秒数（覆盖配置）。
                show_progress: 是否显示进度条，默认 True。

        Returns:
            下载成功返回文件 Path，失败返回 None。
        """
        save_dir = Path(save_path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        # 确定目标文件名
        if filename:
            target_name = filename
            if not target_name.endswith(".pdf"):
                target_name += ".pdf"
            # 防御性清理 — 移除 Windows 非法字符
            target_name = self._safe_filename(target_name)
        else:
            target_name = self._extract_filename_from_url(url)

        target_path = save_dir / target_name
        partial_path = target_path.with_suffix(target_path.suffix + ".part")
        self.logger.info("下载: %s → %s", url[:100], target_path)

        if target_path.exists() and self.validate_pdf(target_path):
            self.logger.info("有效 PDF 已存在，跳过重复下载: %s", target_path)
            return target_path

        timeout = kwargs.get("timeout", self._get_timeout("download"))

        # 断点续传：检查已有部分文件
        headers: Dict[str, str] = {}
        downloaded_bytes = 0
        if resume and partial_path.exists():
            downloaded_bytes = partial_path.stat().st_size
            if downloaded_bytes > 0:
                headers["Range"] = f"bytes={downloaded_bytes}-"
                self.logger.info("断点续传，从 %d bytes 处继续", downloaded_bytes)

        try:
            response = self._session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()

            # 检测是否真的是 PDF（如果服务器给了 Content-Type）
            content_type = response.headers.get("Content-Type", "").lower()
            if any(ct in content_type for ct in _PDF_CONTENT_TYPES):
                pass  # 确认是 PDF 类型
            elif content_type and "text/html" in content_type:
                self.logger.warning("服务器返回 HTML 而非 PDF，跳过: %s", url[:100])
                partial_path.unlink(missing_ok=True)
                return None  # 不保存 HTML 文件，立即返回失败

            # 确定总大小
            total_size = self._get_content_length(response)
            if downloaded_bytes > 0 and response.status_code == 206:
                total_size += downloaded_bytes

            # 写入模式：续传时追加，否则覆盖
            write_mode = "ab" if downloaded_bytes > 0 and response.status_code == 206 else "wb"

            show_progress = kwargs.get("show_progress", True)
            progress_desc = target_name[:40] + (".." if len(target_name) > 40 else "")

            with open(partial_path, write_mode) as f:
                if show_progress and _TQDM_AVAILABLE and total_size:
                    with _tqdm(
                        total=total_size,
                        initial=downloaded_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=progress_desc,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            # 校验
            if not self.validate_pdf(partial_path):
                self.logger.warning("下载的文件不是有效 PDF: %s", partial_path)
                partial_path.unlink(missing_ok=True)
                return None  # 立即返回失败，让上层处理

            partial_path.replace(target_path)
            self.logger.info("下载完成: %s (%.1f KB)", target_path, target_path.stat().st_size / 1024)
            return target_path

        except requests.exceptions.Timeout:
            self.logger.error("下载超时 (%ds): %s", timeout, url[:100])
            if not resume:
                partial_path.unlink(missing_ok=True)
            return None
        except requests.exceptions.HTTPError as exc:
            self.logger.error("HTTP 错误 %s: %s", exc.response.status_code if exc.response else "?", url[:100])
            if not resume:
                partial_path.unlink(missing_ok=True)
            return None
        except requests.exceptions.RequestException as exc:
            self.logger.error("请求异常: %s — URL: %s", exc, url[:100])
            if not resume:
                partial_path.unlink(missing_ok=True)
            return None
        except OSError as exc:
            self.logger.error("文件写入失败: %s", exc)
            return None

    # ── HEAD 预检 ─────────────────────────────────────────────────

    def check_url(self, url: str) -> Optional[Dict[str, Any]]:
        """通过 HEAD 请求检查 URL 是否可达，返回文件信息。

        Args:
            url: 目标 URL。

        Returns:
            dict 包含 content_length、content_type、filename，不可达返回 None。
        """
        try:
            response = self._session.head(
                url,
                timeout=self._get_timeout("connection"),
                allow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            # 从 Content-Disposition 提取文件名
            disposition = response.headers.get("Content-Disposition", "")
            filename = None
            if "filename=" in disposition:
                import re
                match = re.search(r'filename[^;=\n]*=["\']?([^"\';\n]*)', disposition)
                if match:
                    filename = match.group(1).strip()

            return {
                "content_length": int(response.headers.get("Content-Length", 0)),
                "content_type": content_type,
                "filename": filename,
                "url": response.url,  # 最终 URL（跟随重定向后）
                "is_pdf": any(ct in content_type.lower() for ct in _PDF_CONTENT_TYPES),
            }
        except requests.exceptions.RequestException as exc:
            self.logger.warning("HEAD 请求失败: %s — %s", url[:100], exc)
            return None

    # ── 工具 ──────────────────────────────────────────────────────

    def _get_timeout(self, key: str) -> int:
        """从配置获取超时值。"""
        return int(self.config.get("timeout", {}).get(key, 30))

    @staticmethod
    def _get_content_length(response: requests.Response) -> Optional[int]:
        """从响应头解析内容长度。"""
        cl = response.headers.get("Content-Length")
        if cl is not None:
            try:
                return int(cl)
            except (ValueError, TypeError):
                pass

        # 如果服务器返回了 Content-Range
        cr = response.headers.get("Content-Range")
        if cr:
            import re
            match = re.search(r"/(\d+)", cr)
            if match:
                return int(match.group(1))
        return None

    def close(self) -> None:
        """关闭 HTTP 会话。"""
        self._session.close()
        self.logger.debug("HTTP 会话已关闭")

    def __enter__(self) -> "HTTPDownloader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
