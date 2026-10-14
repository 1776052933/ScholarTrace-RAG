"""
interface.py — 统一极简下载接口

为 AI 项目和其他第三方调用者提供最简洁的论文下载入口。
所有函数返回纯字典，无需了解内部 Paper 模型。

Usage::

    from paper_downloader.src.interface import download_pdf, batch_download_pdf, search_papers

    # 下载单篇
    result = download_pdf("Attention Is All You Need")
    if result["success"]:
        print(result["file_path"])

    # 批量下载
    results = batch_download_pdf(["GPT-4", "BERT", "ResNet"])
    for r in results:
        print(r["success"], r["file_path"])

    # 搜索
    papers = search_papers("diffusion models", max_results=5)
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
)
from paper_downloader.src.utils.pdf_identity import pdf_matches_title


# ── 全局单例 ──────────────────────────────────────────────────────

_downloaders: dict[tuple[str, ...], PaperDownloader] = {}
# 兼容已有调用方和测试注入；生产下载默认使用独立实例，避免 Top-K 并发时共享任务队列。
_downloader: Optional[PaperDownloader] = None

PaperInput = Union[str, Mapping[str, Any], Paper]


def _get_downloader(engine: str = "auto", timeout: int = 30) -> PaperDownloader:
    """获取或创建全局下载器实例。"""
    if _downloader is not None:
        return _configure_downloader(_downloader, timeout)
    engines = _resolve_engines(engine)
    key = tuple(engines)
    if key not in _downloaders:
        _downloaders[key] = PaperDownloader(
            engines=engines,
            max_downloads=3,
        )
    return _configure_downloader(_downloaders[key], timeout)


def _new_downloader(engine: str, timeout: int) -> PaperDownloader:
    """为一次下载创建隔离实例，避免并发任务互相污染状态。"""
    if _downloader is not None:
        return _configure_downloader(_downloader, timeout)
    return _configure_downloader(
        PaperDownloader(engines=_resolve_engines(engine), max_downloads=1),
        timeout,
    )


def _configure_downloader(downloader: PaperDownloader, timeout: int) -> PaperDownloader:
    _apply_runtime_api_settings(downloader)
    downloader._config.setdefault("timeout", {})["search"] = timeout
    downloader._config.setdefault("timeout", {})["download"] = timeout * 2
    return downloader


def _resolve_engines(engine: str) -> List[str]:
    """解析引擎选择到引擎列表。

    Args:
        engine: "arxiv" | "openalex" | "crossref" | "scholar" |
                "academic_oa" | "auto"

    Returns:
        引擎名称列表。auto 按优先级尝试所有已注册引擎。
    """
    engine_map = {
        "arxiv":   ["arxiv"],
        "crossref": ["crossref"],
        "scholar": ["google_scholar"],
        "openalex": ["openalex"],
        # Competition-friendly mode: stable academic APIs plus OA PDF sources,
        # without Google Scholar's optional dependency and scraping limits.
        "academic_oa": ["arxiv", "openalex", "crossref"],
        "auto":    ["arxiv", "openalex", "crossref", "google_scholar"],
    }
    return engine_map.get(engine, engine_map["auto"])


def _apply_runtime_api_settings(downloader: PaperDownloader) -> None:
    """Load optional API settings without requiring them for local tests."""
    contact_email = (
        _read_key("contact_email")
        or os.environ.get("OPENALEX_EMAIL", "")
        or os.environ.get("CONTACT_EMAIL", "")
    )
    if contact_email:
        downloader._config["contact_email"] = contact_email
    openalex_api_key = (
        os.environ.get("OPENALEX_API_KEY", "").strip()
        or _read_key("openalex").strip()
        or _read_key("openalex_api_key").strip()
    )
    if openalex_api_key:
        downloader._config["openalex_api_key"] = openalex_api_key


def _read_key(name: str) -> str:
    try:
        import json

        key_path = Path(__file__).resolve().parents[2] / "API_key" / "API_key.json"
        if not key_path.exists():
            return ""
        data = json.loads(key_path.read_text(encoding="utf-8"))
        return str(data.get(name) or "")
    except Exception:
        return ""


# ── 公共 API ──────────────────────────────────────────────────────

def download_pdf(
    title: PaperInput,
    output_dir: str = "./papers",
    engine: str = "auto",
    timeout: int = 30,
    rename: bool = True,
    callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """优先根据精确标识下载 PDF，缺失时再按标题多源搜索。

    最简调用::

        result = download_pdf("Attention Is All You Need")
        if result["success"]:
            print(result["file_path"])

    Args:
        title:      论文标题、Paper 对象或包含 title/doi/arxiv_id/pdf_url/url 的字典。
        output_dir: PDF 输出目录，默认 "./papers"。
        engine:     搜索引擎，"arxiv" | "openalex" | "crossref" |
                    "scholar" | "auto"。
                    默认 "auto" 按优先级自动尝试。
        timeout:    超时秒数，默认 30。
        rename:     是否按元数据重命名 PDF，默认 True。
        callback:   进度回调，签名 (progress: float, message: str) -> None。
                    progress 范围 0.0 ~ 1.0。

    Returns:
        Dict: {
            "success":      bool,        # 是否成功
            "file_path":    str | None,  # 下载的 PDF 路径
            "paper_info":   dict | None,  # 论文元数据
            "error":        str | None,  # 错误信息
            "engine_used":  str | None,  # 实际使用的搜索引擎
        }
    """
    result: Dict[str, Any] = {
        "success": False,
        "file_path": None,
        "paper_info": None,
        "error": None,
        "engine_used": None,
    }

    paper = _paper_from_input(title)
    requested_title = paper.title
    if not paper.title.strip():
        result["error"] = "标题不能为空"
        return result

    _report(callback, 0.0, f"准备下载: {paper.title[:60]}")

    errors: list[str] = []
    if _has_exact_locator(paper):
        _report(callback, 0.1, "使用已有 URL/DOI/arXiv ID 直接解析 PDF")
        direct_downloader: Optional[PaperDownloader] = None
        try:
            direct_downloader = _new_downloader(engine, timeout)
            downloaded = direct_downloader.download(
                paper,
                output_dir=output_dir,
                rename=rename,
            )
            if downloaded and downloaded[0].has_pdf:
                resolved = downloaded[0]
                if pdf_matches_title(resolved.pdf_path, requested_title) is False:
                    raise DownloadError(
                        "downloaded PDF title does not match the requested paper"
                    )
                _fill_success(result, resolved, _paper_source(resolved))
                _report(callback, 1.0, f"直链下载完成: {resolved.pdf_path}")
                return result
            errors.append("[metadata] 已有标识未解析到有效 PDF")
        except Exception as exc:
            errors.append(f"[metadata] 直链/标识下载失败: {exc}")
        finally:
            _release_downloader(direct_downloader)

    # 精确定位不可用或失败时，才回退到标题搜索。
    engines = _resolve_engines(engine)

    for eng_name in engines:
        dl: Optional[PaperDownloader] = None
        try:
            dl = _new_downloader(eng_name, timeout)
            _report(callback, 0.2, f"使用 {eng_name} 搜索中...")

            papers = dl.search(requested_title, max_results=3, engines=[eng_name])
            if not papers:
                errors.append(f"[{eng_name}] 未找到结果")
                continue

            _report(callback, 0.4, f"找到论文，匹配标题...")

            # 按标题相似度选最佳匹配
            best_paper, score = dl._find_best_match(
                requested_title,
                papers,
                min_similarity=0.82,
                strict=True,
            )
            if best_paper is None:
                closest = papers[0].title[:80] if papers else "N/A"
                errors.append(
                    f"[{eng_name}] 标题不匹配 (相似度 {score:.0%}): 最近似 \"{closest}\""
                )
                continue

            paper = best_paper
            used_engine = paper.source or eng_name
            _report(callback, 0.6, f"匹配 (相似度 {score:.0%}): {paper.title[:50]}")

            # 下载
            results = dl.download(paper, output_dir=output_dir, rename=rename)

            if results and results[0].has_pdf:
                p = results[0]
                if pdf_matches_title(p.pdf_path, requested_title) is False:
                    raise DownloadError(
                        "downloaded PDF title does not match the requested paper"
                    )
                _fill_success(result, p, used_engine)

                _report(callback, 1.0, f"下载完成: {p.pdf_path}")
                return result

            # PDF 下载失败但论文存在
            errors.append(
                f"[{eng_name}] 找到论文但 PDF 下载失败\n"
                f"  论文: {paper.title[:80]}\n"
                f"  DOI: {paper.doi or 'N/A'}  来源: {paper.source or eng_name}"
            )
            continue

        except PaperNotFoundError:
            errors.append(f"[{eng_name}] 未找到匹配的论文")
            continue
        except Exception as exc:
            errors.append(f"[{eng_name}] 错误: {exc}")
            continue
        finally:
            _release_downloader(dl)

    # 构建综合错误信息
    if errors:
        result["error"] = (
            f"所有 {len(errors)} 个引擎均失败:\n" +
            "\n".join(f"  {e.split(chr(10))[0]}" for e in errors)
        )
    else:
        result["error"] = "所有搜索引擎均未找到该论文"
    _report(callback, 1.0, result["error"])
    return result


def batch_download_pdf(
    titles: List[PaperInput],
    output_dir: str = "./papers",
    engine: str = "auto",
    max_concurrent: int = 3,
    skip_existing: bool = True,
    callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """批量下载多篇论文 PDF。

    Args:
        titles:         论文标题列表。
        output_dir:     输出目录。
        engine:         搜索引擎，"auto" 自动选择。
        max_concurrent: 最大并发数，默认 3。
        skip_existing:  跳过输出目录中已有的 PDF，默认 True。
        callback:       进度回调，签名 (current: int, total: int, message: str) -> None。

    Returns:
        List[Dict]: 每篇论文的结果（与 download_pdf 返回格式相同）。
    """
    if not titles:
        return []

    total = len(titles)
    results: List[Dict[str, Any]] = [{
        "success": False, "file_path": None,
        "paper_info": None, "error": "未处理", "engine_used": None,
    } for _ in range(total)]

    _report_batch(callback, 0, total, f"批量下载 {total} 篇...")

    # 预扫描已有 PDF
    existing_stems: set = set()
    if skip_existing:
        out = Path(output_dir)
        if out.exists():
            existing_stems = {p.stem.lower() for p in out.glob("*.pdf")}

    def _download_one(idx: int, title: PaperInput) -> None:
        """下载单篇并写入结果。"""
        item = _paper_from_input(title)
        if skip_existing:
            stem = re.sub(r'[<>:"/\\|?*]', '_', item.title[:50]).strip().lower()
            if any(stem in ex for ex in existing_stems):
                results[idx]["success"] = True
                results[idx]["error"] = "已存在（跳过）"
                _report_batch(callback, idx + 1, total, f"跳过: {item.title[:50]}")
                return

        r = download_pdf(
            title=title,
            output_dir=output_dir,
            engine=engine,
            rename=True,
        )
        results[idx] = r
        status = "OK" if r["success"] else "FAIL"
        _report_batch(callback, idx + 1, total, f"[{status}] {item.title[:50]}")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(_download_one, i, t): i
            for i, t in enumerate(titles)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass  # 已在线程内处理

    _report_batch(callback, total, total, "批量下载完成")
    return results


def search_papers(
    query: str,
    max_results: int = 5,
    engine: str = "auto",
) -> List[Dict[str, Any]]:
    """搜索论文（仅获取信息，不下载 PDF）。

    Args:
        query:       搜索关键词或标题。
        max_results: 最大返回数，默认 5。
        engine:      搜索引擎，"auto" 自动选择。

    Returns:
        List[Dict]: 论文信息列表，每项包含:
            - title, authors, year, abstract
            - doi, arxiv_id
            - pdf_url, url
            - source, citation_count, journal
    """
    dl = _get_downloader(engine)
    engines = _resolve_engines(engine)

    try:
        papers = dl.search(query, max_results=max_results, engines=engines)
        return [_paper_to_dict(p) for p in papers]
    except Exception:
        return []


# ── 工具 ──────────────────────────────────────────────────────────

def _paper_from_input(value: PaperInput) -> Paper:
    """把标题或跨模块元数据整理为下载器的 Paper 模型。"""
    if isinstance(value, Paper):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("http://", "https://")):
            data: Dict[str, Any] = {"title": text, "paper_id": text, "url": text}
        else:
            data = {"title": text}
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        return Paper()

    paper_id = str(data.get("paper_id") or "").strip()
    url = str(data.get("url") or "").strip()
    if not url and paper_id.startswith(("http://", "https://")):
        url = paper_id

    doi = _normalize_doi(data.get("doi")) or _extract_doi(paper_id) or _extract_doi(url)
    arxiv_id = (
        _normalize_arxiv_id(data.get("arxiv_id"))
        or _extract_arxiv_id(paper_id)
        or _extract_arxiv_id(url)
    )
    pdf_url = str(data.get("pdf_url") or "").strip()
    if not pdf_url:
        pdf_url = next(
            (candidate for candidate in (paper_id, url) if _looks_like_pdf_url(candidate)),
            "",
        )
    authors = data.get("authors") or []
    if isinstance(authors, str):
        authors = [item.strip() for item in re.split(r"[;,]", authors) if item.strip()]
    title = str(data.get("title") or "").strip()
    return Paper(
        title=title or paper_id or url,
        authors=[str(item).strip() for item in authors if str(item).strip()],
        year=str(data.get("year")) if data.get("year") else None,
        abstract=str(data.get("abstract") or "") or None,
        doi=doi,
        arxiv_id=arxiv_id,
        pdf_url=pdf_url or None,
        url=url or None,
        source=str(data.get("source") or _infer_source(pdf_url or url or paper_id) or "") or None,
        citation_count=int(data.get("citation_count") or 0),
        journal=str(data.get("venue") or data.get("journal") or "") or None,
    )


def _has_exact_locator(paper: Paper) -> bool:
    return bool(paper.pdf_url or paper.arxiv_id or paper.doi)


def _fill_success(result: Dict[str, Any], paper: Paper, engine_used: str) -> None:
    result["success"] = True
    result["file_path"] = paper.pdf_path
    result["paper_info"] = _paper_to_dict(paper)
    result["engine_used"] = engine_used
    result["error"] = None


def _paper_source(paper: Paper) -> str:
    return paper.source or _infer_source(paper.pdf_url or paper.url or "") or "direct"


def _release_downloader(downloader: Optional[PaperDownloader]) -> None:
    if downloader is None or downloader is _downloader:
        return
    try:
        downloader.__exit__()
    except Exception:
        logging.getLogger(__name__).debug("关闭临时下载器失败", exc_info=True)


def _normalize_doi(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I)


def _extract_doi(value: str) -> Optional[str]:
    match = re.search(r"(?:doi\.org/|\bdoi:\s*|^)(10\.\d{4,9}/[^?#\s]+)", value, re.I)
    return match.group(1).rstrip(".,;)") if match else None


def _normalize_arxiv_id(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    clean = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value.strip(), flags=re.I)
    clean = clean.removesuffix(".pdf")
    return clean if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", clean, re.I) else None


def _extract_arxiv_id(value: str) -> Optional[str]:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", value, re.I)
    return match.group(1) if match else None


def _looks_like_pdf_url(value: str) -> bool:
    lowered = value.casefold()
    return bool(value) and (
        lowered.endswith(".pdf")
        or "/pdf/" in lowered
        or "/article/download/" in lowered
        or "pdf=render" in lowered
        or "type=pdf" in lowered
    )


def _infer_source(value: str) -> str:
    lowered = value.casefold()
    if "arxiv.org" in lowered:
        return "arxiv"
    if "openalex.org" in lowered:
        return "openalex"
    if "semanticscholar.org" in lowered:
        return "semantic_scholar"
    if "europepmc.org" in lowered:
        return "europe_pmc"
    return "direct" if _looks_like_pdf_url(value) else ""


def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """将 Paper 对象转为纯字典。"""
    return {
        "title":          paper.title,
        "authors":        paper.authors,
        "first_author":   paper.first_author,
        "year":           paper.year,
        "abstract":       paper.abstract,
        "doi":            paper.doi,
        "arxiv_id":       paper.arxiv_id,
        "pdf_url":        paper.pdf_url,
        "url":            paper.url,
        "source":         paper.source,
        "citation_count": paper.citation_count,
        "journal":        paper.journal,
        "file_path":      paper.pdf_path,
        "file_size":      paper.file_size,
        "downloaded_at":  paper.downloaded_at,
    }


def _report(
    callback: Optional[Callable[[float, str], None]],
    progress: float,
    message: str,
) -> None:
    """调用单篇进度回调（安全）。"""
    if callback is None:
        return
    try:
        callback(progress, message)
    except Exception:
        pass


def _report_batch(
    callback: Optional[Callable[[int, int, str], None]],
    current: int,
    total: int,
    message: str,
) -> None:
    """调用批量进度回调（安全）。"""
    if callback is None:
        return
    try:
        callback(current, total, message)
    except Exception:
        pass
