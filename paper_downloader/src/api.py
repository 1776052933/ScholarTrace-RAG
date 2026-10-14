"""
api.py — 对外暴露的 API 函数

提供 paper_downloader 模块的最简调用接口，
适合作为第三方库被其他 AI / 科研项目直接调用。

每个函数均为模块级函数，无需实例化 PaperDownloader。

Usage::

    import paper_downloader.src.api as pd_api

    # 下载单篇论文
    paper = pd_api.download_paper("Attention Is All You Need")

    # 批量下载
    papers = pd_api.download_papers([
        "GPT-4 Technical Report",
        "BERT: Pre-training of Deep Bidirectional Transformers",
    ])

    # 搜索
    results = pd_api.search_papers("diffusion models", max_results=10)

    # 获取论文信息
    info = pd_api.get_paper_info("10.1038/nature14539")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import (
    PaperDownloaderError,
    PaperNotFoundError,
    DownloadError,
)

# ── 模块级缓存 ────────────────────────────────────────────────────

_downloader: Optional[PaperDownloader] = None


def _get_downloader(**kwargs: Any) -> PaperDownloader:
    """获取或创建下载器实例（模块级单例）。"""
    global _downloader
    if _downloader is None or kwargs:
        _downloader = PaperDownloader(**kwargs)
    return _downloader


def reset_downloader() -> None:
    """重置模块级下载器缓存。

    用于切换配置后重新初始化。
    """
    global _downloader
    if _downloader is not None:
        _downloader.__exit__()
    _downloader = None


# ── API 函数 ──────────────────────────────────────────────────────

def download_paper(
    title: str,
    output_dir: str = "papers",
    engines: Optional[List[str]] = None,
    rename: bool = True,
    max_results: int = 3,
    progress_callback: Optional[Callable[[Paper], None]] = None,
    **kwargs: Any,
) -> Paper:
    """下载单篇论文 PDF。

    自动搜索最匹配的论文并下载其 PDF 到本地。
    这是最常用的接口。

    Args:
        title:             论文标题。
        output_dir:        PDF 输出目录，默认 "papers"。
        engines:           搜索引擎列表，默认使用配置中的引擎。
        rename:            是否自动按规范重命名 PDF。
        max_results:       搜索候选数（取最佳匹配）。
        progress_callback: 下载进度回调函数，签名为 (Paper) -> None。
        **kwargs:          传递给搜索的额外参数。

    Returns:
        下载成功的 Paper 对象（包含 pdf_path）。

    Raises:
        PaperNotFoundError: 搜索无结果。
        DownloadError:      下载失败。

    Example::

        >>> paper = download_paper("Attention Is All You Need")
        >>> print(paper.pdf_path)
        'papers/Vaswani_2017_Attention_Is_All_You_Need.pdf'
    """
    dl = _get_downloader(engines=engines, **kwargs)

    if progress_callback:
        dl.set_progress_callback(progress_callback)

    return dl.download_by_title(
        title=title,
        output_dir=output_dir,
        max_results=max_results,
        engines=engines,
        rename=rename,
        **kwargs,
    )


def download_papers(
    titles: List[str],
    output_dir: str = "papers",
    engines: Optional[List[str]] = None,
    rename: bool = True,
    max_results: int = 3,
    progress_callback: Optional[Callable[[Paper], None]] = None,
    **kwargs: Any,
) -> List[Paper]:
    """批量下载多篇论文 PDF。

    Args:
        titles:            论文标题列表。
        output_dir:        PDF 输出目录。
        engines:           搜索引擎列表。
        rename:            是否重命名。
        max_results:       每个标题的搜索候选数。
        progress_callback: 进度回调。
        **kwargs:          额外参数。

    Returns:
        Paper 对象列表（包含成功和失败的）。

    Example::

        >>> papers = download_papers([
        ...     "GPT-4 Technical Report",
        ...     "BERT: Pre-training of Deep Bidirectional Transformers",
        ... ])
        >>> for p in papers:
        ...     print(f"{p.title}: {'OK' if p.has_pdf else 'FAIL'}")
    """
    dl = _get_downloader(engines=engines, **kwargs)

    if progress_callback:
        dl.set_progress_callback(progress_callback)

    return dl.batch_download(
        titles=titles,
        output_dir=output_dir,
        max_results=max_results,
        engines=engines,
        rename=rename,
        **kwargs,
    )


def search_papers(
    query: str,
    max_results: int = 10,
    engines: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[Paper]:
    """搜索论文，返回 Paper 对象列表。

    Args:
        query:       搜索关键词或标题。
        max_results: 最大返回数量。
        engines:     搜索引擎列表，None 使用默认。
        **kwargs:    额外搜索参数。

    Returns:
        Paper 对象列表。

    Example::

        >>> papers = search_papers("diffusion models", max_results=5)
        >>> for p in papers:
        ...     print(f"{p.citation}: {p.title}")
    """
    dl = _get_downloader(engines=engines, **kwargs)
    return dl.search(query, max_results=max_results, engines=engines, **kwargs)


def get_paper_info(identifier: str, **kwargs: Any) -> Optional[Paper]:
    """根据 DOI 或 arXiv ID 获取论文详细信息。

    Args:
        identifier: DOI（如 "10.1038/nature14539"）或
                    arXiv ID（如 "2401.00001"）。
        **kwargs:   额外参数。

    Returns:
        Paper 对象，未找到返回 None。

    Example::

        >>> paper = get_paper_info("10.1038/nature14539")
        >>> if paper:
        ...     print(paper.title, paper.citation_count)
    """
    dl = _get_downloader(**kwargs)
    return dl.get_paper_info(identifier)


def set_config(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """设置/更新模块级配置。

    调用后将重置下载器缓存，使用新配置。

    Args:
        config_path: YAML 配置文件路径。
        config:      配置字典。
        **kwargs:    覆盖特定配置项。

    Example::

        set_config(engines=["arxiv"], max_downloads=5)
    """
    global _downloader
    _downloader = PaperDownloader(
        config_path=config_path,
        config=config,
        **kwargs,
    )
