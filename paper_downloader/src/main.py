"""
main.py — 命令行测试入口

提供 paper_downloader 的命令行交互接口，
用于快速测试和演示下载功能。

用法::

    # 下载单篇论文
    python -m paper_downloader.src.main --title "Attention Is All You Need"

    # 指定输出目录
    python -m paper_downloader.src.main --title "GPT-4" --output ./my_papers

    # 批量下载（从文件读取标题，每行一个）
    python -m paper_downloader.src.main --file titles.txt

    # 搜索论文（不下载）
    python -m paper_downloader.src.main --search "diffusion models" --max 10

    # 搜索并下载
    python -m paper_downloader.src.main --search "BERT" --download --max 3

    # 指定搜索引擎
    python -m paper_downloader.src.main --title "Deep Learning" --engines arxiv crossref

    # 查询论文信息
    python -m paper_downloader.src.main --info "10.1038/nature14539"

    # 使用自定义配置
    python -m paper_downloader.src.main --title "Attention" --config my_config.yaml

    # 详细输出
    python -m paper_downloader.src.main --title "GPT" --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from paper_downloader.src.core.downloader import PaperDownloader
from paper_downloader.src.models.paper import Paper
from paper_downloader.src.exceptions import PaperDownloaderError


# ═══════════════════════════════════════════════════════════════════
# CLI 核心
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="paper-downloader",
        description="论文自动下载器 — 搜索并下载学术论文 PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --title "Attention Is All You Need"
  %(prog)s --title "GPT-4 Technical Report" --output ./my_papers
  %(prog)s --file titles.txt
  %(prog)s --search "diffusion models" --max 10
  %(prog)s --search "BERT" --download --max 3
  %(prog)s --info "10.1038/nature14539"
        """,
    )

    # 互斥操作组
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "-t", "--title",
        type=str,
        help="论文标题（单篇下载）",
    )
    action_group.add_argument(
        "-f", "--file",
        type=str,
        help="包含论文标题列表的文本文件（每行一篇）",
    )
    action_group.add_argument(
        "-s", "--search",
        type=str,
        help="搜索关键词",
    )
    action_group.add_argument(
        "--info",
        type=str,
        help="根据 DOI 或 arXiv ID 查询论文信息",
    )

    # 选项
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="papers",
        help="PDF 输出目录（默认: papers）",
    )
    parser.add_argument(
        "-m", "--max",
        type=int,
        default=5,
        help="单次搜索最大返回数（默认: 5）",
    )
    parser.add_argument(
        "-e", "--engines",
        nargs="+",
        default=None,
        help="搜索引擎列表（如: arxiv crossref），默认使用配置中的引擎",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "-d", "--download",
        action="store_true",
        default=False,
        help="与 --search 联用时同时下载搜索结果",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        default=False,
        help="不自动重命名下载的 PDF",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="详细输出（DEBUG 级别日志）",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="静默模式（仅输出错误）",
    )

    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """配置日志级别。"""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def read_titles_from_file(file_path: str) -> List[str]:
    """从文件读取论文标题列表（每行一篇，忽略空行和注释行）。"""
    path = Path(file_path)
    if not path.exists():
        print(f"错误: 文件不存在 — {file_path}")
        sys.exit(1)

    titles: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                titles.append(line)
    return titles


# ── 输出格式化 ──────────────────────────────────────────────────

def print_paper(paper: Paper, index: int = 0, show_abstract: bool = False) -> None:
    """格式化打印单篇论文信息。"""
    sep = "─" * 60
    if index > 0:
        print(f"\n{sep}")
        print(f"  [{index}]  {paper.citation}")
    else:
        print(f"\n{sep}")
        print(f"  {paper.citation}")

    print(f"  标题:   {paper.title}")
    if paper.authors:
        authors_str = ", ".join(paper.authors[:5])
        if len(paper.authors) > 5:
            authors_str += f" 等 ({len(paper.authors)} 人)"
        print(f"  作者:   {authors_str}")
    if paper.year:
        print(f"  年份:   {paper.year}")
    if paper.doi:
        print(f"  DOI:    {paper.doi}")
    if paper.arxiv_id:
        print(f"  arXiv:  {paper.arxiv_id}")
    if paper.journal:
        print(f"  期刊:   {paper.journal}")
    if paper.citation_count is not None:
        print(f"  引用:   {paper.citation_count}")
    if paper.pdf_path:
        print(f"  PDF:    {paper.pdf_path}")
    if paper.source:
        print(f"  来源:   {paper.source}")
    if show_abstract and paper.abstract:
        abstract_preview = paper.abstract[:300]
        if len(paper.abstract) > 300:
            abstract_preview += "..."
        print(f"  摘要:   {abstract_preview}")
    print(sep)


def print_summary(papers: List[Paper]) -> None:
    """打印批量下载的摘要。"""
    total = len(papers)
    ok = sum(1 for p in papers if p.has_pdf)
    fail = total - ok
    print(f"\n{'=' * 60}")
    print(f"  结果: {ok} 成功, {fail} 失败 (共 {total} 篇)")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    """命令行主入口。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv。

    Returns:
        退出码（0=成功，1=失败）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose, quiet=args.quiet)
    logger = logging.getLogger("main")

    # 进度回调（非静默模式）
    def on_progress(paper: Paper) -> None:
        status = "✓" if paper.has_pdf else "✗"
        print(f"  [{status}] {paper.title[:60]} → {paper.pdf_path or '失败'}")

    try:
        # 创建下载器
        dl = PaperDownloader(
            config_path=args.config,
            engines=args.engines,
        )
        dl.set_progress_callback(on_progress if not args.quiet else None)

        # ── 单篇下载 ──────────────────────────────────────────
        if args.title:
            logger.info("单篇下载: %s", args.title)
            paper = dl.download_by_title(
                title=args.title,
                output_dir=args.output,
                max_results=args.max,
                rename=not args.no_rename,
            )
            print_paper(paper, show_abstract=args.verbose)
            return 0 if paper.has_pdf else 1

        # ── 批量下载 ──────────────────────────────────────────
        if args.file:
            titles = read_titles_from_file(args.file)
            logger.info("批量下载: %d 篇论文 来自 %s", len(titles), args.file)
            print(f"从文件加载了 {len(titles)} 篇论文标题\n")

            papers = dl.batch_download(
                titles=titles,
                output_dir=args.output,
                max_results=args.max,
                rename=not args.no_rename,
            )

            for i, paper in enumerate(papers, 1):
                print_paper(paper, index=i)

            print_summary(papers)
            return 0 if any(p.has_pdf for p in papers) else 1

        # ── 搜索 ──────────────────────────────────────────────
        if args.search:
            logger.info("搜索: %s", args.search)
            papers = dl.search(args.search, max_results=args.max)

            if not args.download:
                # 仅搜索模式
                print(f"\n搜索 '{args.search}' 结果 ({len(papers)} 篇):")
                for i, paper in enumerate(papers, 1):
                    print_paper(paper, index=i, show_abstract=True)
            else:
                # 搜索并下载模式
                print(f"\n搜索 '{args.search}' 并下载 (最多 {args.max} 篇)...")
                papers = dl.download(
                    papers[:args.max],
                    output_dir=args.output,
                    rename=not args.no_rename,
                )
                for i, paper in enumerate(papers, 1):
                    print_paper(paper, index=i)
                print_summary(papers)

            return 0

        # ── 论文信息 ──────────────────────────────────────────
        if args.info:
            logger.info("查询论文信息: %s", args.info)
            paper = dl.get_paper_info(args.info)
            if paper:
                print_paper(paper, show_abstract=True)
                return 0
            else:
                print(f"未找到论文: {args.info}")
                return 1

    except PaperDownloaderError as exc:
        logger.error("%s: %s", exc.__class__.__name__, exc)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\n中断退出")
        return 130
    except Exception as exc:
        logger.error("未预期的错误: %s", exc)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════════
# 模块入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.exit(main())
