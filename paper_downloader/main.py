"""
paper_downloader 命令行快速测试入口.

用法:
    python paper_downloader/main.py                              # 下载内置的 10 篇测试论文
    python paper_downloader/main.py --title "Central moments of belief information"   # 自定义标题
    python paper_downloader/main.py --file titles.txt            # 批量下载
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paper_downloader.src.interface import download_pdf, batch_download_pdf


# 内置回归测试集：覆盖不同年代以及量子计算、天文学、信号处理、
# 网络科学、粒子物理、自然语言处理、生成模型、计算机视觉等领域。
# 这些论文均可通过 arXiv 检索，适合验证“搜索 -> 匹配 -> 下载”完整流程。
TEST_PAPERS = [
    {
        "year": 1995,
        "field": "量子计算",
        "title": "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer",
    },
    {
        "year": 1998,
        "field": "天文学/宇宙学",
        "title": "Observational Evidence from Supernovae for an Accelerating Universe and a Cosmological Constant",
    },
    {
        "year": 2006,
        "field": "应用数学/信号处理",
        "title": "Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information",
    },
    {
        "year": 2009,
        "field": "网络科学",
        "title": "Community Detection in Graphs",
    },
    {
        "year": 2012,
        "field": "粒子物理",
        "title": "Observation of a New Particle in the Search for the Standard Model Higgs Boson with the ATLAS Detector at the LHC",
    },
    {
        "year": 2013,
        "field": "自然语言处理",
        "title": "Efficient Estimation of Word Representations in Vector Space",
    },
    {
        "year": 2014,
        "field": "生成式人工智能",
        "title": "Generative Adversarial Networks",
    },
    {
        "year": 2015,
        "field": "计算机视觉",
        "title": "Deep Residual Learning for Image Recognition",
    },
    {
        "year": 2016,
        "field": "引力波天文学",
        "title": "Observation of Gravitational Waves from a Binary Black Hole Merger",
    },
    {
        "year": 2017,
        "field": "深度学习/机器翻译",
        "title": "Attention Is All You Need",
    },
]


def run_batch(titles, output_dir, engine):
    """运行批量下载并输出统一的测试摘要。"""
    print(f"批量下载 {len(titles)} 篇论文...")
    results = batch_download_pdf(titles, output_dir=output_dir, engine=engine)
    ok = sum(1 for result in results if result["success"])
    fail = len(results) - ok
    print(f"\n{'=' * 60}")
    print(f"结果: {ok} 成功, {fail} 失败")
    for index, result in enumerate(results, 1):
        status = "OK" if result["success"] else "FAIL"
        title = (
            result["paper_info"]["title"][:60]
            if result["paper_info"]
            else titles[index - 1][:60]
        )
        print(f"  [{status}] {title}")
        if not result["success"]:
            print(f"         {(result['error'] or '未知错误')[:100]}")


def main():
    parser = argparse.ArgumentParser(description="论文下载器 - 快速测试")
    parser.add_argument("-t", "--title", type=str,
                        default=None,
                        help="论文标题；不指定时下载内置的 10 篇测试论文")
    parser.add_argument("-o", "--output", type=str, default="./papers",
                        help="输出目录 (默认: ./papers)")
    parser.add_argument("-e", "--engine", type=str, default="auto",
                        choices=["auto", "arxiv", "openalex", "semantic_scholar", "crossref", "scholar"],
                        help="搜索引擎 (默认: auto)")
    parser.add_argument("-f", "--file", type=str,
                        help="包含标题列表的文本文件（每行一篇）")
    args = parser.parse_args()

    if args.file:
        # 批量模式
        with open(args.file, "r", encoding="utf-8") as f:
            titles = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        run_batch(titles, args.output, args.engine)
    elif args.title:
        # 单篇模式
        print(f"搜索引擎: {args.engine}")
        print(f"论文标题: {args.title}")
        print(f"输出目录: {args.output}")
        print(f"{'='*60}")

        result = download_pdf(args.title, output_dir=args.output, engine=args.engine, timeout=60)

        if result["success"]:
            info = result["paper_info"]
            print(f"状态:    成功")
            print(f"标题:    {info['title']}")
            print(f"作者:    {', '.join(info['authors'][:3])}")
            print(f"年份:    {info.get('year', 'N/A')}")
            print(f"DOI:     {info.get('doi', 'N/A')}")
            print(f"来源:    {result['engine_used']}")
            print(f"PDF:     {result['file_path']}")
        else:
            print(f"状态:    失败")
            print(f"原因:    {result['error']}")
    else:
        # 无参数时运行内置的跨领域、跨年份下载测试。
        print("内置测试论文:")
        for index, paper in enumerate(TEST_PAPERS, 1):
            print(f"  {index:>2}. [{paper['year']}] {paper['field']}: {paper['title']}")
        print(f"搜索引擎: {args.engine}")
        print(f"输出目录: {args.output}")
        print(f"{'=' * 60}")
        run_batch(
            [paper["title"] for paper in TEST_PAPERS],
            args.output,
            args.engine,
        )


if __name__ == "__main__":
    main()
