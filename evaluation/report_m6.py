"""Build a concise, reproducible M6 report from saved local evaluation runs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from .m6_review import summarize_sheet


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} s"


def build_report(qasper_dir: Path, multi_path: Path, answers_dir: Path,
                 ragas_path: Path, review_path: Path) -> str:
    qmanifest = _read(qasper_dir / "manifest.json")
    qsummary = _read(qasper_dir / "summary.json")
    qrecords = [_read(path) for path in sorted((qasper_dir / "records").glob("*.json"))]
    if qmanifest["split"] != "test" or qmanifest["limit"] != 30 or len(qrecords) != 30:
        raise ValueError("Expected the frozen QASPER test-30 run")
    multi = _read(multi_path)
    cached = _read(answers_dir / "summary.json")
    ragas = _read(ragas_path) if ragas_path.is_file() else None
    review = summarize_sheet(review_path)
    variants = qsummary["variants"]
    if any(row["question_count"] != 30 for row in variants.values()):
        raise ValueError("QASPER variant denominator differs from 30")

    lines = ["# M6 固定集评测与交付报告", "",
             "## 评测边界", "",
             "- QASPER test 固定 30 题、30 篇结构化全文；六路检索使用相同语料、切块、BGE-M3 向量与 Top-5。分数是项目自定义 token 覆盖代理，非官方 QASPER F1。",
             "- PDF 集共 10 篇、22 道单论文题；跨论文题 4 道，dev/test 各 2 道。标注来自 M3 首轮，尚待独立人工复核。",
             "- 已把跨论文 test 四处金标准 PDF 页渲染检查表格数值与页码；这是开发过程中的抽查，不代替独立人工复核。",
             "- 跨论文对照用同一个旧 Chroma 索引比较普通 Top-5 与 M5 证据补检图；它不代表 Milvus 混合检索的效果。",
             "- 以下是已保存的离线结果。M6 本轮没有重新向 DeepSeek 提交问答。",
             "", "## QASPER 留出集：六路检索消融", "",
             "| 检索方式 | 成功/30 | 证据覆盖代理 | P50 延迟 | P95 延迟 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in qmanifest["variants"]:
        row = variants[name]
        lines.append(f"| `{name}` | {row['completed']}/30 | {_pct(row['evidence_recall_proxy'])} | {_seconds(row['latency_p50_seconds'])} | {_seconds(row['latency_p95_seconds'])} |")
    dense = variants["chroma_dense"]["evidence_recall_proxy"]
    best = max(variants, key=lambda name: variants[name]["evidence_recall_proxy"] or -1)
    delta = (variants[best]["evidence_recall_proxy"] - dense) * 100
    hybrid_delta = (variants["milvus_hybrid"]["evidence_recall_proxy"] -
                    variants["milvus_dense"]["evidence_recall_proxy"]) * 100
    rerank_slowdown = (variants["milvus_hybrid_rerank"]["latency_p50_seconds"] /
                       variants["milvus_dense"]["latency_p50_seconds"])
    lines.extend(["", f"与 Chroma dense 对照，覆盖代理最高的 `{best}` 高 **{delta:+.1f} 个百分点**。这只是同一批 30 题的检索代理指标；CPU 重排成本见延迟列。",
                  f"Milvus dense 与 Chroma dense 的代理分数相同；Milvus hybrid 比 Milvus dense **{hybrid_delta:+.1f} 个百分点**，没有展示收益。Milvus hybrid + rerank 的 P50 是 Milvus dense 的约 **{rerank_slowdown:.0f} 倍**，因此当前不宜默认开启 CPU 重排。",
                  "", "## PDF 解析与已缓存问答", "",
                  "| 解析路径 | 完成/22 | 答案含金标准片段 | 引用金标准页 | 总 token |", "| --- | ---: | ---: | ---: | ---: |"])
    for name in ("pymupdf", "docling"):
        row = cached["summary"][name]
        lines.append(f"| `{name}` | {row['completed']}/22 | {row['answer_contains_gold_fragments']}/22 | {row['cited_gold_page']}/22 | {row['total_tokens']:,} |")
    lines.extend(["", "片段命中和页码命中只做自动检查。答案正确率、引用支持率需要检查原始 PDF；token 总量来自 M3 已缓存的 44 次调用，不是本轮费用。",
                  "", "## 跨论文证据检索", "",
                  "| 解析 | 分组 | 检索 | 页面召回 | 答案片段召回 | 两篇都命中 | 平均片段数 | P50 延迟 |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for parser in ("pymupdf", "docling"):
        for split in ("dev", "test"):
            for name in ("plain_dense", "bounded_evidence_graph"):
                row = multi["summary"][parser][split][name]
                lines.append(f"| {parser} | {split} (2 题) | {name} | {_pct(row['page_recall'])} | {_pct(row['answer_fragment_recall'])} | {row['all_gold_fragments_found']}/2 | {row['mean_chunk_count']:.1f} | {_seconds(row['latency_p50_seconds'])} |")
    lines.extend(["", "页面/答案片段召回的分母是每题两篇论文的金标准引用；两篇都命中的分母是题数。两题延迟样本过小，不能用来宣称性能优势。",
                  "", "## 失败案例与原因", ""])
    failures = 0
    for row in multi["records"]:
        if row["split"] != "test":
            continue
        variant = row["variants"]["bounded_evidence_graph"]
        if variant["status"] != "completed":
            lines.append(f"- `{row['parser']} / {row['case_id']}`：运行失败，{variant['error']}。")
            failures += 1
            continue
        missed = [paper["document_id"] for paper in variant["metrics"]["per_paper"]
                  if not paper["answer_fragment_present"]]
        if missed:
            lines.append(f"- `{row['parser']} / {row['case_id']}`：金标准页的答案片段未进入检索结果，缺失论文：{', '.join(missed)}。需要检查解析文本和排序；当前图按论文片段数补检，不能判断片段是否真的含答案。")
            failures += 1
    if not failures:
        lines.append("- 本轮跨论文 test 题没有答案片段漏检；仍需增加题量与人工审查。")
    lines.extend(["", "## Ragas 与人工复核", ""])
    if ragas:
        lines.append("Ragas `StringPresence` 对 M3 缓存答案进行离线片段检查：")
        lines.extend(["", "| 分组 | 答案含全部片段 | 平均片段存在率 |", "| --- | ---: | ---: |"])
        for name, row in ragas["summary"].items():
            lines.append(f"| {name} | {row['all_fragments_present']}/{row['cases']} | {_pct(row['mean_fragment_presence'])} |")
        lines.append("")
        lines.append("该指标只检查字串出现；不会判断句子真伪或证据是否支持。")
    else:
        lines.append("Ragas 可选依赖尚未跑通，不能将它列为已验收指标。")
    reviewed = sum(row["answer_reviewed"] for row in review["groups"].values())
    total = sum(row["rows"] for row in review["groups"].values())
    wall_seconds = (datetime.fromisoformat(qsummary["finished_at_utc"]) -
                    datetime.fromisoformat(qmanifest["started_at_utc"])).total_seconds()
    lines.extend([f"人工复核表：**{reviewed}/{total}** 条答案已填写正确性判断；"
                  + ("复核完成。" if review["complete"] else "尚未完成，不能报告人工正确率或引用支持率。"),
                  "", "## 成本、限制与下一步", "",
                  f"- QASPER 检索本轮完成 {len(qrecords)} 题 × {len(qmanifest['variants'])} 路；运行记录包含逐题耗时和失败状态。M6 额外生成调用 0 次、在线判分调用 0 次。",
                  f"- 这次 CPU 本机 QASPER 运行约 {wall_seconds / 60:.1f} 分钟，其中论文清洗、切块和两种索引写入约 {qsummary['ingest_seconds'] / 60:.1f} 分钟；查询延迟单独见表格，不包括入库。",
                  "- 现有首轮 PDF 标注未独立复核；四道跨论文题不足以代表复杂研究任务；结构化 QASPER 不能验证 PDF 页码。",
                  "- LitSearch 全量 64,183 篇语料尚未跑官方固定集；早期 32 篇烟测没有金标准覆盖，不能报告发现质量。",
                  "- 先由第二位审查者核对 44 条回答与 10 篇 PDF 的标注，再扩大多论文 test 集，补无答案/拒答题，并对比真实回答质量与引用支持率。",
                  "", "## 复现入口", "",
                  "- `python -m pytest -q test_m6_evaluation.py`：无密钥评测契约。",
                  "- `python -m evaluation.run_m3_ablation --split test --limit 30 --seed 42 ...`：六路检索；参数见 M6 运行说明。",
                  "- `python -m evaluation.run_m6_multi_evidence ...`：复用固定 PDF 索引的跨论文对照。",
                  "- `python -m evaluation.m6_review`：生成本地审查表；填写后加 `--summary-only` 汇总。",
                  "- `python -m evaluation.run_m6_ragas`：安装可选 Ragas 后，对缓存答案做离线字串检查。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasper-dir", type=Path, default=Path("evaluation/runs/m6_qasper_test30_20261018"))
    parser.add_argument("--multi-report", type=Path, default=Path("evaluation/runs/m6_multi_evidence_v2_20261018/report.json"))
    parser.add_argument("--answers-dir", type=Path, default=Path("evaluation/runs/m3_parser_answers_10pdf_20261017"))
    parser.add_argument("--ragas-report", type=Path, default=Path("evaluation/runs/m6_ragas_cached_answers_20261018.json"))
    parser.add_argument("--review-sheet", type=Path, default=Path("evaluation/runs/m6_answer_review_20261018.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/M6_REPORT.md"))
    args = parser.parse_args()
    report = build_report(args.qasper_dir, args.multi_report, args.answers_dir,
                          args.ragas_report, args.review_sheet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
