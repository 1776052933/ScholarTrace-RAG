"""Presentation helpers and export formats for AutoPaper UI results."""

from __future__ import annotations

import csv
from io import StringIO
import json
from typing import Any, Iterable


SUMMARY_LABELS = {
    "background": "研究背景",
    "method": "研究方法",
    "experiment": "实验设计",
    "result": "主要结果",
    "limitation": "局限性",
    "conclusion": "结论",
}

KNOWLEDGE_LABELS = {
    "research_problem": "研究问题",
    "method": "核心方法",
    "model": "模型",
    "dataset": "数据集",
    "experiment": "实验设置",
    "metrics": "评价指标",
    "results": "实验结果",
    "advantages": "优势",
    "limitations": "局限性",
    "future_work": "未来工作",
}


def selected_papers(result: dict[str, Any]) -> list[dict[str, Any]]:
    stages = result.get("stages", {})
    papers = ranked_papers(result)
    literature = stages.get("literature_database", {})
    explicit = literature.get("selected_papers")
    if isinstance(explicit, list):
        ranked_by_id = {
            str(paper.get("paper_id") or ""): paper
            for paper in papers
        }
        return [
            {
                **ranked_by_id.get(str(paper.get("paper_id") or ""), {}),
                **paper,
            }
            for paper in explicit
            if isinstance(paper, dict)
        ]
    selected_ids = literature.get("selected_paper_ids")
    if isinstance(selected_ids, list):
        wanted = {str(paper_id) for paper_id in selected_ids}
        return [
            {**paper, "original_rank": rank}
            for rank, paper in enumerate(papers, start=1)
            if str(paper.get("paper_id") or "") in wanted
        ]
    selected_count = literature.get(
        "selected_paper_count",
        len(papers),
    )
    return [
        {**paper, "original_rank": rank}
        for rank, paper in enumerate(papers[: int(selected_count)], start=1)
        if isinstance(paper, dict)
    ]


def ranked_papers(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every deduplicated paper in final score order."""
    ranking = (
        result.get("stages", {})
        .get("ranking_agent", {})
        .get("result", {})
    )
    papers = ranking.get("papers", []) or []
    scores = ranking.get("scores", {}) or {}
    evaluated_ids = {
        str(point.get("paper_id") or "")
        for point in ranking.get("graph_points", []) or []
        if isinstance(point, dict) and point.get("llm_evaluated")
    }
    enriched = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id") or "")
        score = scores.get(paper_id, {})
        enriched.append({
            **paper,
            **(score if isinstance(score, dict) else {}),
            "llm_evaluated": paper_id in evaluated_ids,
        })
    return enriched


def search_papers(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every paper retained by the search stage before ranking deduplication."""
    papers = (
        result.get("stages", {})
        .get("search_agent", {})
        .get("result", {})
        .get("papers", [])
        or []
    )
    return [paper for paper in papers if isinstance(paper, dict)]


def rag_outputs(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = result.get("stages", {}).get("rag_reader", {}).get("outputs", {})
    return outputs if isinstance(outputs, dict) else {}


def stage_times(result: dict[str, Any]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for name, value in result.get("stages", {}).items():
        if not isinstance(value, dict) or value.get("elapsed_seconds") is None:
            continue
        rows.append((name, float(value["elapsed_seconds"])))
    return sorted(rows, key=lambda item: item[1], reverse=True)


def result_json_bytes(result: dict[str, Any]) -> bytes:
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")


def papers_csv_bytes(papers: Iterable[dict[str, Any]]) -> bytes:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "精排排名",
        "综合得分",
        "相关等级",
        "DeepSeek二阶段评价",
        "标题",
        "作者",
        "摘要",
        "年份",
        "期刊或会议",
        "引用数",
        "主要来源",
        "全部来源",
        "DOI",
        "arXiv ID",
        "论文链接",
        "PDF 链接",
        "命中查询",
    ])
    for index, paper in enumerate(papers, start=1):
        writer.writerow([
            index,
            paper.get("final_score", ""),
            _relevance_label(paper.get("relevance_level")),
            "是" if paper.get("llm_evaluated") else "否",
            paper.get("title", ""),
            "; ".join(_author_name(author) for author in paper.get("authors", []) or []),
            paper.get("abstract", ""),
            paper.get("year", ""),
            paper.get("venue", ""),
            paper.get("citation_count", ""),
            paper.get("source", ""),
            "; ".join(str(source) for source in paper.get("sources", []) or []),
            paper.get("doi", ""),
            paper.get("arxiv_id", ""),
            paper.get("url", ""),
            paper.get("pdf_url", ""),
            "; ".join(str(query) for query in paper.get("matched_queries", []) or []),
        ])
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def markdown_report(result: dict[str, Any]) -> str:
    papers = selected_papers(result)
    outputs = rag_outputs(result)
    lines = [
        "# AutoPaper 科研分析报告",
        "",
        f"- 研究主题：{result.get('query', '')}",
        f"- 运行状态：{result.get('status', '')}",
        f"- 总耗时：{_duration(result.get('elapsed_seconds', 0))}",
        f"- 入选论文：{len(papers)} 篇",
        "",
        "## Top 论文",
        "",
        "| 原始排名 | 论文 | 年份 | 引用数 |",
        "| --- | --- | ---: | ---: |",
    ]
    for index, paper in enumerate(papers, start=1):
        title = _escape_table(paper.get("title") or "未命名论文")
        url = paper.get("url") or ""
        linked_title = f"[{title}]({url})" if url else title
        lines.append(
            f"| {paper.get('original_rank', index)} | {linked_title} | {paper.get('year') or '-'} | "
            f"{paper.get('citation_count') or 0} |"
        )

    for index, paper in enumerate(papers, start=1):
        paper_id = str(paper.get("paper_id") or "")
        output = outputs.get(paper_id, {})
        lines.extend([
            "",
            f"## {index}. {paper.get('title') or '未命名论文'}",
            "",
            f"**原始精排：** 第 {paper.get('original_rank', index)} 名",
            "",
            f"**作者：** {', '.join(paper.get('authors', []) or []) or '未知'}",
            "",
            f"**来源：** {paper.get('venue') or '未知'}",
            "",
            "### 原始摘要",
            "",
            str(paper.get("abstract") or "暂无摘要"),
        ])
        summary = output.get("summary")
        if isinstance(summary, dict):
            lines.extend(["", "### 智能摘要"])
            for key, value in summary.items():
                if value:
                    lines.extend(["", f"**{SUMMARY_LABELS.get(key, key)}：** {value}"])
        knowledge = output.get("knowledge")
        if isinstance(knowledge, dict):
            lines.extend(["", "### 知识抽取"])
            for key, value in knowledge.items():
                if key != "title" and value:
                    lines.extend(["", f"**{KNOWLEDGE_LABELS.get(key, key)}：** {value}"])
        answer = output.get("answer")
        if isinstance(answer, dict):
            _append_answer(lines, answer, heading="论文问答")
        interactive_answers = output.get("interactive_answers") or []
        for answer_index, interactive_answer in enumerate(interactive_answers, start=1):
            if isinstance(interactive_answer, dict):
                _append_answer(
                    lines,
                    interactive_answer,
                    heading=f"交互问答 {answer_index}",
                )

    lines.extend(["", "## 阶段耗时", ""])
    for stage, seconds in stage_times(result):
        lines.append(f"- {stage}: {_duration(seconds)}")
    return "\n".join(lines).strip() + "\n"


def format_duration(seconds: Any) -> str:
    return _duration(seconds)


def _duration(seconds: Any) -> str:
    value = float(seconds or 0)
    if value < 60:
        return f"{value:.1f} 秒"
    minutes, remainder = divmod(value, 60)
    return f"{int(minutes)} 分 {remainder:.0f} 秒"


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _author_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value)


def _relevance_label(value: Any) -> str:
    return {
        "highly_relevant": "高度相关",
        "relevant": "相关",
        "partially_relevant": "部分相关",
    }.get(str(value or ""), str(value or ""))


def _append_answer(lines: list[str], answer: dict[str, Any], *, heading: str) -> None:
    lines.extend([
        "",
        f"### {heading}",
        "",
        f"**问题：** {answer.get('question', '')}",
        "",
        f"**回答：** {answer.get('answer', '')}",
        "",
        f"**置信度：** {answer.get('confidence', '-')}",
    ])
