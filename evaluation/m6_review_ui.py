"""Local, no-key PDF answer review page for the M6 worksheet.

Run from the repository root with Streamlit. This page never calls a model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pymupdf
import streamlit as st

from evaluation.m6_review import ERROR_TYPES, FIELDS, summarize_sheet
from evaluation.m6_review_priorities import prioritize


SHEET = Path("evaluation/runs/m6_answer_review_20261018.csv")
PDF_MANIFEST = Path("evaluation/gold/m3_pdf_layout/manifest.json")
ANSWERS_DIR = Path("evaluation/runs/m3_parser_answers_10pdf_20261017")
RISK_LABELS = {
    "answer_missing_annotated_fragment": "回答未包含标注片段",
    "gold_page_not_cited": "未引用标注页",
    "annotated_fragment_not_contiguous_in_cited_page": "引用摘录未连续包含标注片段",
    "possible_abstention": "可能拒答",
    "no_citations": "无引用",
}


@st.cache_data(show_spinner=False)
def _page_png(path: str, number: int, mtime: float) -> bytes:
    del mtime
    with pymupdf.open(path) as document:
        if number < 1 or number > len(document):
            raise ValueError(f"Page {number} is outside this PDF")
        return document[number - 1].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False).tobytes("png")


def _rows() -> list[dict[str, str]]:
    with SHEET.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Review sheet columns have changed")
        return list(reader)


def _save(rows: list[dict[str, str]]) -> None:
    temporary = SHEET.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SHEET)
    summarize_sheet(SHEET)


def main() -> None:
    st.set_page_config(page_title="M6 答案复核", layout="wide")
    st.title("M6 论文问答复核")
    st.caption("逐题核对原始 PDF、回答和引用；本页不发送论文或答案到外部服务。")
    if not SHEET.is_file():
        st.error("未找到复核表。先运行 python -m evaluation.m6_review 生成本地 CSV。")
        st.stop()
    rows = _rows()
    summary = summarize_sheet(SHEET)
    reviewed = sum(bool(row["answer_correct"] and row["citations_supported"]) for row in rows)
    st.progress(reviewed / len(rows), text=f"回答和引用均已复核 {reviewed}/{len(rows)}")
    priorities = prioritize(ANSWERS_DIR)
    risks = {(item["split"], item["document_id"], item["question_id"], item["variant"]): item["reasons"]
             for item in priorities}
    pending_only = st.checkbox("只显示未完成的题目", value=True)
    risk_first = st.checkbox("优先显示需重点核对的题目", value=True)
    indexes = [i for i, row in enumerate(rows)
               if not pending_only or not row["answer_correct"] or not row["citations_supported"]]
    if risk_first:
        indexes.sort(key=lambda i: (not bool(risks.get((rows[i]["split"], rows[i]["document_id"],
                                                       rows[i]["question_id"], rows[i]["variant"]))), i))
    if not indexes:
        st.success("所有行都有判断。请导出并检查汇总结果。")
        st.stop()
    selected = st.selectbox(
        "选择题目", indexes,
        format_func=lambda i: f"{i + 1:02d} · {rows[i]['split']} · {rows[i]['document_id']} · {rows[i]['variant']} · {rows[i]['question'][:65]}",
    )
    row = rows[selected]
    flags = risks.get((row["split"], row["document_id"], row["question_id"], row["variant"]), [])
    if flags:
        st.warning("自动排查提示：" + "、".join(RISK_LABELS.get(flag, flag) for flag in flags)
                   + "。这些是核对线索，不是错误判定。")
    pdfs = {item["id"]: Path(item["pdf"])
            for item in json.loads(PDF_MANIFEST.read_text(encoding="utf-8"))["documents"]}
    pdf = pdfs[row["document_id"]]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("题目与回答")
        st.write(row["question"])
        st.text_area("回答原文", row["answer"], height=180, disabled=True)
        st.caption(f"待核对的标注候选：{row['answer_fragments']}；页码：{row['gold_page']}。请以原始 PDF 为准。")
        st.subheader("引用")
        pages = [int(value) for value in row["cited_pages"].split(" | ") if value.strip().isdigit()]
        st.write("引用页：" + (", ".join(map(str, pages)) if pages else "无"))
        st.text_area("引用摘录", row["cited_excerpts"], height=260, disabled=True)
        with st.form(key=f"review_{selected}"):
            choices = ["unreviewed", "yes", "no"]
            answer = st.radio("回答事实是否正确？", choices,
                              index=choices.index(row["answer_correct"] or "unreviewed"),
                              format_func={"unreviewed": "待核", "yes": "正确", "no": "错误"}.get,
                              horizontal=True)
            citation = st.radio("所引原文是否支持回答？", choices,
                                index=choices.index(row["citations_supported"] or "unreviewed"),
                                format_func={"unreviewed": "待核", "yes": "支持", "no": "不支持"}.get,
                                horizontal=True)
            errors = sorted(ERROR_TYPES)
            error_type = st.selectbox("错误类型", errors,
                                      index=errors.index(row["error_type"]),
                                      format_func=lambda value: value or "无 / 待判断")
            reviewer = st.text_input("复核人", value=row["reviewer"] or st.session_state.get("reviewer", ""))
            notes = st.text_area("备注", value=row["notes"])
            submitted = st.form_submit_button("保存本题", type="primary")
        if submitted:
            if (answer != "unreviewed" or citation != "unreviewed") and not reviewer.strip():
                st.error("填写判断时需要署名复核人。")
            else:
                row.update(answer_correct="" if answer == "unreviewed" else answer,
                           citations_supported="" if citation == "unreviewed" else citation,
                           error_type=error_type, reviewer=reviewer.strip(), notes=notes)
                st.session_state["reviewer"] = reviewer.strip()
                _save(rows)
                st.rerun()
    with right:
        st.subheader("原始 PDF")
        if not pdf.is_file():
            st.warning(f"本机缺少 PDF：{pdf}")
        else:
            display_pages = list(dict.fromkeys([int(row["gold_page"]), *pages]))
            for page in display_pages:
                with st.expander(f"PDF 第 {page} 页", expanded=page == int(row["gold_page"])):
                    st.image(_page_png(str(pdf), page, pdf.stat().st_mtime), width="stretch")


if __name__ == "__main__":
    main()
