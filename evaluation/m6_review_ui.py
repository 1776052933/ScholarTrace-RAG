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
from evaluation.m6_review_translations import load_translations, translate, reference_translation


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
ERROR_LABELS = {
    "": "无 / 待判断",
    "retrieval_miss": "检索漏掉答案证据",
    "parser_miss": "PDF 解析丢失答案内容",
    "unsupported_claim": "回答有缺乏证据的断言",
    "wrong_value": "回答数值或事实错误",
    "wrong_citation": "引用页或引用内容错误",
    "appropriate_abstention": "合理拒答（原论文也无法回答）",
    "other": "其他",
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


@st.cache_data(show_spinner=False)
def _answer_records() -> dict[tuple[str, str, str, str], dict]:
    records = {}
    for path in sorted(ANSWERS_DIR.glob("[0-9][0-9][0-9]_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item["split"], item["document_id"], item["question_id"], item["variant"])
        records[key] = item
    return records


def _save(rows: list[dict[str, str]]) -> None:
    temporary = SHEET.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summarize_sheet(temporary)
    temporary.replace(SHEET)


def _bilingual(label: str, original: str, catalog: dict, key: str) -> None:
    st.markdown(f"**{label}**")
    chinese = translate(original, catalog)
    if chinese is None:
        st.warning("这段内容尚无匹配的中文译文，请查看原文；不要据此猜测评分。")
        st.text(original)
    else:
        st.markdown(chinese)
    with st.expander(f"{label} · 查看原文"):
        st.text_area("原始内容", original, height=180, disabled=True, key=key)


def main() -> None:
    st.set_page_config(page_title="M6 中文辅助复核", layout="wide")
    st.title("M6 论文问答复核 · 中文辅助")
    st.caption("中文译文已保存在本机，打开页面无需调用模型，也不会向外部服务发送内容。")
    st.info("先读中文问题和回答，再对照引用及右侧 PDF 判断。译文由 AI 辅助生成，不是标准答案；"
            "可能有翻译误差，尤其是原文串栏或表格错位时。无法确认就保留“待复核”，不要猜测。")
    if not SHEET.is_file():
        st.error("未找到复核表。先运行 python -m evaluation.m6_review 生成本地 CSV。")
        st.stop()
    rows = _rows()
    catalog = load_translations()
    unique_questions = len({(row["document_id"], row["question_id"]) for row in rows})
    st.caption(f"共 {unique_questions} 道问题、{len(rows)} 条回答（两种 PDF 解析方案）。"
               "本次属于机器翻译辅助的人工复核；问题、模型回答及已有评分保持原样。")
    summarize_sheet(SHEET)
    reviewed = sum(bool(row["answer_correct"] and row["citations_supported"]) for row in rows)
    st.progress(reviewed / len(rows), text=f"累计已完成 {reviewed}/{len(rows)} 条回答复核（包含之前保存的记录）")
    st.caption("这是原来的同一批记录，中文版只增加翻译。编号是固定记录编号，不是本次作答顺序；"
               "旧界面默认优先显示需重点核对的记录，现在默认按编号显示，因此首条可能不同。"
               "同一道问题的两种解析方案各算一条复核记录。")
    with st.expander(f"查看已保存的判断，找回需要修改的记录（累计完成 {reviewed} 条）"):
        st.caption("要修改下面的记录，请取消勾选“只显示未完成的题目”，再按固定编号选择。"
                   "修改已完成记录不会增加累计完成数。")
        saved = [
            {"固定编号": f"{i + 1:02d}", "问题": translate(row["question"], catalog) or row["question"],
             "解析方案": row["variant"],
             "回答判断": {"": "待复核", "yes": "正确", "no": "错误"}[row["answer_correct"]],
             "引用判断": {"": "待复核", "yes": "支持", "no": "不支持"}[row["citations_supported"]]}
            for i, row in enumerate(rows) if row["answer_correct"] or row["citations_supported"]
        ]
        if saved:
            st.dataframe(saved, hide_index=True, width="stretch")
        else:
            st.write("尚无已保存的判断。")
    priorities = prioritize(ANSWERS_DIR)
    risks = {(item["split"], item["document_id"], item["question_id"], item["variant"]): item["reasons"]
             for item in priorities}
    pending_only = st.checkbox("只显示未完成的题目", value=True)
    show_hints = st.checkbox("显示自动排查提示（可能影响独立判断，默认关闭）", value=False)
    risk_first = st.checkbox("优先显示需重点核对的题目", value=False) if show_hints else False
    indexes = [i for i, row in enumerate(rows)
               if not pending_only or not row["answer_correct"] or not row["citations_supported"]]
    if risk_first:
        indexes.sort(key=lambda i: (not bool(risks.get((rows[i]["split"], rows[i]["document_id"],
                                                       rows[i]["question_id"], rows[i]["variant"]))), i))
    if not indexes:
        st.success("所有行都有判断。请导出并检查汇总结果。")
        st.stop()
    selected = st.selectbox(
        "选择题目（固定记录编号）", indexes,
        format_func=lambda i: f"记录 {i + 1:02d} · "
                              f"{'已完成' if rows[i]['answer_correct'] and rows[i]['citations_supported'] else '待完成'} · "
                              f"{rows[i]['variant']} · "
                              f"{translate(rows[i]['question'], catalog) or rows[i]['question']}",
    )
    row = rows[selected]
    key = (row["split"], row["document_id"], row["question_id"], row["variant"])
    flags = risks.get(key, [])
    source = _answer_records()[key]
    citations = source.get("citations") or []
    if show_hints and flags:
        st.warning("自动排查提示：" + "、".join(RISK_LABELS.get(flag, flag) for flag in flags)
                   + "。这些是核对线索，不是错误判定。")
    pdfs = {item["id"]: Path(item["pdf"])
            for item in json.loads(PDF_MANIFEST.read_text(encoding="utf-8"))["documents"]}
    pdf = pdfs[row["document_id"]]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("题目与回答")
        st.caption(f"论文：{row['document_id']} · 解析方案：{row['variant']} · "
                   f"{'开发集' if row['split'] == 'dev' else '测试集'}")
        _bilingual("问题（中文）", row["question"], catalog, f"question_original_{selected}")
        _bilingual("模型回答（中文，待你判断对错）", row["answer"], catalog, f"answer_original_{selected}")
        st.caption(f"标注候选（定位线索）：{reference_translation(row['answer_fragments'], catalog)}；"
                   f"PDF 第 {row['gold_page']} 页。请以原始 PDF 为准。")
        with st.expander("标注候选原文"):
            st.text(row["answer_fragments"])
        st.subheader("引用")
        pages = [int(cite["page"]) for cite in citations]
        st.write("引用页：" + (", ".join(map(str, pages)) if pages else "无"))
        if not citations:
            st.info("该回答没有引用片段。")
        for index, cite in enumerate(citations, start=1):
            _bilingual(f"引用 {index}（中文）· PDF 第 {cite['page']} 页",
                       str(cite.get("excerpt") or ""), catalog, f"citation_{selected}_{index}")
        with st.expander("判定口径"):
            st.markdown("回答应完整、准确地回答问题，以原始 PDF 为准；标注候选仅作定位线索。"
                        "引用需要在所标页提供足以支持回答的原文，仅页码相同不算支持。"
                        "如果论文中能找到答案而系统拒答，回答应判为错误，并记录漏检或解析问题。"
                        "只有原论文也无法回答时，才选择合理拒答。")
        with st.form(key=f"review_{selected}"):
            choices = ["unreviewed", "yes", "no"]
            answer = st.radio("回答事实是否正确？", choices,
                              index=choices.index(row["answer_correct"] or "unreviewed"),
                              format_func={"unreviewed": "无法判断 / 待复核", "yes": "正确", "no": "错误"}.get,
                              horizontal=True)
            citation = st.radio("所引原文是否支持回答？", choices,
                                index=choices.index(row["citations_supported"] or "unreviewed"),
                                format_func={"unreviewed": "无法判断 / 待复核", "yes": "支持", "no": "不支持"}.get,
                                horizontal=True)
            errors = sorted(ERROR_TYPES)
            error_type = st.selectbox("错误类型", errors,
                                      index=errors.index(row["error_type"]),
                                      format_func=lambda value: ERROR_LABELS[value])
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
        st.caption("PDF 保留原版英文；左侧提供引用片段中文对照。表格请同时核对行名、列名、数字及页码。"
                   "仅凭译文无法确认时，请留待进一步核对。")
        if not pdf.is_file():
            st.warning(f"本机缺少 PDF：{pdf}")
        else:
            with pymupdf.open(pdf) as document:
                page_count = len(document)
            display_pages = list(dict.fromkeys([int(row["gold_page"]), *pages]))
            for page in display_pages:
                with st.expander(f"PDF 第 {page} 页", expanded=page == int(row["gold_page"])):
                    st.image(_page_png(str(pdf), page, pdf.stat().st_mtime), width="stretch")
            browse_page = st.number_input("需要时查看 PDF 其他页", min_value=1,
                                          max_value=page_count, value=int(row["gold_page"]),
                                          step=1, key=f"browse_page_{selected}")
            if browse_page not in display_pages:
                st.image(_page_png(str(pdf), browse_page, pdf.stat().st_mtime),
                         caption=f"PDF 第 {browse_page} 页", width="stretch")


if __name__ == "__main__":
    main()
