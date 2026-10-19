"""Shared single- and multi-paper conversation view; all history lives in the API."""
from __future__ import annotations

import hashlib
import json

import streamlit as st

from api_client import (
    WorkflowRunError, create_conversation, export_answer, get_conversation,
    recent_conversations, stream_conversation,
)


def render_answer(answer, *, answer_id=None, key="answer"):
    st.markdown(str(answer.get("answer") or ""))
    if answer.get("confidence") is not None:
        st.caption(f"回答置信度：{answer['confidence']}")
    citations = [item for item in answer.get("citations") or [] if isinstance(item, dict)]
    if citations:
        with st.expander(f"证据片段（{len(citations)}）"):
            for index, citation in enumerate(citations, 1):
                st.markdown(f"**{index}. {citation.get('title') or citation.get('paper_id') or '论文证据'}**")
                location = [str(citation.get("paper_id") or "")]
                if citation.get("page") is not None:
                    location.append(f"第 {citation['page']} 页")
                location.extend(str(citation.get(field) or "") for field in ("section", "chunk_id"))
                st.caption(" · ".join(value for value in location if value))
                st.write(str(citation.get("excerpt") or ""))
    elif answer.get("sources"):
        with st.expander(f"证据片段（{len(answer['sources'])}）"):
            for source in answer["sources"]:
                st.code(str(source), language="text")
    if answer_id and st.button("准备导出本次问答", key=f"export_{key}_{answer_id}"):
        try:
            report = export_answer(answer_id)
            st.download_button("导出本次问答", report.encode("utf-8"),
                               file_name="scholartrace_answer.md", mime="text/markdown",
                               key=f"download_{key}_{answer_id}")
        except WorkflowRunError as exc:
            st.warning(str(exc))


def render_conversation(paper_ids, *, key, question_label, submit_label):
    scope = hashlib.sha256(json.dumps(sorted(paper_ids)).encode()).hexdigest()[:16]
    prefix = f"conversation_{key}_{scope}"
    widget_key = prefix + "_selected"
    connected = bool(st.session_state.get("api_connected"))
    sessions = []
    if connected:
        try:
            sessions = recent_conversations(paper_ids)
        except WorkflowRunError as exc:
            st.warning(f"无法读取会话记录：{exc}")
            return
    if st.button("新建会话", key=prefix + "_new", disabled=not connected):
        st.session_state[widget_key] = ""
        st.session_state.pop(prefix + "_older", None)
    pending = st.session_state.pop(prefix + "_pending", None)
    if pending:
        st.session_state[widget_key] = pending
    by_id = {item["conversation_id"]: item for item in sessions}
    options = [""] + list(by_id)
    if st.session_state.get(widget_key) not in options:
        st.session_state.pop(widget_key, None)
    conversation_id = st.selectbox(
        "会话记录", options, key=widget_key,
        index=None if widget_key in st.session_state else (1 if sessions else 0),
        format_func=lambda value: (
            f"{by_id[value]['title'][:55]} · {by_id[value]['updated_at'][:16]} · {value[:6]}" if value else "新会话"
        ), disabled=not connected,
    )
    st.caption("会话保留当前所选论文，可连续追问。更换论文范围会切换会话；刷新后可从会话记录继续。")
    snapshot = None
    if conversation_id:
        try:
            snapshot = get_conversation(conversation_id)
        except WorkflowRunError as exc:
            st.error(str(exc))
            return
    turns = (snapshot or {}).get("turns") or []
    older_page = st.session_state.get(prefix + "_older") or {}
    # Older pages belong to one conversation; switching the selector cannot leak them into another.
    if older_page.get("conversation_id") != conversation_id:
        older_page = {}
        st.session_state.pop(prefix + "_older", None)
    older = older_page.get("turns") or []
    if snapshot and older_page.get("has_older", snapshot.get("has_older")):
        if st.button("加载更早的对话", key=prefix + "_load_older"):
            try:
                before = min(t["turn_number"] for t in older + turns)
                page = get_conversation(conversation_id, before=before)
            except WorkflowRunError as exc:
                st.warning(str(exc))
            else:
                older = page["turns"] + older
                st.session_state[prefix + "_older"] = {
                    "conversation_id": conversation_id, "turns": older, "has_older": page["has_older"],
                }
    if snapshot and snapshot.get("active_turn_id"):
        st.button("刷新会话", key=prefix + "_refresh")
    for turn in older + turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn["status"] == "completed":
                render_answer(turn["answer"], answer_id=turn["turn_id"], key=prefix)
            elif turn["status"] == "generating":
                st.info("这条回答仍在生成。可稍后刷新会话查看结果。")
            else:
                st.warning(turn.get("error") or "本轮未完成，可重新提问。")
    error = st.session_state.pop(prefix + "_error", None)
    if error:
        st.error(error)
    with st.form(prefix + "_form", clear_on_submit=True):
        question = st.text_area(question_label, placeholder="例如：它的实验使用了哪些数据集？与另一篇有什么区别？",
                                height=88, max_chars=2000)
        submitted = st.form_submit_button(submit_label, type="primary", disabled=not connected)
    if not submitted:
        return
    if not question.strip():
        st.warning("请输入问题。")
        return
    try:
        if not conversation_id:
            conversation_id = create_conversation(paper_ids)["conversation_id"]
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            progress = st.empty()
            text = st.empty()
            accumulated = ""
            labels = {"rewriting": "正在结合历史理解追问…", "retrieving": "正在检索论文原文…",
                      "generating": "正在生成回答…"}
            for event in stream_conversation(conversation_id, question.strip()):
                if event["event"] == "status":
                    progress.caption(labels.get(event["data"].get("stage"), "正在处理…"))
                elif event["event"] == "delta":
                    accumulated += event["data"]["text"]
                    text.markdown(accumulated + " ▌")
                elif event["event"] == "done":
                    text.markdown(event["data"]["answer"]["answer"])
                    progress.empty()
    except WorkflowRunError as exc:
        st.session_state[prefix + "_error"] = str(exc)
    if conversation_id:
        st.session_state[prefix + "_pending"] = conversation_id
    st.rerun()
