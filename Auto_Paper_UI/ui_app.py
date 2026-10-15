"""Streamlit user interface for the AutoPaper research workflow."""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import streamlit as st


UI_ROOT = Path(__file__).resolve().parent
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from api_client import (
    RunOptions,
    WorkflowRunError,
    api_status,
    ask_paper_question,
    cancel_workflow,
    poll_workflow,
    recent_workflows,
    start_workflow,
)
from reporting import (
    KNOWLEDGE_LABELS,
    SUMMARY_LABELS,
    format_duration,
    markdown_report,
    papers_csv_bytes,
    ranked_papers,
    rag_outputs,
    selected_papers,
    stage_times,
)


st.set_page_config(
    page_title="AutoPaper Studio",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)


STAGE_LABELS = {
    "query_understanding": "查询理解",
    "query_decomposition": "问题拆解",
    "search_agent": "论文检索",
    "citation_tracking": "引文追踪",
    "ranking_agent": "论文精排",
    "literature_database": "下载与入库",
    "rag_reader": "RAG 智能阅读",
}

STAGE_PROGRESS = {
    "query_understanding": 10,
    "query_decomposition": 22,
    "search_agent": 36,
    "citation_tracking": 50,
    "ranking_agent": 64,
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ap-ink: #16202a;
            --ap-muted: #65717d;
            --ap-line: #dfe4e8;
            --ap-teal: #087f78;
            --ap-coral: #d85b45;
            --ap-yellow: #f2c14e;
            --ap-surface: #f7f9fa;
        }
        .stApp { background: #ffffff; color: var(--ap-ink); }
        [data-testid="stSidebar"] { background: var(--ap-surface); border-right: 1px solid var(--ap-line); }
        [data-testid="stSidebar"] > div { padding-top: 0; }
        [data-testid="stSidebarUserContent"] {
            padding-top: 0 !important;
            margin-top: calc(-4rem - 4px);
        }
        [data-testid="stSidebarHeader"] {
            position: relative;
            z-index: 1000;
            pointer-events: none;
        }
        [data-testid="stSidebarHeader"] button,
        [data-testid="stSidebarCollapseButton"] {
            position: relative;
            z-index: 1001;
            pointer-events: auto !important;
        }
        .ap-brand { display: flex; align-items: center; gap: 12px; margin: 0 0 4px; }
        .ap-mark {
            width: 36px; height: 36px; display: grid; place-items: center;
            color: white; background: var(--ap-teal); border-radius: 7px;
            font-size: 18px; font-weight: 750;
        }
        .ap-title { font-size: 1.45rem; line-height: 1.2; font-weight: 760; letter-spacing: 0; }
        .ap-subtitle { color: var(--ap-muted); margin: 0 0 1.25rem 48px; font-size: .92rem; }
        .ap-kicker { color: var(--ap-coral); font-size: .78rem; font-weight: 700; text-transform: uppercase; }
        .ap-section-title { font-size: 1.05rem; font-weight: 720; margin: .2rem 0 .15rem; }
        .ap-paper-meta { color: var(--ap-muted); font-size: .88rem; margin-bottom: .55rem; }
        .ap-empty {
            min-height: 260px; border: 1px dashed #bcc6cc; border-radius: 8px;
            background: var(--ap-surface); display: grid; place-items: center;
            text-align: center; color: var(--ap-muted); padding: 2rem;
        }
        .ap-empty strong { display: block; color: var(--ap-ink); font-size: 1.05rem; margin-bottom: .35rem; }
        .ap-api-ok { color: var(--ap-teal); font-weight: 650; }
        .ap-api-missing { color: var(--ap-coral); font-weight: 650; }
        .ap-api-status {
            display: grid;
            grid-template-columns: 5.25rem minmax(0, 1fr);
            column-gap: .45rem;
            row-gap: .2rem;
            align-items: baseline;
            margin-bottom: .65rem;
        }
        .ap-api-label { color: var(--ap-ink); }
        [data-testid="stSidebar"] [data-testid="stTextInput"] [data-testid="stWidgetLabel"] > div {
            width: 100%;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] [data-testid="stTooltipIcon"] {
            margin-left: auto;
            margin-right: .7rem;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"] {
            border: 1px solid #87949d !important;
            border-radius: 7px;
            box-shadow: inset 0 0 0 1px #87949d !important;
        }
        [data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: var(--ap-teal) !important;
            box-shadow: inset 0 0 0 1px var(--ap-teal) !important;
        }
        div[data-testid="stMetric"] { background: var(--ap-surface); border: 1px solid var(--ap-line); padding: 12px 14px; border-radius: 7px; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--ap-line); border-radius: 8px; }
        .stButton > button, .stDownloadButton > button, .stLinkButton > a { border-radius: 6px; }
        .stButton > button[kind="primary"] { background: var(--ap-teal); border-color: var(--ap-teal); }
        .stTabs [data-baseweb="tab-list"] { gap: 18px; border-bottom: 1px solid var(--ap-line); }
        .stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
        code { white-space: pre-wrap !important; overflow-wrap: anywhere; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="ap-brand">
          <div class="ap-mark">A</div>
          <div class="ap-title">AutoPaper Studio</div>
        </div>
        <div class="ap-subtitle">科研论文检索、精排与证据问答工作台</div>
        """,
        unsafe_allow_html=True,
    )


def render_controls() -> tuple[RunOptions, bool]:
    st.sidebar.markdown("### 研究任务")
    query = st.sidebar.text_area(
        "研究主题",
        placeholder="例如：Swin UNETR medical image segmentation",
        height=100,
    )
    max_papers = st.sidebar.number_input("论文数量", min_value=1, max_value=20, value=3)
    max_search_rounds = st.sidebar.number_input(
        "最大搜索轮次",
        min_value=1,
        max_value=5,
        value=1,
    )

    st.sidebar.markdown("### 处理功能")
    download_papers = st.sidebar.toggle("下载论文 PDF", value=True)
    enable_rag_widget = st.sidebar.toggle(
        "启用 RAG 阅读",
        value=True,
        disabled=not download_papers,
    )
    enable_rag = bool(download_papers and enable_rag_widget)
    summarize = st.sidebar.toggle("生成论文摘要", value=True, disabled=not enable_rag)
    extract_knowledge = st.sidebar.toggle("抽取结构化知识", value=True, disabled=not enable_rag)

    with st.sidebar.expander("高级设置"):
        use_query_llm = st.toggle("使用 DeepSeek 理解查询", value=True)
        enable_citation_tracking = st.toggle("启用引文追踪", value=True)
        publication_year = st.number_input(
            "论文发表截止年份（可选）",
            min_value=1900,
            max_value=datetime.now().year,
            value=None,
            step=1,
            placeholder="留空表示截至当前最新",
            help="只保留该年份及以前发表的论文；留空时不限制发表年份。",
        )
        download_engine_labels = {
            "academic_oa": "Academic-OA",
            "auto": "Auto",
            "arxiv": "ArXiv",
            "openalex": "OpenAlex",
            "crossref": "Crossref",
            "scholar": "Scholar",
        }
        download_engine = st.selectbox(
            "下载来源",
            list(download_engine_labels),
            index=0,
            format_func=download_engine_labels.get,
        )
        download_timeout = st.number_input(
            "单次下载超时（秒）",
            min_value=10,
            max_value=300,
            value=30,
            step=10,
        )
        retry_failed = st.checkbox(
            "重试历史失败任务",
            value=True,
            help="建议保持开启，网络波动或下载策略升级后会重新尝试失败论文。",
        )

    with st.sidebar.expander("API 设置"):
        status = api_status()
        st.session_state["api_connected"] = status["connected"]
        st.write("研究 API：" + ("已连接" if status["connected"] else "未连接"))
        st.write("MySQL/数据库：" + ("可用" if status["database"] else "不可用"))
        for label, key in (("DeepSeek", "deepseek"), ("SerpAPI", "serpapi"), ("OpenAlex", "openalex")):
            st.write(f"{label}：" + ("服务端已配置" if status[key] else "服务端未配置"))
        if not status["connected"]:
            st.warning("请先启动研究 API，再提交任务。")

    workflow_active = bool(st.session_state.get("active_workflow_job_id"))
    run_clicked = st.sidebar.button(
        "取消检索" if workflow_active else "开始检索",
        type="secondary" if workflow_active else "primary",
        use_container_width=True,
        help=(
            "终止本次检索；已完成并核验的 PDF 与缓存会保留。"
            if workflow_active
            else None
        ),
    )
    options = RunOptions(
        query=query,
        max_papers=int(max_papers),
        max_search_rounds=int(max_search_rounds),
        download_papers=download_papers,
        enable_rag=enable_rag,
        summarize=bool(enable_rag and summarize),
        extract_knowledge=bool(enable_rag and extract_knowledge),
        question=None,
        use_query_llm=use_query_llm,
        enable_citation_tracking=enable_citation_tracking,
        publication_date_to=(
            f"{int(publication_year)}-12-31"
            if publication_year is not None
            else None
        ),
        download_engine=download_engine,
        download_timeout=int(download_timeout),
        retry_failed_downloads=retry_failed,
    )
    return options, run_clicked


def execute(options: RunOptions) -> None:
    if not options.query.strip():
        st.sidebar.error("请先输入研究主题。")
        return

    try:
        job_id = start_workflow(options)
    except WorkflowRunError as exc:
        st.session_state["workflow_notice"] = {
            "level": "error",
            "message": str(exc),
            "log_path": str(exc.log_path) if exc.log_path else None,
        }
        st.rerun()
        return

    st.session_state["active_workflow_job_id"] = job_id
    st.session_state.pop("result", None)
    st.session_state.pop("output_path", None)
    st.session_state.pop("log_path", None)
    st.session_state.pop("workflow_notice", None)
    st.rerun()


def cancel_active_workflow() -> None:
    job_id = st.session_state.get("active_workflow_job_id")
    if not job_id:
        return
    try:
        snapshot = cancel_workflow(str(job_id))
    except WorkflowRunError as exc:
        st.session_state["workflow_notice"] = {"level": "error", "message": f"取消检索失败：{exc}"}
    else:
        if snapshot.get("status") == "completed":
            _store_completed_workflow(snapshot)
            st.session_state.pop("active_workflow_job_id", None)
        elif snapshot.get("status") in {"cancelled", "failed", "interrupted"}:
            st.session_state.pop("active_workflow_job_id", None)
        else:
            st.session_state["workflow_notice"] = {
                "level": "info",
                "message": "已请求取消。当前步骤结束后将停止，请等待任务状态更新。",
            }
    st.rerun()


def _store_completed_workflow(snapshot: dict[str, Any]) -> None:
    st.session_state["result"] = snapshot.get("result")
    st.session_state["logs"] = list(snapshot.get("logs") or [])


@st.fragment(run_every=1.0)
def render_active_workflow() -> None:
    job_id = st.session_state.get("active_workflow_job_id")
    if not job_id:
        return
    try:
        snapshot = poll_workflow(str(job_id))
    except WorkflowRunError as exc:
        st.error(f"暂时无法查询任务：{exc}")
        return

    status = snapshot.get("status")
    if status in {"queued", "running", "cancelling"}:
        stage = str(snapshot.get("stage") or "")
        label = STAGE_LABELS.get(stage, "准备运行环境")
        if status == "cancelling":
            label = "等待当前步骤结束后取消"
        with st.status("正在运行研究工作流", state="running", expanded=True):
            st.write(f"正在执行：{label}")
            st.caption("任务状态由 API 持久化；刷新页面后可从最近任务中恢复。")
        st.progress(int(snapshot.get("progress") or 0), text=label)
        return

    st.session_state.pop("active_workflow_job_id", None)
    if status == "completed":
        _store_completed_workflow(snapshot)
        st.session_state["workflow_notice"] = {"level": "success", "message": "检索与处理已完成。"}
    elif status == "cancelled":
        st.session_state["workflow_notice"] = {"level": "warning", "message": "本次检索已取消，已完成的论文资源保留。", "logs": snapshot.get("logs")}
    else:
        st.session_state["workflow_notice"] = {"level": "error", "message": str(snapshot.get("error") or "检索运行失败。"), "logs": snapshot.get("logs")}
    st.rerun()


def _workflow_progress(logs: list[str]) -> tuple[int, str]:
    percentage = 2
    label = "准备运行环境"
    for line in logs:
        for stage, stage_percentage in STAGE_PROGRESS.items():
            if f"开始阶段：{stage}" in line:
                percentage = stage_percentage
                label = STAGE_LABELS[stage]
            elif f"完成阶段：{stage}" in line:
                percentage = min(stage_percentage + 10, 78)
                label = f"{STAGE_LABELS[stage]}完成"
        if "prepare_and_ingest" in line or "下载" in line or "PDF" in line:
            percentage = max(percentage, 82)
            label = "下载论文并构建知识库"
        if "开始阶段：rag_reader" in line:
            percentage = 90
            label = STAGE_LABELS["rag_reader"]
    return percentage, label


def render_workflow_notice() -> None:
    notice = st.session_state.get("workflow_notice")
    if not isinstance(notice, dict):
        return
    level = str(notice.get("level") or "info")
    message = str(notice.get("message") or "")
    getattr(st, level if level in {"success", "warning", "error", "info"} else "info")(message)
    logs = list(notice.get("logs") or [])
    if logs and level in {"warning", "error"}:
        st.download_button(
            "下载本次运行日志",
            data=("\n".join(logs) + "\n").encode("utf-8"),
            file_name="scholartrace_job.log",
            mime="text/plain",
            key="workflow_notice_log_download",
        )


def render_result(
    result: dict[str, Any],
    *,
    deepseek_api_key: str | None = None,
) -> None:
    papers = selected_papers(result)
    outputs = rag_outputs(result)
    search_result = result.get("stages", {}).get("search_agent", {}).get("result", {})
    searched_count = search_result.get("total") or len(search_result.get("papers", []) or [])
    deduplicated_count = len(ranked_papers(result))
    literature_result = result.get("stages", {}).get("literature_database", {})
    download_results = literature_result.get("download_results", []) or []
    downloaded_count = literature_result.get("download_success_count")
    if downloaded_count is None:
        downloaded_count = sum(
            1 for item in download_results
            if isinstance(item, dict) and item.get("success")
        )

    metric_columns = st.columns(5)
    metric_columns[0].metric("检索结果", f"{searched_count} 篇")
    metric_columns[1].metric("去重结果", f"{deduplicated_count} 篇")
    metric_columns[2].metric("Top 论文", f"{len(papers)} 篇")
    metric_columns[3].metric("下载成功", f"{downloaded_count} 篇")
    metric_columns[4].metric("总耗时", format_duration(result.get("elapsed_seconds", 0)))

    paper_tab, graph_tab, reader_tab, overview_tab, export_tab = st.tabs([
        "Top 论文",
        "论文关系图",
        "智能阅读",
        "运行概览",
        "导出报告",
    ])
    with paper_tab:
        render_papers(papers)
    with graph_tab:
        render_relationship_graph(result)
    with reader_tab:
        render_reader(
            papers,
            outputs,
            result,
            deepseek_api_key=deepseek_api_key,
        )
    with overview_tab:
        render_overview(result)
    with export_tab:
        render_exports(result, st.session_state.get("logs"))


def render_papers(papers: list[dict[str, Any]]) -> None:
    if not papers:
        st.info("没有获得可展示的论文。")
        return
    for index, paper in enumerate(papers, start=1):
        with st.container(border=True):
            st.markdown(f"#### {index}. {paper.get('title') or '未命名论文'}")
            st.caption(f"原始精排第 {paper.get('original_rank', index)} 名")
            authors = ", ".join(str(item) for item in paper.get("authors", []) or [])
            meta = " · ".join(
                item
                for item in (
                    authors or "作者未知",
                    str(paper.get("year") or "年份未知"),
                    f"引用 {paper.get('citation_count') or 0}",
                )
                if item
            )
            st.markdown(f"<div class='ap-paper-meta'>{meta}</div>", unsafe_allow_html=True)
            if paper.get("final_score") is not None:
                score = float(paper["final_score"])
                level = _relevance_label(paper.get("relevance_level"))
                st.caption(f"综合得分 {score:.3f} · {level}")
            if paper.get("venue"):
                st.caption(str(paper["venue"]))
            st.write(paper.get("abstract") or "暂无摘要。")
            if paper.get("url"):
                st.link_button("查看论文", str(paper["url"]))


def _build_sphere_mesh(
    spheres: list[dict[str, Any]],
    radius: float,
    *,
    latitude_steps: int = 5,
    longitude_steps: int = 8,
) -> dict[str, list[Any]]:
    mesh: dict[str, list[Any]] = {
        "x": [], "y": [], "z": [],
        "i": [], "j": [], "k": [],
        "intensity": [], "customdata": [],
    }

    def add_vertex(
        x: float,
        y: float,
        z: float,
        intensity: float,
        customdata: Any,
    ) -> int:
        index = len(mesh["x"])
        mesh["x"].append(x)
        mesh["y"].append(y)
        mesh["z"].append(z)
        mesh["intensity"].append(intensity)
        mesh["customdata"].append(customdata)
        return index

    for sphere in spheres:
        center_x = float(sphere.get("x") or 0.0)
        center_y = float(sphere.get("y") or 0.0)
        center_z = float(sphere.get("z") or 0.0)
        sphere_radius = float(sphere.get("radius") or radius)
        intensity = float(sphere.get("intensity") or 0.0)
        customdata = sphere.get("customdata")
        top = add_vertex(
            center_x,
            center_y,
            center_z + sphere_radius,
            intensity,
            customdata,
        )
        rings: list[list[int]] = []
        for latitude in range(1, latitude_steps):
            theta = math.pi * latitude / latitude_steps
            ring: list[int] = []
            for longitude in range(longitude_steps):
                phi = 2 * math.pi * longitude / longitude_steps
                ring.append(add_vertex(
                    center_x + sphere_radius * math.sin(theta) * math.cos(phi),
                    center_y + sphere_radius * math.sin(theta) * math.sin(phi),
                    center_z + sphere_radius * math.cos(theta),
                    intensity,
                    customdata,
                ))
            rings.append(ring)
        bottom = add_vertex(
            center_x,
            center_y,
            center_z - sphere_radius,
            intensity,
            customdata,
        )

        for longitude in range(longitude_steps):
            following = (longitude + 1) % longitude_steps
            mesh["i"].append(top)
            mesh["j"].append(rings[0][longitude])
            mesh["k"].append(rings[0][following])
        for ring_index in range(len(rings) - 1):
            upper = rings[ring_index]
            lower = rings[ring_index + 1]
            for longitude in range(longitude_steps):
                following = (longitude + 1) % longitude_steps
                mesh["i"].extend([upper[longitude], upper[following]])
                mesh["j"].extend([lower[longitude], lower[longitude]])
                mesh["k"].extend([upper[following], lower[following]])
        for longitude in range(longitude_steps):
            following = (longitude + 1) % longitude_steps
            mesh["i"].append(bottom)
            mesh["j"].append(rings[-1][following])
            mesh["k"].append(rings[-1][longitude])
    return mesh


def render_relationship_graph(result: dict[str, Any]) -> None:
    ranking = (
        result.get("stages", {})
        .get("ranking_agent", {})
        .get("result", {})
    )
    points = [
        point for point in ranking.get("graph_points", []) or []
        if isinstance(point, dict)
    ]
    edges = _relationship_edges(
        points,
        ranking.get("graph_edges", []) or [],
    )
    if not points:
        st.info("本次结果没有可展示的论文关系数据，请重新运行检索。")
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("论文关系图依赖 Plotly，请安装 UI 依赖后重新启动。")
        return

    view_mode = st.segmented_control(
        "关系图视图",
        options=["三维关系图", "二维层级图"],
        default="三维关系图",
        label_visibility="collapsed",
        key="relationship_graph_view",
    )
    if view_mode == "二维层级图":
        _render_2d_relationship_graph(result, ranking, points, edges, go)
        _render_graph_metrics(ranking, points)
        return

    coordinate_values: list[float] = []
    for point in points:
        for axis in ("x", "y", "z"):
            try:
                coordinate_values.append(abs(float(point.get(axis, 0.0))))
            except (TypeError, ValueError):
                pass
    axis_extent = max(1.0, max(coordinate_values, default=1.0) * 1.2)
    shaft_end = axis_extent * 0.94

    figure = go.Figure()
    point_by_id = {
        str(point.get("paper_id") or ""): point
        for point in points
        if point.get("paper_id")
    }
    edge_x: list[Any] = []
    edge_y: list[Any] = []
    edge_z: list[Any] = []
    for edge in edges:
        target = point_by_id.get(str(edge.get("target_id") or ""))
        if target is None:
            continue
        source = point_by_id.get(str(edge.get("source_id") or ""))
        start = (
            float(source.get("x") or 0.0),
            float(source.get("y") or 0.0),
            float(source.get("z") or 0.0),
        ) if source else (0.0, 0.0, 0.0)
        edge_x.extend([start[0], float(target.get("x") or 0.0), None])
        edge_y.extend([start[1], float(target.get("y") or 0.0), None])
        edge_z.extend([start[2], float(target.get("z") or 0.0), None])
    if edge_x:
        figure.add_trace(go.Scatter3d(
            name="论文发现关系",
            mode="lines",
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line={"color": "#9aa3aa", "width": 2},
            opacity=0.42,
            hoverinfo="skip",
            showlegend=False,
        ))
    for axis, coordinates in (
        ("X", ([-axis_extent, shaft_end], [0, 0], [0, 0])),
        ("Y", ([0, 0], [-axis_extent, shaft_end], [0, 0])),
        ("Z", ([0, 0], [0, 0], [-axis_extent, shaft_end])),
    ):
        figure.add_trace(go.Scatter3d(
            name=f"{axis} 轴",
            mode="lines",
            x=coordinates[0],
            y=coordinates[1],
            z=coordinates[2],
            line={"color": "#1f2529", "width": 4},
            hoverinfo="skip",
            showlegend=False,
        ))
    figure.add_trace(go.Cone(
        name="坐标轴正方向",
        x=[axis_extent, 0.0, 0.0],
        y=[0.0, axis_extent, 0.0],
        z=[0.0, 0.0, axis_extent],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        w=[0.0, 0.0, 1.0],
        anchor="tip",
        sizemode="absolute",
        sizeref=axis_extent * 0.03,
        colorscale=[[0.0, "#1f2529"], [1.0, "#1f2529"]],
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    ))
    first_ranked = [
        point for point in points
        if str(point.get("rank") or "").strip() in {"1", "1.0"}
    ]
    paper_scores = [float(point.get("final_score") or 0.0) for point in points]
    minimum_score = min(paper_scores, default=0.0)
    maximum_score = max(paper_scores, default=1.0)
    score_span = maximum_score - minimum_score

    def visual_strength(point: dict[str, Any]) -> float:
        score = float(point.get("final_score") or 0.0)
        if score_span <= 1e-9:
            return min(max(score, 0.0), 1.0)
        return min(max((score - minimum_score) / score_span, 0.0), 1.0)

    groups = (
        (
            "排名第一",
            first_ranked,
            [[0.0, "#ffb45c"], [1.0, "#d94f00"]],
        ),
        (
            "高度相关",
            [
                point for point in points
                if point not in first_ranked
                and point.get("relevance_level") == "highly_relevant"
            ],
            [[0.0, "#f3b7ad"], [1.0, "#a61f1f"]],
        ),
        (
            "部分相关",
            [
                point for point in points
                if point not in first_ranked
                and point.get("relevance_level") != "highly_relevant"
            ],
            [[0.0, "#b9d7f2"], [1.0, "#19558c"]],
        ),
    )
    sphere_lighting = {
        "ambient": 0.5,
        "diffuse": 0.8,
        "specular": 0.45,
        "roughness": 0.35,
        "fresnel": 0.15,
    }
    light_position = {"x": 100, "y": 150, "z": 200}
    for label, group, colorscale in groups:
        if not group:
            continue
        paper_mesh = _build_sphere_mesh(
            [
                {
                    "x": point.get("x", 0.0),
                    "y": point.get("y", 0.0),
                    "z": point.get("z", 0.0),
                    "radius": axis_extent * (
                        0.015 + 0.008 * visual_strength(point)
                    ),
                    "intensity": visual_strength(point),
                    "customdata": [
                        point.get("title") or "未命名论文",
                        point.get("rank") or "-",
                        float(point.get("final_score") or 0.0),
                        "是" if point.get("llm_evaluated") else "否",
                    ],
                }
                for point in group
            ],
            axis_extent * 0.01,
        )
        figure.add_trace(go.Mesh3d(
            name=label,
            x=paper_mesh["x"],
            y=paper_mesh["y"],
            z=paper_mesh["z"],
            i=paper_mesh["i"],
            j=paper_mesh["j"],
            k=paper_mesh["k"],
            intensity=paper_mesh["intensity"],
            intensitymode="vertex",
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            showscale=False,
            opacity=1.0,
            flatshading=False,
            lighting=sphere_lighting,
            lightposition=light_position,
            customdata=paper_mesh["customdata"],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "精排第 %{customdata[1]} 名<br>"
                "综合得分 %{customdata[2]:.3f}<br>"
                "DeepSeek 评价：%{customdata[3]}<extra></extra>"
            ),
            showlegend=True,
        ))
    target_mesh = _build_sphere_mesh(
        [{
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "intensity": 1.0,
            "customdata": result.get("query") or "研究目标",
        }],
        axis_extent * 0.016,
        latitude_steps=7,
        longitude_steps=12,
    )
    figure.add_trace(go.Mesh3d(
        name="研究目标",
        x=target_mesh["x"],
        y=target_mesh["y"],
        z=target_mesh["z"],
        i=target_mesh["i"],
        j=target_mesh["j"],
        k=target_mesh["k"],
        color="#f2c14e",
        opacity=1.0,
        flatshading=False,
        lighting=sphere_lighting,
        lightposition=light_position,
        customdata=target_mesh["customdata"],
        hovertemplate="<b>研究目标</b><br>%{customdata}<extra></extra>",
        showlegend=True,
    ))
    figure.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 18, "b": 0},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "bgcolor": "#ffffff",
            "aspectmode": "cube",
            "xaxis": {
                "visible": False,
                "range": [-axis_extent, axis_extent],
                "showspikes": False,
            },
            "yaxis": {
                "visible": False,
                "range": [-axis_extent, axis_extent],
                "showspikes": False,
            },
            "zaxis": {
                "visible": False,
                "range": [-axis_extent, axis_extent],
                "showspikes": False,
            },
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
        },
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    _render_graph_metrics(ranking, points)


def _relationship_edges(
    points: list[dict[str, Any]],
    raw_edges: Any,
) -> list[dict[str, Any]]:
    """Provide root links for results created before graph edges were persisted."""
    edges = [edge for edge in raw_edges or [] if isinstance(edge, dict)]
    if edges:
        return edges
    return [
        {
            "source_id": "__research_target__",
            "target_id": str(point.get("paper_id") or ""),
            "direction": "search",
            "depth": 0,
        }
        for point in points
        if str(point.get("paper_id") or "").strip()
    ]


def _render_2d_relationship_graph(
    result: dict[str, Any],
    ranking: dict[str, Any],
    points: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    go: Any,
) -> None:
    layers: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for point in points:
        depth = min(2, max(0, int(point.get("citation_depth") or 0)))
        layers[depth].append(point)
    positions: dict[str, tuple[float, float]] = {
        "__research_target__": (0.0, 0.0),
    }
    for depth, layer in layers.items():
        layer.sort(key=lambda point: int(point.get("rank") or 10**9))
        count = len(layer)
        spacing = max(0.16, min(0.72, 12.0 / max(count, 1)))
        for index, point in enumerate(layer):
            positions[str(point.get("paper_id") or "")] = (
                (index - (count - 1) / 2.0) * spacing,
                float(-(depth + 1)),
            )

    figure = go.Figure()
    line_x: list[Any] = []
    line_y: list[Any] = []
    for edge in edges:
        source = positions.get(str(edge.get("source_id") or ""))
        target = positions.get(str(edge.get("target_id") or ""))
        if source is None or target is None:
            continue
        line_x.extend([source[0], target[0], None])
        line_y.extend([source[1], target[1], None])
    if line_x:
        figure.add_trace(go.Scatter(
            mode="lines",
            x=line_x,
            y=line_y,
            line={"color": "#a7afb5", "width": 1.2},
            opacity=0.55,
            hoverinfo="skip",
            showlegend=False,
        ))

    scores = [float(point.get("final_score") or 0.0) for point in points]
    minimum = min(scores, default=0.0)
    span = max(max(scores, default=1.0) - minimum, 1e-9)
    groups = (
        ("排名第一", lambda point: int(point.get("rank") or 0) == 1, "#e86f00"),
        (
            "高度相关",
            lambda point: int(point.get("rank") or 0) != 1
            and point.get("relevance_level") == "highly_relevant",
            "#bd3f3f",
        ),
        (
            "部分相关",
            lambda point: int(point.get("rank") or 0) != 1
            and point.get("relevance_level") != "highly_relevant",
            "#4f86b5",
        ),
    )
    for label, predicate, color in groups:
        group = [point for point in points if predicate(point)]
        if not group:
            continue
        figure.add_trace(go.Scatter(
            name=label,
            mode="markers",
            x=[positions[str(point.get("paper_id") or "")][0] for point in group],
            y=[positions[str(point.get("paper_id") or "")][1] for point in group],
            marker={
                "size": [
                    10 + 8 * ((float(point.get("final_score") or 0.0) - minimum) / span)
                    for point in group
                ],
                "color": color,
                "line": {"color": "#ffffff", "width": 1},
                "opacity": 0.92,
            },
            customdata=[[
                point.get("title") or "未命名论文",
                point.get("rank") or "-",
                float(point.get("final_score") or 0.0),
                int(point.get("citation_depth") or 0),
            ] for point in group],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "精排第 %{customdata[1]} 名<br>"
                "综合得分 %{customdata[2]:.3f}<br>"
                "发现层级 %{customdata[3]}<extra></extra>"
            ),
        ))
    figure.add_trace(go.Scatter(
        name="研究目标",
        mode="markers",
        x=[0.0],
        y=[0.0],
        marker={
            "size": 18,
            "color": "#f2c14e",
            "line": {"color": "#9a7112", "width": 1},
        },
        customdata=[result.get("query") or "研究目标"],
        hovertemplate="<b>研究目标</b><br>%{customdata}<extra></extra>",
    ))
    layer_labels = {
        0.0: "研究目标",
        -1.0: "初始检索",
        -2.0: "第一层引文追踪",
        -3.0: "第二层引文追踪",
    }
    figure.update_layout(
        height=680,
        margin={"l": 120, "r": 20, "t": 28, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        xaxis={"visible": False, "fixedrange": False},
        yaxis={
            "tickmode": "array",
            "tickvals": list(layer_labels),
            "ticktext": list(layer_labels.values()),
            "range": [-3.45, 0.35],
            "gridcolor": "#edf0f2",
            "zeroline": False,
            "fixedrange": False,
        },
        hovermode="closest",
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )


def _render_graph_metrics(
    ranking: dict[str, Any],
    points: list[dict[str, Any]],
) -> None:
    first, second, third = st.columns(3)
    first.metric("图中论文", f"{len(points)} 篇")
    second.metric("高度相关", f"{int(ranking.get('highly_relevant_count') or 0)} 篇")
    third.metric("DeepSeek 评价", f"{int(ranking.get('llm_evaluated_count') or 0)} 篇")


def _relevance_label(value: Any) -> str:
    return {
        "highly_relevant": "高度相关",
        "relevant": "相关",
        "partially_relevant": "部分相关",
    }.get(str(value or ""), "未分级")


def render_reader(
    papers: list[dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    result: dict[str, Any],
    *,
    deepseek_api_key: str | None = None,
) -> None:
    ingested_ids = {
        str(item.get("paper_id") or "")
        for item in result.get("stages", {})
        .get("literature_database", {})
        .get("ingestion_results", []) or []
        if isinstance(item, dict) and item.get("success")
    }
    if not ingested_ids:
        ingested_ids = set(outputs)
    available = [
        paper for paper in papers
        if str(paper.get("paper_id") or "") in ingested_ids
    ]
    if not available:
        st.info("本次没有已完成 RAG 处理的论文。")
        return

    title_to_paper = {
        f"{index}. {paper.get('title') or '未命名论文'}": paper
        for index, paper in enumerate(available, start=1)
    }
    selected_title = st.selectbox("选择论文", list(title_to_paper))
    paper = title_to_paper[selected_title]
    output = outputs.get(str(paper.get("paper_id") or ""), {})
    summary_tab, knowledge_tab, answer_tab = st.tabs(["摘要", "知识抽取", "问答"])

    with summary_tab:
        render_mapping(output.get("summary"), SUMMARY_LABELS, "本论文没有生成智能摘要。")
        if output.get("summary_error"):
            st.error(str(output["summary_error"]))
    with knowledge_tab:
        render_mapping(output.get("knowledge"), KNOWLEDGE_LABELS, "本论文没有生成知识抽取结果。")
        if output.get("knowledge_error"):
            st.error(str(output["knowledge_error"]))
    with answer_tab:
        render_interactive_qa(
            paper,
            output,
            deepseek_api_key=deepseek_api_key,
        )


def render_interactive_qa(
    paper: dict[str, Any],
    output: dict[str, Any],
    *,
    deepseek_api_key: str | None,
) -> None:
    paper_id = str(paper.get("paper_id") or "")
    history = output.setdefault("interactive_answers", [])
    if not isinstance(history, list):
        history = []
        output["interactive_answers"] = history
    form_key = str(abs(hash(paper_id)))

    with st.form(f"paper_question_{form_key}", clear_on_submit=True):
        question = st.text_area(
            "向当前论文提问",
            placeholder="例如：论文使用了哪些数据集和评价指标？",
            height=88,
        )
        submitted = st.form_submit_button(
            "发送问题",
            type="primary",
            use_container_width=False,
        )

    if submitted:
        if not question.strip():
            st.warning("请输入问题。")
        else:
            try:
                with st.spinner("正在检索论文证据并生成回答"):
                    answer = ask_paper_question(
                        paper_id,
                        question,
                        deepseek_api_key=deepseek_api_key,
                    )
                history.append(answer)
            except Exception as exc:
                st.error(f"问答失败：{type(exc).__name__}: {exc}")

    answers = []
    original_answer = output.get("answer")
    if isinstance(original_answer, dict):
        answers.append(original_answer)
    answers.extend(item for item in history if isinstance(item, dict))
    if not answers:
        st.info("还没有向这篇论文提出问题。")
    for answer in answers:
        with st.chat_message("user"):
            st.write(answer.get("question") or "")
        with st.chat_message("assistant"):
            st.write(answer.get("answer") or "")
            confidence = answer.get("confidence")
            if confidence is not None:
                st.caption(f"回答置信度：{confidence}")
            citations = [
                item for item in answer.get("citations") or []
                if isinstance(item, dict)
            ]
            sources = answer.get("sources") or []
            if citations:
                with st.expander(f"证据片段（{len(citations)}）"):
                    for index, citation in enumerate(citations, start=1):
                        st.markdown(f"**{index}. {citation.get('title') or '论文证据'}**")
                        location = []
                        if citation.get("paper_id"):
                            location.append(str(citation["paper_id"]))
                        if citation.get("page") is not None:
                            location.append(f"第 {citation['page']} 页")
                        if citation.get("section"):
                            location.append(str(citation["section"]))
                        if citation.get("chunk_id"):
                            location.append(str(citation["chunk_id"]))
                        if location:
                            st.caption(" · ".join(location))
                        st.write(str(citation.get("excerpt") or ""))
            elif sources:
                with st.expander(f"证据片段（{len(sources)}）"):
                    for source in sources:
                        st.code(str(source), language="text")
    if output.get("answer_error"):
        st.error(str(output["answer_error"]))


def render_mapping(value: Any, labels: dict[str, str], empty_message: str) -> None:
    if not isinstance(value, dict) or not value:
        st.info(empty_message)
        return
    for key, content in value.items():
        if key == "title" or not content:
            continue
        st.markdown(
            f"<div class='ap-section-title'>{labels.get(key, key)}</div>",
            unsafe_allow_html=True,
        )
        st.write(content)
        st.divider()


def render_overview(result: dict[str, Any]) -> None:
    times = stage_times(result)
    if times:
        maximum = max(seconds for _, seconds in times) or 1
        st.markdown("#### 各阶段耗时")
        for name, seconds in times:
            left, right = st.columns([3, 1])
            left.write(STAGE_LABELS.get(name, name))
            left.progress(min(seconds / maximum, 1.0))
            right.write(format_duration(seconds))

    citation_stage = result.get("stages", {}).get("citation_tracking", {})
    citation = citation_stage.get("result", {}) if isinstance(citation_stage, dict) else {}
    if isinstance(citation, dict) and citation:
        st.markdown("#### 引文追踪")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("种子论文", citation.get("seed_count", 0))
        col2.metric("第一层保留", citation.get("first_layer_retained_count", 0))
        col3.metric("第二层保留", citation.get("second_layer_retained_count", 0))
        col4.metric("追踪新增", citation.get("retained_count", 0))
        col5.metric("合并候选池", citation.get("merged_paper_count", 0))
        cache_hits = int(citation.get("cache_hit_count") or 0)
        api_calls = int(citation.get("api_call_count") or 0)
        cutoff = citation.get("publication_date_to") or "未限制"
        st.caption(
            f"OpenAlex API 调用 {api_calls} 次，缓存命中 {cache_hits} 次，"
            f"论文日期截止：{cutoff}"
        )

    literature = result.get("stages", {}).get("literature_database", {})
    st.markdown("#### 文献处理")
    col1, col2, col3 = st.columns(3)
    col1.metric("选中论文", literature.get("selected_paper_count", 0))
    col2.metric("下载功能", "已启用" if literature.get("download_enabled") else "未启用")
    col3.metric("RAG 摄取", "已启用" if literature.get("rag_ingest_enabled") else "未启用")
    ingestion = literature.get("ingestion_results") or []
    downloads = literature.get("download_results") or []
    if downloads:
        with st.expander("查看 Top 论文下载明细"):
            render_download_details(downloads)
    if ingestion:
        with st.expander("查看论文处理明细"):
            render_ingestion_details(ingestion, result)


def render_download_details(downloads: list[dict[str, Any]]) -> None:
    rows = []
    for item in downloads:
        if not isinstance(item, dict):
            continue
        success = bool(item.get("success"))
        if success and item.get("skipped"):
            status = "已存在，直接复用"
        elif success:
            status = "下载成功"
        elif item.get("need_user_upload"):
            status = "需要手动上传"
        else:
            status = "下载失败"
        file_path = str(item.get("file_path") or "")
        rows.append({
            "原始排名": item.get("original_rank") or None,
            "论文": item.get("title") or item.get("paper_id") or "未命名论文",
            "状态": status,
            "下载来源": item.get("engine_used") or "-",
            "本地文件": Path(file_path).name if file_path else "-",
            "说明": item.get("error") or ("文件已准备完成" if success else "未获得可用 PDF"),
        })
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "原始排名": st.column_config.NumberColumn(width="small"),
                "论文": st.column_config.TextColumn(width="large"),
                "状态": st.column_config.TextColumn(width="medium"),
                "下载来源": st.column_config.TextColumn(width="small"),
                "本地文件": st.column_config.TextColumn(width="large"),
                "说明": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("没有可显示的下载记录。")


def render_ingestion_details(
    ingestion: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    papers = selected_papers(result)
    titles = {
        str(paper.get("paper_id") or ""): paper.get("title") or "未命名论文"
        for paper in papers
    }
    rows = []
    for item in ingestion:
        if not isinstance(item, dict):
            continue
        success = bool(item.get("success"))
        text_path = str(item.get("text_path") or "")
        paper_id = str(item.get("paper_id") or "")
        rows.append({
            "论文": titles.get(paper_id, paper_id or "未命名论文"),
            "处理状态": "RAG 处理完成" if success else "RAG 处理失败",
            "可检索片段": int(item.get("chunk_count") or 0),
            "资源状态": _availability_label(item.get("status")),
            "文本文件": Path(text_path).name if text_path else "-",
            "说明": item.get("error") or ("已写入向量知识库" if success else "未完成向量化"),
        })
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "论文": st.column_config.TextColumn(width="large"),
                "处理状态": st.column_config.TextColumn(width="medium"),
                "可检索片段": st.column_config.NumberColumn(width="small"),
                "资源状态": st.column_config.TextColumn(width="small"),
                "文本文件": st.column_config.TextColumn(width="medium"),
                "说明": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.info("没有可显示的处理记录。")


def _availability_label(value: Any) -> str:
    labels = {
        "FULL_AVAILABLE": "全文可用",
        "ABSTRACT_AVAILABLE": "仅摘要可用",
        "MISSING": "全文缺失",
        "UNAVAILABLE": "不可用",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def render_exports(result: dict[str, Any], logs: Any = None) -> None:
    st.markdown("#### 下载本次研究结果")
    st.caption("Markdown 汇总 Top 论文分析，CSV 保存去重并按评分排序的完整论文列表，日志用于运行排障。")
    markdown = markdown_report(result)
    all_papers = ranked_papers(result)
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "下载 Top 论文 Markdown 报告",
        data=markdown.encode("utf-8"),
        file_name="autopaper_top_papers_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    col2.download_button(
        "下载完整精排论文 CSV",
        data=papers_csv_bytes(all_papers),
        file_name="autopaper_ranked_papers.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if isinstance(logs, list) and logs:
        col3.download_button(
            "下载运行日志",
            data=("\n".join(str(line) for line in logs) + "\n").encode("utf-8"),
            file_name="scholartrace_job.log",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        col3.button("下载运行日志", disabled=True, use_container_width=True)
    with st.expander("预览 Markdown 报告"):
        st.markdown(markdown)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="ap-empty">
          <div>
            <strong>从一个研究问题开始</strong>
            在左侧输入主题并选择处理功能，运行后将在这里查看论文、摘要和证据问答。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_jobs() -> None:
    if not st.session_state.get("api_connected"):
        return
    try:
        jobs = recent_workflows(10)
    except WorkflowRunError:
        return
    if not jobs:
        return
    with st.sidebar.expander("最近任务"):
        labels = [f"{item['status']} · {item['query'][:35]} · {item['job_id'][:8]}" for item in jobs]
        choice = st.selectbox("选择已有任务", range(len(jobs)), format_func=lambda i: labels[i])
        if st.button("恢复查看任务", use_container_width=True):
            selected = jobs[choice]
            if selected["status"] == "completed":
                _store_completed_workflow(selected)
                st.session_state.pop("active_workflow_job_id", None)
            elif selected["status"] in {"queued", "running", "cancelling"}:
                st.session_state["active_workflow_job_id"] = selected["job_id"]
            else:
                st.session_state["workflow_notice"] = {
                    "level": "warning",
                    "message": str(selected.get("error") or f"任务状态：{selected['status']}"),
                    "logs": selected.get("logs"),
                }
            st.rerun()


def main() -> None:
    apply_styles()
    render_header()
    workflow_was_active = bool(st.session_state.get("active_workflow_job_id"))
    options, run_clicked = render_controls()
    render_recent_jobs()
    if run_clicked:
        if workflow_was_active:
            cancel_active_workflow()
        else:
            execute(options)
    render_active_workflow()
    render_workflow_notice()
    result = st.session_state.get("result")
    if isinstance(result, dict):
        render_result(result)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
