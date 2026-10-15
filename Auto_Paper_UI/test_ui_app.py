from pathlib import Path

from streamlit.testing.v1 import AppTest

from Auto_Paper_UI.ui_app import _relationship_edges


APP_PATH = Path(__file__).with_name("ui_app.py")


def _result():
    paper = {
        "paper_id": "paper-1",
        "title": "Example paper",
        "abstract": "Example abstract",
        "authors": ["Researcher"],
        "year": 2025,
        "citation_count": 3,
        "url": "https://example.test/paper-1",
    }
    return {
        "status": "completed",
        "query": "example research topic",
        "elapsed_seconds": 12.0,
        "stages": {
            "search_agent": {
                "elapsed_seconds": 2.0,
                "result": {"papers": [paper]},
            },
            "ranking_agent": {
                "elapsed_seconds": 3.0,
                "result": {
                    "papers": [paper],
                    "total": 1,
                    "scores": {
                        "paper-1": {
                            "semantic_score": 0.9,
                            "method_score": 0.8,
                            "task_score": 0.9,
                            "time_score": 0.9,
                            "quality_score": 0.7,
                            "llm_score": 0.9,
                            "final_score": 0.86,
                            "relevance_level": "highly_relevant",
                        },
                    },
                    "graph_points": [{
                        "paper_id": "paper-1",
                        "title": "Example paper",
                        "rank": 1,
                        "x": 0.1,
                        "y": 0.1,
                        "z": 0.1,
                        "distance": 0.17,
                        "final_score": 0.86,
                        "relevance_level": "highly_relevant",
                        "llm_evaluated": True,
                        "citation_depth": 0,
                        "relation_keys": ["title:examplepaper"],
                    }],
                    "graph_edges": [{
                        "source_id": "__research_target__",
                        "target_id": "paper-1",
                        "direction": "search",
                        "depth": 0,
                    }],
                    "llm_evaluated_count": 1,
                    "highly_relevant_count": 1,
                    "adaptive_threshold": 0.86,
                },
            },
            "literature_database": {
                "elapsed_seconds": 4.0,
                "selected_paper_count": 1,
                "download_enabled": True,
                "rag_ingest_enabled": True,
                "download_success_count": 1,
                "download_results": [{
                    "paper_id": "paper-1",
                    "title": "Example paper",
                    "success": True,
                    "file_path": "papers/example.pdf",
                    "engine_used": "arxiv",
                    "skipped": False,
                    "error": None,
                }],
                "ingestion_results": [{
                    "paper_id": "paper-1",
                    "success": True,
                    "chunk_count": 12,
                    "text_path": "papers/text/example.txt",
                    "status": "FULL_AVAILABLE",
                    "error": None,
                }],
            },
            "rag_reader": {
                "processed_paper_count": 1,
                "outputs": {"paper-1": {}},
            },
        },
    }


def test_completed_run_has_readable_details_and_interactive_question_form():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    assert not app.exception

    app.session_state["result"] = _result()
    app.session_state["output_path"] = "result.json"
    app.run(timeout=20)

    assert not app.exception
    assert len(app.dataframe) == 2
    assert any(area.label == "向当前论文提问" for area in app.text_area)
    assert not any(area.label == "问题" for area in app.text_area)
    assert any(button.label == "开始检索" for button in app.button)
    assert any(toggle.label == "启用引文追踪" for toggle in app.toggle)
    assert any(
        number.label == "论文发表截止年份（可选）"
        for number in app.number_input
    )
    assert not any(selectbox.label == "日志级别" for selectbox in app.selectbox)
    assert not any(tab.label == "运行日志" for tab in app.tabs)
    assert any(tab.label == "论文关系图" for tab in app.tabs)
    download_buttons = app.get("download_button")
    assert any(
        button.label == "下载 Top 论文 Markdown 报告"
        for button in download_buttons
    )
    assert any(
        button.label == "下载完整精排论文 CSV"
        for button in download_buttons
    )
    assert not any(button.label == "下载完整 JSON" for button in download_buttons)


def test_old_graph_results_receive_research_target_edges():
    edges = _relationship_edges(
        [{"paper_id": "paper-1"}, {"paper_id": "paper-2"}],
        [],
    )

    assert edges == [
        {
            "source_id": "__research_target__",
            "target_id": "paper-1",
            "direction": "search",
            "depth": 0,
        },
        {
            "source_id": "__research_target__",
            "target_id": "paper-2",
            "direction": "search",
            "depth": 0,
        },
    ]

def test_answer_panel_shows_locatable_evidence():
    result = _result()
    result["stages"]["rag_reader"]["outputs"]["paper-1"] = {
        "answer": {
            "question": "What changed?",
            "answer": "Attention replaced recurrence.",
            "confidence": 0.8,
            "sources": ["paper-1-chunk-2"],
            "citations": [{
                "paper_id": "paper-1",
                "chunk_id": "paper-1-chunk-2",
                "title": "Transformer source",
                "page": 2,
                "section": "Methods",
                "excerpt": "Attention replaces recurrent layers.",
            }],
        },
    }
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    app.session_state["result"] = result
    app.run(timeout=20)

    assert not app.exception
    visible_text = "\n".join(str(item.value) for item in [*app.markdown, *app.get("caption")])
    assert "Transformer source" in visible_text
    assert "第 2 页" in visible_text
    assert "Methods" in visible_text
    assert "Attention replaces recurrent layers." in visible_text


def test_partial_or_failed_job_keeps_result_and_shows_warning(monkeypatch):
    from Auto_Paper_UI import ui_app

    state = {}
    monkeypatch.setattr(ui_app.st, "session_state", state)
    for status, level in (("partial_success", "warning"), ("failed", "error")):
        state.clear()
        snapshot = {
            "status": status,
            "result": {"status": status, "stages": {"literature_database": {
                "ingestion_results": [{"paper_id": "paper-1", "success": False, "error": "index error"}],
            }}},
            "error": "入库失败 1 篇，请查看论文处理明细。",
            "logs": ["论文处理失败"],
        }
        ui_app._record_finished_workflow(snapshot)
        assert state["result"] == snapshot["result"]
        assert state["workflow_notice"]["level"] == level
        assert "入库失败 1 篇" in state["workflow_notice"]["message"]
        assert state["logs"] == snapshot["logs"]


def test_exported_report_names_partial_status_and_failed_paper():
    from Auto_Paper_UI.reporting import markdown_report

    result = _result()
    result["status"] = "partial_success"
    result["stages"]["literature_database"]["ingestion_results"][0].update(
        success=False, error="index error"
    )
    report = markdown_report(result)
    assert "运行状态：部分完成" in report
    assert "入库失败：paper-1；原因：index error" in report
