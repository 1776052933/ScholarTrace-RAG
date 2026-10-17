"""M4 Streamlit interaction checks without network or model calls."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from Auto_Paper_UI.test_ui_app import _result


APP_PATH = Path(__file__).with_name("ui_app.py")


def _offline_api(monkeypatch, *, connected: bool = True):
    from Auto_Paper_UI import ui_app  # noqa: F401 - add UI directory to import path
    import api_client

    monkeypatch.setattr(api_client, "api_status", lambda: {
        "connected": connected, "database": connected,
        "deepseek": False, "serpapi": False, "openalex": False,
    })
    monkeypatch.setattr(api_client, "recent_workflows", lambda _limit: [])
    monkeypatch.setattr(api_client, "paper_library", lambda: [])
    return api_client


def test_candidate_selection_submits_selected_paper_ids(monkeypatch):
    ui_app = _offline_api(monkeypatch)
    submissions = []
    monkeypatch.setattr(ui_app, "start_selected_ingestion", lambda job_id, paper_ids: (
        submissions.append((job_id, paper_ids)) or {"job_id": "ingestion-1"}
    ))
    monkeypatch.setattr(ui_app, "poll_workflow", lambda _job_id: {
        "status": "queued", "result": {"papers": []}, "progress": 0,
    })

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    result = _result()
    result["stages"]["literature_database"].update(
        download_enabled=False, download_results=[], ingestion_results=[]
    )
    app.session_state["result"] = result
    app.session_state["last_workflow_job_id"] = "discovery-1"
    app.run(timeout=20)
    assert not app.exception
    assert any(widget.label == "候选论文" for widget in app.multiselect)
    next(button for button in app.button if button.label == "下载并建立论文库").click()
    app.run(timeout=20)
    assert not app.exception
    assert submissions == [("discovery-1", ["paper-1"])]
    assert app.session_state["last_ingestion_job_id"] == "ingestion-1"


def test_library_multi_paper_answer_and_export(monkeypatch):
    ui_app = _offline_api(monkeypatch)
    monkeypatch.setattr(ui_app, "paper_library", lambda: [
        {"paper_id": "paper-1", "title": "First", "embedding_status": "COMPLETED"},
        {"paper_id": "paper-2", "title": "Second", "embedding_status": "COMPLETED"},
    ])
    calls = []
    monkeypatch.setattr(ui_app, "ask_multiple_papers", lambda ids, question: (
        calls.append((ids, question)) or {
            "answer_id": "answer-1", "answer": {
                "answer": "Both papers agree.", "citations": [
                    {"paper_id": "paper-1", "title": "First", "page": 2, "excerpt": "Evidence"},
                ],
            },
        }
    ))
    monkeypatch.setattr(ui_app, "export_answer", lambda _id: "# Exported answer")

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    next(widget for widget in app.multiselect if widget.label == "论文库").set_value(
        ["paper-1", "paper-2"]
    )
    next(widget for widget in app.text_input if widget.label == "跨论文问题").set_value("Compare")
    app.run(timeout=20)
    next(button for button in app.button if button.label == "根据所选论文回答").click()
    app.run(timeout=20)
    assert not app.exception
    assert calls == [(["paper-1", "paper-2"], "Compare")]
    assert any("Both papers agree." in str(item.value) for item in app.markdown)
    assert any(button.label == "导出本次问答" for button in app.get("download_button"))
