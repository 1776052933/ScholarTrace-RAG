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
    import conversation_panel
    sessions, turns = [], []
    answer = {"question": "Compare", "answer": "Both papers agree.", "confidence": 0.9,
              "citations": [{"paper_id": "paper-1", "title": "First", "page": 2, "excerpt": "Evidence"}]}
    monkeypatch.setattr(conversation_panel, "recent_conversations", lambda ids: sessions)

    def create(ids):
        sessions.append({"conversation_id": "conversation-1", "title": "First / Second",
                         "updated_at": "2026-10-19T12:00", "paper_ids": ids})
        return sessions[0]

    def stream(cid, question):
        calls.append((cid, question))
        yield {"event": "delta", "data": {"text": "Both papers "}}
        yield {"event": "delta", "data": {"text": "agree."}}
        turn_id = f"answer-{len(turns) + 1}"
        turns.append({"turn_id": turn_id, "turn_number": len(turns) + 1, "status": "completed",
                      "question": question, "answer": {**answer, "question": question}})
        yield {"event": "done", "data": {"answer_id": "answer-1", "answer": answer}}

    monkeypatch.setattr(conversation_panel, "create_conversation", create)
    monkeypatch.setattr(conversation_panel, "stream_conversation", stream)
    monkeypatch.setattr(conversation_panel, "get_conversation", lambda cid: {"turns": turns})
    monkeypatch.setattr(conversation_panel, "export_answer", lambda _id: "# Exported answer")

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=20)
    next(widget for widget in app.multiselect if widget.label == "论文库").set_value(
        ["paper-1", "paper-2"]
    )
    app.run(timeout=20)
    next(widget for widget in app.text_area if widget.label == "论文问题").set_value("Compare")
    next(button for button in app.button if button.label == "根据所选论文回答").click()
    app.run(timeout=20)
    assert not app.exception
    assert calls == [("conversation-1", "Compare")]
    assert any("Both papers agree." in str(item.value) for item in app.markdown)
    next(button for button in app.button if button.label == "准备导出本次问答").click()
    app.run(timeout=20)
    assert not app.exception
    assert any(button.label == "导出本次问答" for button in app.get("download_button"))
    next(widget for widget in app.text_area if widget.label == "论文问题").set_value("What about its limits?")
    next(button for button in app.button if button.label == "根据所选论文回答").click()
    app.run(timeout=20)
    assert not app.exception
    assert calls[-1] == ("conversation-1", "What about its limits?")
    assert len(turns) == 2 and len(sessions) == 1
    # A fresh browser session restores server history after reselecting the same papers.
    restored = AppTest.from_file(str(APP_PATH)).run(timeout=20)
    next(widget for widget in restored.multiselect if widget.label == "论文库").set_value(["paper-1", "paper-2"])
    restored.run(timeout=20)
    assert not restored.exception
    assert any("What about its limits?" in str(item.value) for item in restored.markdown)
