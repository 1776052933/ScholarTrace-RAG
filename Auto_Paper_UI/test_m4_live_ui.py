"""Opt-in Streamlit widget acceptance against a running local M4 API."""
from __future__ import annotations

import os
from pathlib import Path
from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from Auto_Paper_UI.api_client import poll_workflow


APP_PATH = Path(__file__).with_name("ui_app.py")


@pytest.mark.skipif(not os.getenv("M4_DISCOVERY_JOB_ID"), reason="requires a local seeded discovery job")
def test_live_candidate_selection_reuses_durable_ingestion():
    source_id = os.environ["M4_DISCOVERY_JOB_ID"]
    discovery = poll_workflow(source_id)
    assert discovery["status"] == "completed"

    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=25)
    result = deepcopy(discovery["result"])
    result["stages"].setdefault("literature_database", {"download_enabled": False})
    app.session_state["result"] = result
    app.session_state["last_workflow_job_id"] = source_id
    app.run(timeout=25)
    assert not app.exception
    assert any(widget.label == "候选论文" for widget in app.multiselect)
    assert any(widget.label == "论文库" for widget in app.multiselect)

    next(button for button in app.button if button.label == "下载并建立论文库").click()
    app.run(timeout=25)
    assert not app.exception
    ingestion_id = app.session_state["last_ingestion_job_id"]
    snapshot = poll_workflow(ingestion_id)
    assert snapshot["job_type"] == "ingestion"
    assert snapshot["status"] == "completed"
    assert len(snapshot["result"]["papers"]) == 2
