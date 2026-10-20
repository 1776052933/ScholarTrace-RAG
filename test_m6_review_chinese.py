"""Chinese review coverage, rendering and safe persistence; never edit the real sheet."""
import json
from pathlib import Path

import pytest

from evaluation.m6_review_translations import load_translations, translate


def test_translation_requires_exact_source():
    catalog = load_translations()
    assert translate("High sample utilization", catalog) == "样本利用率高"
    assert translate("High sample utilization changed", catalog) is None


@pytest.fixture
def local_review():
    from evaluation import m6_review_ui as ui
    if not ui.SHEET.is_file() or not ui.ANSWERS_DIR.is_dir():
        pytest.skip("Local 44-answer review assets not installed")
    return ui


def test_all_cached_sources_have_chinese(local_review):
    catalog = load_translations()
    records = local_review._answer_records()
    assert len(records) == 44
    for row in local_review._rows():
        assert translate(row["question"], catalog)
        assert translate(row["answer"], catalog)
    for item in records.values():
        for cite in item.get("citations") or []:
            assert translate(cite["excerpt"], catalog)


def test_render_and_save_only_temporary_sheet(local_review, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    original_path = local_review.SHEET
    original_bytes = original_path.read_bytes()
    temporary_sheet = tmp_path / "review.csv"
    temporary_sheet.write_bytes(original_bytes)
    monkeypatch.setattr(local_review, "SHEET", temporary_sheet)
    # The user may have reviewed any rows since the previous test run.
    # Reset judgments in the temporary copy only to exercise a fixed first row.
    initial_rows = local_review._rows()
    for row in initial_rows:
        for field in ("answer_correct", "citations_supported", "error_type", "reviewer", "notes"):
            row[field] = ""
    local_review._save(initial_rows)
    initial_bytes = temporary_sheet.read_bytes()

    def app():
        from evaluation.m6_review_ui import main
        main()

    at = AppTest.from_function(app).run(timeout=60)
    assert not at.exception
    assert "中文辅助" in at.title[0].value
    assert at.checkbox[1].value is False
    assert any("编码器" in block.value for block in at.markdown)
    assert not any("自动排查提示：" in warning.value for warning in at.warning)
    assert temporary_sheet.read_bytes() == initial_bytes

    # Submit and reload through the actual form; this must only touch the copy.
    at.radio[0].set_value("yes")
    at.radio[1].set_value("no")
    at.text_input[0].set_value("UI test reviewer")
    at.button[0].click().run(timeout=60)
    assert not at.exception
    saved = local_review._rows()
    assert saved[0]["answer_correct"] == "yes"
    assert saved[0]["citations_supported"] == "no"
    assert saved[0]["reviewer"] == "UI test reviewer"
    assert original_path.read_bytes() == original_bytes

    # An invalid replacement must leave the last valid saved file intact.
    valid_bytes = temporary_sheet.read_bytes()
    saved[0]["answer_correct"] = "invalid"
    with pytest.raises(ValueError):
        local_review._save(saved)
    assert temporary_sheet.read_bytes() == valid_bytes
