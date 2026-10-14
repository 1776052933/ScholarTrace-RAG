import json
import os
from pathlib import Path
import sys
import time

from Auto_Paper_UI import backend
from Auto_Paper_UI.reporting import papers_csv_bytes, ranked_papers, selected_papers


class FakeProcess:
    def __init__(self, command, **kwargs) -> None:
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"status": "completed", "stages": {}}',
            encoding="utf-8",
        )
        secret = kwargs["env"].get("DEEPSEEK_API_KEY", "")
        self.stdout = iter([
            f"starting DEEPSEEK_API_KEY={secret}\n",
            "finished\n",
            "{\n",
        ])

    def wait(self):
        return 0


def test_log_redaction_removes_known_and_pattern_keys():
    line = "DEEPSEEK_API_KEY=secret-value token=sk-abcdefghijklmnop"
    sanitized = backend._redact_log_line(line, ["secret-value"])

    assert "secret-value" not in sanitized
    assert "sk-abcdefghijklmnop" not in sanitized
    assert sanitized.count("[REDACTED]") == 2


def test_output_retention_keeps_the_latest_thirty_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path)
    for index in range(32):
        for suffix in (".json", ".log"):
            path = tmp_path / f"result_{index:02d}{suffix}"
            path.write_text("test", encoding="utf-8")
            os.utime(path, (index + 1, index + 1))

    backend._prune_output_history(max_runs=30)

    assert len(list(tmp_path.glob("*.json"))) == 30
    assert len(list(tmp_path.glob("*.log"))) == 30
    assert not (tmp_path / "result_00.json").exists()
    assert not (tmp_path / "result_01.log").exists()


def test_workflow_saves_a_matching_redacted_log(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(backend, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(backend.subprocess, "Popen", FakeProcess)
    options = backend.RunOptions(
        query="test query",
        deepseek_api_key="secret-deepseek-key",
    )

    run = backend.run_workflow(options)

    assert run.output_path.stem == run.log_path.stem
    assert run.log_path.is_file()
    content = run.log_path.read_text(encoding="utf-8")
    assert "secret-deepseek-key" not in content
    assert "[REDACTED]" in content


def test_command_passes_optional_year_and_citation_controls(tmp_path):
    enabled = backend.build_command(
        backend.RunOptions(query="test query"),
        tmp_path / "enabled.json",
    )
    disabled = backend.build_command(
        backend.RunOptions(
            query="test query",
            enable_citation_tracking=False,
            publication_date_to="2024-12-31",
        ),
        tmp_path / "disabled.json",
    )

    assert "--skip-citation-tracking" not in enabled
    assert "--publication-date-to" not in enabled
    assert "--skip-citation-tracking" in disabled
    cutoff_index = disabled.index("--publication-date-to")
    assert disabled[cutoff_index + 1] == "2024-12-31"


def test_background_workflow_can_be_cancelled_and_records_state(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(backend, "RUNTIME_ROOT", tmp_path / "runtime")

    def sleeping_command(options, output_path):
        return [
            sys.executable,
            "-c",
            (
                "import os,time; "
                "print('DEEPSEEK_API_KEY=' + os.environ.get('DEEPSEEK_API_KEY',''), flush=True); "
                "time.sleep(30)"
            ),
        ]

    monkeypatch.setattr(backend, "build_command", sleeping_command)
    options = backend.RunOptions(
        query="cancel test",
        deepseek_api_key="secret-cancel-key",
    )

    job_id = backend.start_workflow(options)
    part_file = backend.RUNTIME_ROOT / "papers" / "unfinished.pdf.part"
    part_file.write_bytes(b"partial")
    time.sleep(0.2)
    snapshot = backend.cancel_workflow(job_id)

    assert snapshot["status"] == "cancelled"
    assert snapshot["output_path"].stem == snapshot["log_path"].stem
    assert not part_file.exists()
    cancellation = json.loads(snapshot["output_path"].read_text(encoding="utf-8"))
    assert cancellation["status"] == "cancelled"
    log_text = snapshot["log_path"].read_text(encoding="utf-8")
    assert "secret-cancel-key" not in log_text
    assert "[REDACTED]" in log_text
    assert "用户已取消本次检索" in log_text


def test_background_workflow_returns_completed_result(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(backend, "RUNTIME_ROOT", tmp_path / "runtime")

    def completing_command(options, output_path):
        payload = json.dumps({"status": "completed", "query": options.query})
        script = (
            "from pathlib import Path; "
            f"Path({str(output_path)!r}).write_text({payload!r}, encoding='utf-8'); "
            "print('finished', flush=True)"
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(backend, "build_command", completing_command)
    job_id = backend.start_workflow(backend.RunOptions(query="complete test"))
    deadline = time.monotonic() + 5
    snapshot = backend.poll_workflow(job_id)
    while snapshot["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        snapshot = backend.poll_workflow(job_id)

    assert snapshot["status"] == "completed"
    assert snapshot["result"] == {"status": "completed", "query": "complete test"}
    assert snapshot["log_path"].read_text(encoding="utf-8").strip() == "finished"


def test_csv_uses_all_ranked_papers_not_only_top_papers():
    result = {
        "stages": {
            "search_agent": {"result": {"papers": [
                {"title": "First", "abstract": "A"},
                {"title": "Second", "abstract": "B"},
                {"title": "Search duplicate", "abstract": "C"},
            ]}},
            "ranking_agent": {"result": {"papers": [
                {"title": "First", "abstract": "A"},
                {"title": "Second", "abstract": "B"},
            ]}},
            "literature_database": {"selected_paper_count": 1},
        }
    }

    assert len(ranked_papers(result)) == 2
    assert len(selected_papers(result)) == 1
    csv_text = papers_csv_bytes(ranked_papers(result)).decode("utf-8-sig")
    assert "First" in csv_text
    assert "Second" in csv_text
    assert "Search duplicate" not in csv_text
    assert "摘要" in csv_text


def test_selected_papers_use_downloaded_ids_and_keep_original_rank():
    result = {
        "stages": {
            "ranking_agent": {"result": {"papers": [
                {"paper_id": "paper-1", "title": "First"},
                {"paper_id": "paper-2", "title": "Second"},
                {"paper_id": "paper-3", "title": "Third"},
                {"paper_id": "paper-4", "title": "Fourth"},
            ]}},
            "literature_database": {
                "selected_paper_count": 2,
                "selected_paper_ids": ["paper-3", "paper-4"],
            },
        }
    }

    papers = selected_papers(result)

    assert [paper["paper_id"] for paper in papers] == ["paper-3", "paper-4"]
    assert [paper["original_rank"] for paper in papers] == [3, 4]
