"""Bounded live deployment regression on a dedicated Compose project.

Phases: ingest (local PDFs through Celery), model (6 logical provider operations),
verify (read-only restart checks). No scoring or modifying the 44-row review.
Run inside API with --output under its persistent data root. A completed/started
model phase is not repeated automatically. Inspect saved failures before reruns.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

import httpx

PAPERS = ["repro-attention", "repro-dexterous"]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def request(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def ingest(client, report):
    from app.storage.models import ResearchJobModel
    from literature_database.database import DatabaseManager
    from literature_database.models import PaperFileModel, PaperModel
    from uuid import uuid4

    db = DatabaseManager(os.environ["DATABASE_URL"])
    job_id = uuid4().hex
    fixtures = [("attention.pdf", "Attention Is All You Need"),
                ("dexterous.pdf", "Dexterous Manipulation for Multi-Fingered Robotic Hands With Reinforcement Learning: A Review")]
    metadata = []
    try:
        previous = report.get("ingestion")
        if previous:
            job_id = None
            with db.get_session() as session:
                existing = session.get(ResearchJobModel, previous["job"]["job_id"])
                assert existing is not None and existing.job_type == "ingestion"
                source = session.get(ResearchJobModel, existing.options["discovery_job_id"])
                assert source is not None and source.query == "Reproducible deployment fixtures"
                paper = session.get(PaperModel, PAPERS[1])
                assert paper is not None
                correct_title = fixtures[1][1]
                expected_pdf = Path("/app/data/m2/fixtures/dexterous.pdf")
                assert expected_pdf.is_file()
                files = [f for f in session.query(PaperFileModel).filter_by(paper_id=PAPERS[1]).all()
                         if Path(f.file_path) == expected_pdf and f.file_type == "pdf"]
                assert len(files) == 1 and files[0].status in {"MISMATCH", "AVAILABLE"}
                assert previous["fixtures"][1]["sha256"] == hashlib.sha256(expected_pdf.read_bytes()).hexdigest()
                paper.title = correct_title
                files[0].status = "AVAILABLE"
                result = deepcopy(source.result)
                candidates = result["stages"]["ranking_agent"]["result"]["papers"]
                next(item for item in candidates if item["paper_id"] == PAPERS[1])["title"] = correct_title
                source.result = result
                result = deepcopy(existing.result)
                next(item for item in result["papers"] if item["paper_id"] == PAPERS[1])["title"] = correct_title
                existing.result = result
                session.commit()
            previous["fixtures"][1]["title"] = correct_title
            job = request(client, "POST", f"/api/v1/ingestions/{previous['job']['job_id']}/retry")
        else:
            with db.get_session() as session:
                for pid, (name, title) in zip(PAPERS, fixtures):
                    path = Path("/app/data/m2/fixtures") / name
                    assert path.is_file(), f"Missing fixture: {path}"
                    assert session.get(PaperModel, pid) is None, "Use an empty acceptance database"
                    session.add(PaperModel(paper_id=pid, title=title, authors=[]))
                    session.flush()
                    session.add(PaperFileModel(paper_id=pid, file_path=str(path),
                                              status="AVAILABLE", file_type="pdf"))
                    metadata.append({"paper_id": pid, "title": title,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                session.add(ResearchJobModel(
                    job_id=job_id, job_type="research", query="Reproducible deployment fixtures",
                    options={"query": "local acceptance", "download_papers": False},
                    status="completed", stage="ranking_agent", progress=100,
                    logs=[], retryable=False,
                    result={"stages": {"ranking_agent": {"result": {"papers": metadata}}}},
                ))
                session.commit()
    finally:
        db.dispose()
    if not previous:
        payload = {"discovery_job_id": job_id, "paper_ids": PAPERS}
        job = request(client, "POST", "/api/v1/ingestions", json=payload)
        same = request(client, "POST", "/api/v1/ingestions", json=payload)
        assert same["job_id"] == job["job_id"]
    deadline = time.monotonic() + 1800
    last_stage = None
    while time.monotonic() < deadline:
        job = request(client, "GET", f"/api/v1/jobs/{job['job_id']}")
        if job["stage"] != last_stage:
            print(json.dumps({"ingestion_status": job["status"], "stage": job["stage"]}), flush=True)
            last_stage = job["stage"]
        if job["status"] in {"completed", "failed", "partial_success", "cancelled", "interrupted"}:
            break
        time.sleep(3)
    report["ingestion"] = {"fixtures": previous["fixtures"] if previous else metadata,
                           "job": job, "duplicate_request_same_job": True}
    assert job["status"] == "completed", job
    papers = request(client, "GET", "/api/v1/papers")
    assert all(any(p["paper_id"] == pid and p["embedding_status"] == "COMPLETED" for p in papers)
               for pid in PAPERS)


def validate_answer(client, result, ids, *, require_citations=True):
    answer = result["answer"]
    assert answer["answer"].strip()
    citations = answer["citations"]
    if require_citations:
        assert citations, "No citations for an answerable fixture"
    assert all(c["paper_id"] in ids and c["page"] >= 1 and c["excerpt"].strip() for c in citations)
    assert set(answer["sources"]) == {c["chunk_id"] for c in citations}
    evidence = request(client, "GET", f"/api/v1/answers/{result['answer_id']}/evidence")
    assert evidence == citations
    exported = client.get(f"/api/v1/answers/{result['answer_id']}/export")
    exported.raise_for_status()
    assert answer["answer"] in exported.text


def stream_turn(client, cid, question):
    start = time.monotonic()
    events = []
    event_name, data_lines = "", []
    with client.stream("POST", f"/api/v1/conversations/{cid}/messages/stream",
                       json={"question": question}) as response:
        response.raise_for_status()
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line and data_lines:
                events.append({"event": event_name, "data": json.loads("\n".join(data_lines)),
                               "seconds": round(time.monotonic() - start, 4)})
                event_name, data_lines = "", []
    assert not any(e["event"] == "error" for e in events), events
    done = [e for e in events if e["event"] == "done"]
    delta = [e for e in events if e["event"] == "delta"]
    assert len(done) == 1 and len(delta) > 1, "Expected incremental deltas followed by done"
    assert delta[0]["seconds"] < done[0]["seconds"], "No delta arrived before completion"
    assert "".join(e["data"]["text"] for e in delta) == done[0]["data"]["answer"]["answer"]
    return {"result": done[0]["data"], "events": events, "question": question}


def model(client, report, path):
    assert report.get("ingest_status") == "passed", "Finish ingestion before calling the provider"
    if report.get("model_started"):
        raise ValueError("This report already started model calls; use a new output only after reviewing it")
    report.update(model_started=True, model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                  planned_logical_provider_operations=6, cases=[])
    save(path, report)

    def record(name, result, check):
        case = {"name": name, **result}
        report["cases"].append(case)
        save(path, report)
        check()
        case["passed"] = True
        save(path, report)
        print(json.dumps({"case": name, "passed": True}), flush=True)

    single = request(client, "POST", "/api/v1/questions", json={
        "paper_id": PAPERS[0],
        "question": "在表 2 中，Transformer (big) 的 EN-DE 和 EN-FR BLEU 分数分别是多少？请引用该表。",
    })
    def check_single():
        validate_answer(client, single, PAPERS[:1])
        assert all(value in single["answer"]["answer"] for value in ["28.4", "41.8"])
    record("single_table", {"result": single}, check_single)

    conv = request(client, "POST", "/api/v1/conversations",
                   json={"paper_ids": PAPERS[:1], "title": "真实模型回归 · Transformer 多轮"})
    report["conversation_id"] = conv["conversation_id"]
    save(path, report)
    for name, question in [
        ("stream_first", "原始 Transformer 的编码器有多少层？请根据论文回答。"),
        ("stream_followup", "那它的解码器呢？"),
    ]:
        streamed = stream_turn(client, conv["conversation_id"], question)
        def check_stream():
            result = streamed["result"]
            validate_answer(client, result, PAPERS[:1])
            assert re.search(r"\b6\b|六|six", result["answer"]["answer"], re.I)
            expected_history = 1 if name == "stream_followup" else 0
            assert result["context"]["history_turns"] == expected_history
            assert result["context"]["prompt_chars"] <= int(os.getenv("CHAT_PROMPT_MAX_CHARS", "24000"))
        record(name, streamed, check_stream)

    multi = request(client, "POST", "/api/v1/questions/multi", json={
        "paper_ids": PAPERS,
        "question": "这两篇论文各自研究什么任务？对比机器翻译与机器人灵巧操作，分别引用每篇论文的证据。",
    })
    def check_multi():
        validate_answer(client, multi, PAPERS)
        assert {c["paper_id"] for c in multi["answer"]["citations"]} == set(PAPERS)
        assert "翻译" in multi["answer"]["answer"]
        assert any(word in multi["answer"]["answer"] for word in ["操作", "抓取"])
    record("multi_paper", {"result": multi}, check_multi)

    abstain = request(client, "POST", "/api/v1/questions", json={
        "paper_id": PAPERS[0], "question": "论文第一作者每天早上喝什么品牌的咖啡？只根据论文证据回答。",
    })
    def check_abstain():
        validate_answer(client, abstain, PAPERS[:1], require_citations=False)
        assert any(word in abstain["answer"]["answer"] for word in ["无法", "未提供", "未提及", "没有提供", "不能确定"])
        assert abstain["answer"]["confidence"] <= 0.5
    record("insufficient_evidence", {"result": abstain}, check_abstain)
    snapshot = request(client, "GET", f"/api/v1/conversations/{conv['conversation_id']}")
    assert [t["status"] for t in snapshot["turns"]] == ["completed", "completed"]
    report["conversation_before_restart"] = snapshot
    report["model_complete"] = True


def verify(client, report):
    assert report.get("model_complete"), "Complete model phase before restart verification"
    snapshot = request(client, "GET", f"/api/v1/conversations/{report['conversation_id']}")
    assert snapshot == report["conversation_before_restart"], "Conversation changed across restart"
    for case in report["cases"]:
        result = case["result"]
        validate_answer(client, result, PAPERS, require_citations=case["name"] != "insufficient_evidence")
    # Reopen the real vector store and embedding model after restart without LLM calls.
    from app.services.questions import QuestionService
    from literature_database.database import DatabaseManager
    database = DatabaseManager(os.environ["DATABASE_URL"])
    try:
        reader = QuestionService(database, Path("/app/data/m2")).reader_for(PAPERS[:1])
        chunks = reader.retriever.retrieve(PAPERS[0], "encoder decoder N = 6 layers", top_k=5)
        assert chunks and all(c.paper_id == PAPERS[0] for c in chunks)
        report["restart"] = {"conversation_equal": True, "answers_and_exports_equal": True,
                             "retrieved_chunks": len(chunks), "pages": [c.page for c in chunks]}
    finally:
        database.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["ingest", "model", "verify"], required=True)
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--output", type=Path, default=Path("/app/data/m2/deployment_regression.json"))
    parser.add_argument("--real-model", action="store_true", help="Required opt-in for paid provider calls")
    args = parser.parse_args()
    if args.phase == "model" and not args.real_model:
        parser.error("model phase requires --real-model")
    report = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    report.setdefault("schema_version", "deployment-regression-v1")
    report.setdefault("scope", "2 public PDFs; fixed smoke questions, not a quality benchmark")
    start = time.monotonic()
    try:
        with httpx.Client(base_url=args.api_url, timeout=180) as client:
            request(client, "GET", "/api/v1/health")
            if args.phase == "ingest":
                ingest(client, report)
            elif args.phase == "model":
                model(client, report, args.output)
            else:
                verify(client, report)
        report[args.phase + "_status"] = "passed"
    except Exception as exc:
        report[args.phase + "_status"] = "failed"
        report[args.phase + "_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report[args.phase + "_seconds"] = round(time.monotonic() - start, 2)
        save(args.output, report)


if __name__ == "__main__":
    main()
