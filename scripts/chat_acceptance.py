"""Exercise real MySQL/Milvus, OpenAI-compatible streaming and UI with a loopback model fixture.

Run inside the API container after migrations. No external model is contacted.
The placeholder answers test transport/persistence, not answer quality. Temporary
conversation records are removed after the run; the JSON report remains local.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    # Override before application imports so neither model calls nor traces leave this process.
    os.environ["DEEPSEEK_API_KEY"] = "loopback-acceptance-fixture"
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = payload["messages"]
            value = json.loads(messages[-1]["content"])
            calls.append({"stream": bool(payload.get("stream")),
                          "history_messages": len(messages) - 2,
                          "chunk_count": len(value.get("chunks") or [])})
            self.send_response(200)
            if not payload.get("stream"):
                prior = messages[1]["content"] if len(messages) > 2 else ""
                result = json.dumps({"question": prior + " " + value["question"]}, ensure_ascii=False)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = {"id": "local-rewrite", "object": "chat.completion", "created": 1,
                        "model": "local-fixture", "choices": [{"index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": result}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
                self.wfile.write(json.dumps(body).encode())
                return
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = value["chunks"]
            result = json.dumps({"answer": "【本机协议验收占位回答】原文片段：" + chunks[0]["text"][:100],
                                 "sources": [chunk["chunk_id"] for chunk in chunks[:2]], "confidence": 0.5})
            def send(choices, usage=None):
                body = {"id": "local-stream", "object": "chat.completion.chunk", "created": 1,
                        "model": "local-fixture", "choices": choices, "usage": usage}
                self.wfile.write(("data: " + json.dumps(body) + "\n\n").encode())
                self.wfile.flush()
            for index in range(0, len(result), 12):
                send([{"index": 0, "delta": {"content": result[index:index + 12]}, "finish_reason": None}])
                time.sleep(0.005)
            send([{"index": 0, "delta": {}, "finish_reason": "stop"}])
            send([], {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    os.environ["DEEPSEEK_BASE_URL"] = f"http://127.0.0.1:{provider.server_port}/v1"

    import httpx
    import uvicorn
    from sqlalchemy import delete, select
    from streamlit.testing.v1 import AppTest
    from app.api.main import create_app
    from app.storage.models import ConversationModel, ConversationTurnModel, MultiPaperAnswerModel, QuestionAnswerModel

    app = create_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    os.environ["AUTO_PAPER_API_URL"] = base_url
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    created_ids = set()
    started = time.monotonic()
    report = {"external_model_calls": 0, "model": "loopback protocol fixture; answers are placeholders"}
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert server.started
        with httpx.Client(base_url=base_url, timeout=120) as client:
            papers = client.get("/api/v1/papers").json()
            by_title = {}
            for paper in papers:
                if paper["embedding_status"] == "COMPLETED":
                    by_title.setdefault(paper["title"].strip().casefold(), paper["paper_id"])
            selected = list(by_title.values())[:2]
            assert len(selected) == 2, "acceptance requires two already indexed papers"
            fixture = client.post("/api/v1/conversations", json={
                "paper_ids": selected, "title": "【本机协议验收】临时会话，非真实模型回答",
            })
            assert fixture.status_code == 201, fixture.text
            cid = fixture.json()["conversation_id"]
            created_ids.add(cid)
            ui = AppTest.from_file("/app/Auto_Paper_UI/ui_app.py").run(timeout=90)
            assert not ui.exception
            next(w for w in ui.multiselect if w.label == "论文库").set_value(selected)
            ui.run(timeout=90)
            for question in ("请根据论文片段概述方法。", "它们的主要差异是什么？"):
                next(w for w in ui.text_area if w.label == "论文问题").set_value(question)
                next(b for b in ui.button if b.label == "根据所选论文回答").click()
                ui.run(timeout=120)
                assert not ui.exception
                errors = [str(item.value) for item in ui.error]
                assert not errors, errors
            snapshot = client.get(f"/api/v1/conversations/{cid}").json()
            assert [t["status"] for t in snapshot["turns"]] == ["completed", "completed"], snapshot
            assert snapshot["turns"][1]["context"]["history_turns"] == 1
            aid = snapshot["turns"][-1]["turn_id"]
            evidence = client.get(f"/api/v1/answers/{aid}/evidence").json()
            assert evidence and all(e["page"] for e in evidence)
            assert "本机协议验收" in client.get(f"/api/v1/answers/{aid}/export").text
            restored = AppTest.from_file("/app/Auto_Paper_UI/ui_app.py").run(timeout=90)
            next(w for w in restored.multiselect if w.label == "论文库").set_value(selected)
            restored.run(timeout=90)
            assert not restored.exception
            assert any("它们的主要差异是什么" in str(item.value) for item in restored.markdown)
            report.update({"papers": selected, "completed_turns": 2,
                           "followup_history_turns": snapshot["turns"][1]["context"]["history_turns"],
                           "citation_pages": [e["page"] for e in evidence],
                           "ui_submit_followup_restore_export": "passed", "provider_calls": calls,
                           "elapsed_seconds": round(time.monotonic() - started, 2)})
            assert sum(c["stream"] for c in calls) == 2
            assert sum(not c["stream"] for c in calls) == 1
    finally:
        server.should_exit = True
        thread.join(10)
        provider.shutdown()
        provider.server_close()
        sock.close()
        # Delete only the temporary conversations this run created and their duplicate export records.
        if created_ids:
            with app.state.database.get_session() as session:
                ids = session.scalars(select(ConversationTurnModel.turn_id).where(
                    ConversationTurnModel.conversation_id.in_(created_ids))).all()
                session.execute(delete(QuestionAnswerModel).where(QuestionAnswerModel.answer_id.in_(ids)))
                session.execute(delete(MultiPaperAnswerModel).where(MultiPaperAnswerModel.answer_id.in_(ids)))
                session.execute(delete(ConversationModel).where(ConversationModel.conversation_id.in_(created_ids)))
                session.commit()
        app.state.database.dispose()
    path = Path("/app/data/m2/chat_acceptance_report.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
