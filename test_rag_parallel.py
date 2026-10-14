from threading import Lock
from time import sleep

from main import _run_rag_tasks


class ConcurrentReader:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def _run(self, kind: str, paper_id: str):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            sleep(0.04)
            return {"kind": kind, "paper_id": paper_id}
        finally:
            with self.lock:
                self.active -= 1

    def summarize(self, paper_id: str):
        return self._run("summary", paper_id)

    def extract_knowledge(self, paper_id: str):
        return self._run("knowledge", paper_id)

    def ask(self, paper_id: str, question: str):
        return self._run("answer", paper_id)


def test_rag_tasks_run_concurrently_and_keep_paper_order():
    reader = ConcurrentReader()
    outputs = _run_rag_tasks(
        reader,
        [
            {"paper_id": "paper-1", "success": True},
            {"paper_id": "paper-2", "success": True},
        ],
        summarize=True,
        extract_knowledge=True,
        question=None,
        max_workers=3,
    )

    assert list(outputs) == ["paper-1", "paper-2"]
    assert outputs["paper-1"]["summary"]["kind"] == "summary"
    assert outputs["paper-2"]["knowledge"]["kind"] == "knowledge"
    assert reader.max_active >= 2
