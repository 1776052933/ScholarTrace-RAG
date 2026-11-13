"""Small live API check beside the fixed gate-only benchmark.

This uses two previously indexed public PDFs. It is an integration smoke test,
not a statistically meaningful answer-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib import request


BASE_URL = "http://127.0.0.1:18020"
OUTPUT = Path("evaluation/runs/current_rag_smoke_20261104")
CASES = [
    {"id": "single_known", "path": "/api/v1/questions",
     "body": {"paper_id": "repro-attention",
              "question": "How many identical layers are in the original Transformer encoder? Answer from this paper."}},
    {"id": "multi_two_facts", "path": "/api/v1/questions/multi",
     "body": {"paper_ids": ["repro-attention", "repro-dexterous"],
              "question": "分别说明原始 Transformer 编码器有多少层，以及价值型强化学习在采样方面的一项优点。请分别引用两篇论文。"}},
    {"id": "multi_insufficient", "path": "/api/v1/questions/multi",
     "body": {"paper_ids": ["repro-attention", "repro-dexterous"],
              "question": "两篇论文是否都在完全相同的数据集上报告了 F1 分数？若证据不足，请明确说明。"}},
]


def post(path: str, body: dict) -> dict:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(BASE_URL + path, data=payload, headers={
        "Content-Type": "application/json",
    })
    with request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for item in CASES:
        path = args.output / (item["id"] + ".json")
        if path.exists():
            continue
        started = time.perf_counter()
        try:
            result = post(item["path"], item["body"])
            record = {**item, "status": "completed", "response": result,
                      "seconds": round(time.perf_counter() - started, 4)}
        except Exception as exc:
            record = {**item, "status": "failed", "error_type": type(exc).__name__,
                      "seconds": round(time.perf_counter() - started, 4)}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        answer = record.get("response", {}).get("answer", {})
        print(f'{item["id"]}: {record["status"]}, {record["seconds"]}s, '
              f'sources={len(answer.get("sources") or [])}, '
              f'text={str(answer.get("answer") or "")[:120]}', flush=True)


if __name__ == "__main__":
    main()
