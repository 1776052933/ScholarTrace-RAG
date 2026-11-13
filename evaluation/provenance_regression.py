"""Fixed dev regression using source sections and an isolated candidate index."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from evaluation.current_datasets import qasper_source
from evaluation.current_runtime import CurrentRuntime
from evaluation.current_qasper_regression import selected_cases
from evaluation.run_current import code_digest, public_settings
from evaluation.score_current import score_run
from rag_reader.config import RAGSettings
from rag_reader.prompts import RAG_QA_PROMPT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/runs/provenance_20261107'
DATA = ROOT / 'data/benchmarks'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    os.environ['LANGSMITH_TRACING'] = 'false'
    settings = RAGSettings(embedding_model='/workspace/data/m3/bge_m3_model', embedding_device='cpu',
        embedding_batch_size=8, collection_name='eval_current_provenance_20261107', vector_backend='milvus',
        milvus_uri='http://milvus:19530', retrieval_mode='hybrid', reranker_enabled=True,
        reranker_model='/workspace/data/m3/reranker_windows_model', evidence_strategy='balanced',
        llm_model='qwen3-4b-local', llm_base_url='http://host.docker.internal:18510/v1',
        llm_timeout_seconds=180)
    cases = selected_cases()
    source_path = DATA / 'qasper/qasper-dev-v0.3.json'
    data = read(source_path)
    manifest = {'dataset': 'qasper', 'protocol': {'split': 'dev', 'source': 'structured_sections'},
        'case_ids': [c['id'] for c in cases], 'settings': public_settings(settings),
        'files': [{'path': str(source_path), 'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}],
        'code_sha256': code_digest(), 'prompt': RAG_QA_PROMPT, 'max_tokens': 2048,
        'baseline': 'current_qasper_regression_20261106_2048',
        'scope': 'combined source-structure, prompt metadata and candidate-filter regression; no single-factor attribution'}
    if (OUT / 'manifest.json').exists():
        assert read(OUT / 'manifest.json') == manifest, 'Runtime changed; use a new output directory'
    runtime = CurrentRuntime(settings, OUT, generation=True)
    if not (OUT / 'manifest.json').exists():
        runtime.ensure_new_index()
        save(OUT / 'manifest.json', manifest)
        save(OUT / 'references.json', {c['id']: c['references'] for c in cases})
    indexed = read(OUT / 'index.json') if (OUT / 'index.json').exists() else {}
    for pid in sorted({c['paper_id'] for c in cases}):
        if pid in indexed:
            assert runtime.verify_ids(indexed[pid]), 'Missing indexed vectors'
            continue
        indexed.update(runtime.index_sources([qasper_source(pid, data[pid])]))
        save(OUT / 'index.json', indexed)
        print(f'indexed {len(indexed)} papers', flush=True)
    for i, case in enumerate(cases, 1):
        path = OUT / 'answer' / f"{case['id']}.json"
        if path.exists():
            continue
        row = {'case_id': case['id'], 'question': case['question'], 'status': 'started'}
        save(path, row)
        started = time.monotonic()
        try:
            graph = runtime.reader.qa.workflow
            state = graph.graph.invoke(graph._input([case['paper_id']], case['question'], max_tokens=2048))
            row.update(status='completed', answer=state['answer'], messages=state.get('messages', []),
                       raw_answer=state.get('raw_answer'), evidence=state['evidence'],
                       retrieved=[c.model_dump(mode='json') for c in state['chunks']])
        except Exception as exc:
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        row['seconds'] = time.monotonic() - started
        save(path, row)
        print(f"{i}/30 {row['status']}", flush=True)
    scores = score_run(OUT, data_root=DATA)
    print(json.dumps({k: scores[k] for k in ('completed', 'statuses', 'quality_scores')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
