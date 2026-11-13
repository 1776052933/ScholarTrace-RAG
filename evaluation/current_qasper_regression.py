"""Current-code QASPER regression on a frozen subset of the 80-question audit.

Runs retrieval and local generation anew; writes each result before scoring.
The old candidate answers are read only for a paired, same-question comparison.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import time

from evaluation.score_current import project_paragraphs
from evaluation.prompt_hybrid_80 import COLLECTION
from rag_reader.config import RAGSettings
from rag_reader.llm_client import DeepSeekClient
from rag_reader.milvus_store import MilvusVectorStore
from rag_reader.pipeline import RAGReader


ROOT = Path('/workspace')
FROZEN = ROOT / 'evaluation/runs/prompt_hybrid_80_dev_20261104'
OUTPUT = ROOT / 'evaluation/runs/current_qasper_regression_20261106_2048'
QUOTAS = {'extractive': 10, 'abstractive': 8, 'boolean': 6, 'none': 6}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def selected_cases():
    counts = Counter()
    selected = []
    for case in read(FROZEN / 'cases.json'):
        kind = case['stratum']
        if counts[kind] < QUOTAS[kind]:
            selected.append(case)
            counts[kind] += 1
    if dict(counts) != QUOTAS:
        raise ValueError('Frozen 80-question cohort is incomplete')
    return selected


def run():
    os.environ['LANGSMITH_TRACING'] = 'false'
    cases = selected_cases()
    settings = RAGSettings(
        embedding_model='/workspace/data/m3/bge_m3_model', embedding_device='cpu',
        embedding_batch_size=8, collection_name=COLLECTION, vector_backend='milvus',
        milvus_uri='http://milvus:19530', retrieval_mode='hybrid', retrieval_candidate_k=20,
        retrieval_top_k=5, reranker_enabled=True,
        reranker_model='/workspace/data/m3/reranker_windows_model',
        evidence_strategy='balanced', strict_support_enabled=False,
        llm_model='qwen3-4b-local', llm_base_url='http://host.docker.internal:18510/v1',
        llm_timeout_seconds=180,
    )
    store = MilvusVectorStore(collection_name=COLLECTION, uri=settings.milvus_uri,
                              pipeline_signature='page-section-1000-200-pymupdf-v1')
    if (not store.client.has_collection(COLLECTION)
            or store.client.get_collection_stats(COLLECTION)['row_count'] != 4962):
        raise ValueError('Frozen QASPER index missing or changed')
    reader = RAGReader(settings=settings, vector_store=store,
                       llm_client=DeepSeekClient(model=settings.llm_model,
                                                 base_url=settings.llm_base_url, timeout=180))
    write(OUTPUT / 'manifest.json', {'case_ids': [c['id'] for c in cases],
        'quotas': QUOTAS, 'collection': COLLECTION, 'retrieval_mode': 'hybrid',
        'candidate_k': 20, 'reranker': settings.reranker_model, 'top_k': 5,
        'model': settings.llm_model, 'source': str(FROZEN / 'manifest.json')})
    for index, case in enumerate(cases, 1):
        path = OUTPUT / 'results' / f"{case['id']}.json"
        if path.exists():
            continue
        write(path, {'id': case['id'], 'status': 'started'})
        started = time.perf_counter()
        try:
            graph = reader.qa.workflow
            result = graph.graph.invoke(graph._input([case['paper_id']], case['question'],
                                                          max_tokens=2048))
            chunks = result['chunks']
            output = {'id': case['id'], 'status': 'completed', 'stratum': case['stratum'],
                      'answer': result['answer'], 'chunks': [c.model_dump(mode='json') for c in chunks],
                      'evidence': result['evidence'],
                      'seconds': round(time.perf_counter() - started, 3)}
        except Exception as exc:
            output = {'id': case['id'], 'status': 'failed', 'stratum': case['stratum'],
                      'error': f'{type(exc).__name__}: {exc}',
                      'seconds': round(time.perf_counter() - started, 3)}
        write(path, output)
        print(f"{index}/{len(cases)} {case['stratum']} {output['status']}", flush=True)
    score()


def score():
    cases = selected_cases()
    spec = importlib.util.spec_from_file_location('qasper_official',
                  ROOT / 'data/benchmarks/qasper/qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    data = read(ROOT / 'data/benchmarks/qasper/qasper-dev-v0.3.json')
    gold_all = official.get_answers_and_evidence(data, False)
    gold = {c['id']: gold_all[c['id']] for c in cases}
    paragraphs = {c['id']: [p for section in data[c['paper_id']]['full_text']
                             for p in section.get('paragraphs', [])] for c in cases}
    predictions = {}
    previous = {}
    rows = []
    for case in cases:
        qid = case['id']
        path = OUTPUT / 'results' / f'{qid}.json'
        row = read(path) if path.exists() else {'status': 'pending'}
        prior = read(FROZEN / 'candidate' / f'{qid}.json')
        frozen_retrieval = read(FROZEN / 'retrieval' / f'{qid}.json')
        for source, record, target in [('current', row, predictions), ('prior', prior, previous)]:
            if record['status'] != 'completed':
                continue
            answer = record['answer']
            excerpts = [c['excerpt'] for c in answer.get('citations', [])]
            evidence, _ = project_paragraphs(excerpts, paragraphs[qid])
            target[qid] = {'answer': answer['answer'], 'evidence': evidence}
        chunks = row.get('chunks', [])
        rows.append({'id': qid, 'stratum': case['stratum'], 'status': row['status'],
                     'seconds': row.get('seconds'), 'retrieval_mode': sorted({
                         c['metadata'].get('retrieval_mode') for c in chunks}),
                     'reranked_chunks': sum(c['metadata'].get('rerank_score') is not None
                                            for c in chunks), 'retrieved_chunks': len(chunks),
                     'same_top5_ids': [c['chunk_id'] for c in chunks] == [
                         c['chunk_id'] for c in frozen_retrieval['chunks']],
                     'same_answer_text': row.get('answer', {}).get('answer') ==
                         prior.get('answer', {}).get('answer')})
    report = {'questions': len(cases), 'statuses': dict(Counter(r['status'] for r in rows)),
              'current': official.evaluate(gold, predictions),
              'prior_same_cases': official.evaluate(gold, previous),
              'retrieval_hybrid_cases': sum(r['retrieval_mode'] == ['hybrid'] for r in rows),
              'reranked_cases': sum(r['reranked_chunks'] > 0 for r in rows),
              'same_top5_as_prior': sum(r['same_top5_ids'] for r in rows),
              'same_answer_text_as_prior': sum(r['same_answer_text'] for r in rows),
              'mean_seconds': round(sum(r['seconds'] or 0 for r in rows) / len(rows), 3),
              'note': 'QASPER official lexical scores, not human correctness; QASPER structured text, no PDF/API/UI.'}
    write(OUTPUT / 'score.json', report)
    write(OUTPUT / 'per_case.json', rows)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
