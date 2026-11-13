"""Fixed QASPER single/multi-paper diagnostic using an existing Milvus index.

Gold references are used for selection and offline scoring only. The answer
graph receives only paper IDs and questions; no gold text enters its prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import time
from pathlib import Path

ROOT = Path('/workspace')
SOURCE = ROOT / 'data/benchmarks/qasper/qasper-test-v0.3.json'
OLD = ROOT / 'evaluation/runs/m6_qasper_test30_20261018/manifest.json'
OUT = ROOT / 'evaluation/runs/local_diagnostic_20261104'
EVALUATOR = ROOT / 'data/benchmarks/qasper/qasper_evaluator.py'
COLLECTION = 'm6_qasper_test30_20261018'
SEED = 20261104


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official():
    spec = importlib.util.spec_from_file_location('qasper_official_diagnostic', EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    source = read(SOURCE)
    old = read(OLD)
    if sha(SOURCE) != old['dataset_sha256']:
        raise ValueError('QASPER dataset differs from index manifest')
    all_gold = official().get_answers_and_evidence(source, False)
    selected_ids = set(old['question_ids'])
    candidates = [(pid, q) for pid, paper in source.items() for q in paper['qas']
                  if q['question_id'] in selected_ids]
    if len(candidates) != 30 or len({p for p, _ in candidates}) != 30:
        raise ValueError('Expected original 30 questions on 30 papers')
    answerable, unanswerable = [], []
    for pid, q in candidates:
        refs = all_gold[q['question_id']]
        states = {r['type'] == 'none' for r in refs}
        paragraphs = source[pid]['full_text']
        text = ' '.join(p for section in paragraphs for p in section['paragraphs'])
        if states == {False} and any(e in text for ref in refs for e in ref['evidence'] if e):
            answerable.append((pid, q))
        elif states == {True}:
            unanswerable.append((pid, q))
    rng = random.Random(SEED)
    if len(answerable) < 15 or len(unanswerable) != 3:
        raise ValueError(f'Unexpected eligible samples: {len(answerable)}, {len(unanswerable)}')
    chosen = rng.sample(sorted(answerable, key=lambda x: x[1]['question_id']), 15)
    negatives = sorted(unanswerable, key=lambda x: x[1]['question_id'])
    cases = []
    def item(pid, q):
        return {'paper_id': pid, 'question_id': q['question_id'], 'question': q['question'],
                'references': all_gold[q['question_id']]}
    positives = [item(pid, q) for pid, q in chosen]
    for index, part in enumerate(positives, 1):
        cases.append({'id': f'single_{index:02}', 'kind': 'single_answerable',
                      'question': part['question'], 'paper_ids': [part['paper_id']], 'parts': [part]})
    for index, (pid, q) in enumerate(negatives, 1):
        part = item(pid, q)
        cases.append({'id': f'unanswerable_{index:02}', 'kind': 'single_unanswerable',
                      'question': part['question'], 'paper_ids': [pid], 'parts': [part]})
    for index in range(6):
        a, b = positives[index * 2:index * 2 + 2]
        question = (f"For paper {a['paper_id']}: {a['question']} "
                    f"For paper {b['paper_id']}: {b['question']} "
                    'Answer the two questions separately and cite each paper.')
        cases.append({'id': f'multi_{index+1:02}', 'kind': 'constructed_multi',
                      'question': question, 'paper_ids': [a['paper_id'], b['paper_id']],
                      'parts': [a, b]})
    manifest = {'protocol': 'qasper-test30-index-diagnostic-v1', 'seed': SEED,
                'scope': '15 answerable + 3 unanswerable singles + 6 constructed two-paper questions',
                'collection': COLLECTION, 'source_sha256': sha(SOURCE), 'index_manifest_sha256': sha(OLD),
                'evaluator_sha256': sha(EVALUATOR), 'case_ids': [c['id'] for c in cases],
                'model': 'qwen3-4b-local', 'reranker': 'BAAI/bge-reranker-v2-m3',
                'retrieval': 'Milvus dense top20 -> BGE reranker -> top5; balanced multi-paper',
                'strict_support_enabled': False, 'gold_is_model_input': False,
                'limitations': 'previously examined QASPER test subset; structured text; multi questions constructed'}
    if (OUT / 'manifest.json').exists():
        if read(OUT / 'manifest.json') != manifest or read(OUT / 'cases.json') != cases:
            raise ValueError('Frozen cases differ; use a new output directory')
    else:
        write(OUT / 'manifest.json', manifest)
        write(OUT / 'cases.json', cases)
    print('Frozen 24 cases: 15 answerable, 3 unanswerable, 6 constructed multi-paper', flush=True)


def run(limit=0):
    from evaluation.run_current import code_digest
    from rag_reader.config import RAGSettings
    from rag_reader.llm_client import DeepSeekClient
    from rag_reader.milvus_store import MilvusVectorStore
    from rag_reader.pipeline import RAGReader
    settings = RAGSettings(embedding_model='/app/data/m3/bge_m3_model', embedding_device='cpu',
        embedding_batch_size=8, collection_name=COLLECTION, vector_backend='milvus',
        milvus_uri='http://milvus:19530', retrieval_mode='dense', retrieval_candidate_k=20,
        retrieval_top_k=5, reranker_enabled=True,
        reranker_model='/app/data/m3/reranker_hf_cache/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
        evidence_strategy='balanced', strict_support_enabled=False,
        llm_model='qwen3-4b-local', llm_base_url='http://host.docker.internal:18510/v1',
        llm_timeout_seconds=180)
    store = MilvusVectorStore(collection_name=COLLECTION, uri=settings.milvus_uri,
                              pipeline_signature='qasper-structured-1000-200-v1')
    if not store.client.has_collection(COLLECTION) or store.client.get_collection_stats(COLLECTION)['row_count'] != 1062:
        raise ValueError('Expected pre-existing 1062-row collection; refusing to create or change index')
    runtime = {'code_sha256': code_digest(), 'settings': settings.model_dump(exclude={'milvus_token'}),
               'index_description': store.client.describe_collection(COLLECTION)['description'],
               'expected_index_rows': 1062, 'external_tracing': False}
    runtime_path = OUT / 'runtime.json'
    if runtime_path.exists() and read(runtime_path) != runtime:
        raise ValueError('Code or runtime settings changed; do not mix runs')
    if not runtime_path.exists():
        write(runtime_path, runtime)
    client = DeepSeekClient(model=settings.llm_model, base_url=settings.llm_base_url, timeout=180)
    reader = RAGReader(settings=settings, vector_store=store, llm_client=client)
    cases = read(OUT / 'cases.json')
    for case in cases[:limit or None]:
        output = OUT / 'results' / (case['id'] + '.json')
        if output.exists():
            continue
        started = time.perf_counter()
        write(output, {'status': 'started', 'id': case['id']})
        try:
            graph = reader.qa.workflow
            state = graph.graph.invoke(graph._input(case['paper_ids'], case['question']))
            result = {'id': case['id'], 'kind': case['kind'], 'status': 'completed',
                      'answer': state['answer'], 'retrieved': [c.model_dump(mode='json') for c in state['chunks']],
                      'evidence': state['evidence'], 'raw_answer': state.get('raw_answer', ''),
                      'usage': client.get_token_usage().get('last', {}),
                      'seconds': round(time.perf_counter() - started, 3)}
        except Exception as exc:
            result = {'id': case['id'], 'kind': case['kind'], 'status': 'failed',
                      'error': f'{type(exc).__name__}: {exc}',
                      'seconds': round(time.perf_counter() - started, 3)}
        write(output, result)
        print(f"{case['id']}: {result['status']} {result['seconds']}s", flush=True)


def score():
    evaluator = official()
    cases = read(OUT / 'cases.json')
    rows = []
    def words(text):
        return set(re.findall(r'[a-z0-9]+', text.casefold()))
    for case in cases:
        path = OUT / 'results' / (case['id'] + '.json')
        record = read(path) if path.exists() else {'status': 'pending'}
        row = {'id': case['id'], 'kind': case['kind'], 'status': record['status']}
        if record['status'] == 'completed':
            answer = record['answer']['answer']
            row.update(answer=answer, sources=record['answer']['sources'], seconds=record['seconds'])
            if len(case['parts']) == 1:
                refs = case['parts'][0]['references']
                row['answer_token_f1'] = max(evaluator.token_f1_score(answer, r['answer']) for r in refs)
            hits = []
            for part in case['parts']:
                gold = [e for ref in part['references'] for e in ref['evidence'] if e and 'FLOAT SELECTED' not in e]
                chunks = [c['text'] for c in record['retrieved'] if c['paper_id'] == part['paper_id']]
                # Diagnostic M0-style token coverage proxy; not official semantic evidence F1.
                scores = [len(words(e) & words(c)) / len(words(e)) for e in gold for c in chunks if words(e)]
                hits.append(None if not scores else max(scores) >= .8)
            row['per_paper_evidence_hit_proxy'] = hits
            row['correctness_review'] = 'pending_semantic_review'
            row['citation_support_review'] = 'pending_semantic_review'
        rows.append(row)
    write(OUT / 'scored_cases.json', rows)
    summary = {'cases': len(cases), 'completed': sum(r['status'] == 'completed' for r in rows),
               'failed': sum(r['status'] == 'failed' for r in rows),
               'pending': sum(r['status'] not in ('completed', 'failed') for r in rows),
               'note': 'Token F1 and 80% token-coverage proxy are not semantic correctness or citation support.'}
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run', 'score'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be non-negative')
    if args.phase == 'run':
        run(args.limit)
    else:
        {'prepare': prepare, 'score': score}[args.phase]()
