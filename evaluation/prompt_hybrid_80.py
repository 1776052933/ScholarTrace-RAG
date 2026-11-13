"""80-question paired prompt audit using an existing QASPER dev index.

Sampling uses annotation type only. Each prompt sees identical hybrid+rereanked
chunks, never gold answers. Existing result files are not silently retried.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from evaluation.prompt_baseline_20261104 import BASELINE_PROMPT
from evaluation.prompt_hybrid_refine import PROMPT as CANDIDATE_PROMPT

ROOT = Path('/workspace')
SOURCE = ROOT / 'data/benchmarks/qasper/qasper-dev-v0.3.json'
EVALUATOR = ROOT / 'data/benchmarks/qasper/qasper_evaluator.py'
OLD = ROOT / 'evaluation/runs/qasper_qwen_dev200_20261021/manifest.json'
HOLDOUT = ROOT / 'evaluation/runs/qasper_followup_20261022/holdout_60.json'
OUT = ROOT / 'evaluation/runs/prompt_hybrid_80_dev_20261104'
COLLECTION = 'eval_current_f17d4b078e13aaab6ac9e586'
COUNTS = {'extractive': 30, 'abstractive': 20, 'boolean': 18, 'none': 12}
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


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def evaluator():
    spec = importlib.util.spec_from_file_location('qasper_official_80', EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    source, old, holdout = read(SOURCE), read(OLD), read(HOLDOUT)
    if sha(SOURCE) != holdout['source_sha256']:
        raise ValueError('QASPER source differs from previous index')
    previous = set(old['case_ids']) | {row['question_id'] for row in holdout['cases']}
    old_ids = set(old['case_ids'])
    indexed_papers = {pid for pid, paper in source.items() for q in paper['qas']
                      if q['question_id'] in old_ids}
    if len(indexed_papers) != 151:
        raise ValueError('Expected 151 previously indexed papers')
    gold = evaluator().get_answers_and_evidence(source, False)
    groups = defaultdict(list)
    for pid, paper in source.items():
        if pid not in indexed_papers:
            continue
        for q in paper['qas']:
            qid = q['question_id']
            if qid in previous:
                continue
            types = {item['type'] for item in gold[qid]}
            if len(types) == 1 and next(iter(types)) in COUNTS:
                groups[next(iter(types))].append((pid, q))
    rng = random.Random(SEED)
    cases = []
    for kind, n in COUNTS.items():
        choices = sorted(groups[kind], key=lambda pair: pair[1]['question_id'])
        if len(choices) < n:
            raise ValueError(f'Not enough {kind} questions: {len(choices)} < {n}')
        for pid, q in rng.sample(choices, n):
            cases.append({'id': q['question_id'], 'paper_id': pid, 'question': q['question'],
                          'stratum': kind, 'references': gold[q['question_id']]})
    rng.shuffle(cases)
    manifest = {'protocol': 'hybrid-rerank-frozen-prompt-80-v1', 'seed': SEED, 'counts': COUNTS,
                'dataset_sha256': sha(SOURCE), 'evaluator_sha256': sha(EVALUATOR),
                'old_manifest_sha256': sha(OLD), 'old_60_sha256': sha(HOLDOUT),
                'case_ids': [c['id'] for c in cases], 'paper_count': len({c['paper_id'] for c in cases}),
                'prior_question_overlap': sum(c['id'] in previous for c in cases),
                'collection': COLLECTION, 'model': 'qwen3-4b-local', 'retrieval_mode': 'hybrid',
                'candidate_k': 20, 'reranker': 'BAAI/bge-reranker-v2-m3', 'top_k': 5,
                'baseline_prompt_sha256': hashlib.sha256(BASELINE_PROMPT.encode('utf-8')).hexdigest(),
                'candidate_prompt_sha256': hashlib.sha256(CANDIDATE_PROMPT.encode('utf-8')).hexdigest(),
                'source_code_sha256': sha(Path(__file__)),
                'limitations': 'stratified dev questions on previously indexed papers; not a population estimate or new-paper holdout'}
    if (OUT / 'manifest.json').exists():
        frozen = read(OUT / 'manifest.json')
        # The evaluator was updated to retain its old baseline after promotion.
        # Keep the recorded code hash immutable while validating every input.
        comparable = lambda value: {key: item for key, item in value.items()
                                    if key != 'source_code_sha256'}
        if comparable(frozen) != comparable(manifest) or read(OUT / 'cases.json') != cases:
            raise ValueError('Frozen experiment inputs differ; do not mix runs')
    else:
        write(OUT / 'manifest.json', manifest)
        write(OUT / 'cases.json', cases)
        write(OUT / 'prompts.json', {'baseline': BASELINE_PROMPT, 'candidate': CANDIDATE_PROMPT})
    print(f"Frozen {len(cases)} dev questions on {manifest['paper_count']} indexed papers; no previous question overlap", flush=True)
    return cases


def run(limit=0):
    from evaluation.run_current import code_digest
    from rag_reader.config import RAGSettings
    from rag_reader.evidence_graph import EvidenceGraph
    from rag_reader.llm_client import DeepSeekClient
    from rag_reader.milvus_store import MilvusVectorStore
    from rag_reader.pipeline import RAGReader
    from rag_reader.schema import DocumentChunk
    import rag_reader.answer_graph as answer_graph

    cases = prepare()
    settings = RAGSettings(embedding_model='/workspace/data/m3/bge_m3_model', embedding_device='cpu',
        embedding_batch_size=8, collection_name=COLLECTION, vector_backend='milvus',
        milvus_uri='http://milvus:19530', retrieval_mode='hybrid', retrieval_candidate_k=20,
        retrieval_top_k=5, reranker_enabled=True, reranker_model='/workspace/data/m3/reranker_windows_model',
        evidence_strategy='balanced', strict_support_enabled=False,
        llm_model='qwen3-4b-local', llm_base_url='http://host.docker.internal:18510/v1',
        llm_timeout_seconds=180)
    store = MilvusVectorStore(collection_name=COLLECTION, uri=settings.milvus_uri,
                              pipeline_signature='page-section-1000-200-pymupdf-v1')
    if not store.client.has_collection(COLLECTION) or store.client.get_collection_stats(COLLECTION)['row_count'] != 4962:
        raise ValueError('Expected existing 4962-row QASPER index; refusing to create or modify it')
    description = store.client.describe_collection(COLLECTION)
    if not {'dense', 'sparse', 'text'}.issubset({f['name'] for f in description['fields']}):
        raise ValueError('Existing QASPER index cannot perform hybrid search')
    runtime = {'code_sha256': code_digest(), 'settings': settings.model_dump(exclude={'milvus_token'}),
               'index_description': description['description'], 'index_rows': 4962,
               'external_tracing': False}
    runtime_path = OUT / 'runtime.json'
    if runtime_path.exists() and read(runtime_path) != runtime:
        raise ValueError('Runtime changed; use a new output directory')
    if not runtime_path.exists():
        write(runtime_path, runtime)
    client = DeepSeekClient(model=settings.llm_model, base_url=settings.llm_base_url, timeout=180)
    reader = RAGReader(settings=settings, vector_store=store, llm_client=client)
    for index, case in enumerate(cases[:limit or None]):
        evidence_file = OUT / 'retrieval' / (case['id'] + '.json')
        if not evidence_file.exists():
            started = time.perf_counter()
            chunks, info = EvidenceGraph(reader.retriever, top_k=5, strategy='balanced').run(
                [case['paper_id']], case['question'])
            write(evidence_file, {'id': case['id'], 'question': case['question'],
                                 'paper_id': case['paper_id'],
                                 'chunks': [c.model_dump(mode='json') for c in chunks],
                                 'evidence': info, 'seconds': round(time.perf_counter()-started, 3)})
        evidence = read(evidence_file)
        chunks = [DocumentChunk.model_validate(c) for c in evidence['chunks']]
        if not chunks:
            raise RuntimeError(f"No evidence returned for indexed paper {case['paper_id']}")
        order = ('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')
        for variant in order:
            output = OUT / variant / (case['id'] + '.json')
            if output.exists():
                continue
            write(output, {'id': case['id'], 'status': 'started'})
            started = time.perf_counter()
            try:
                graph = reader.qa.workflow
                state = graph._input([case['paper_id']], case['question'])
                state.update(chunks=chunks, evidence=evidence['evidence'],
                             retrieval_question=case['question'])
                original = answer_graph.RAG_QA_PROMPT
                answer_graph.RAG_QA_PROMPT = BASELINE_PROMPT if variant == 'baseline' else CANDIDATE_PROMPT
                try:
                    state.update(graph._prepare(state, {'configurable': {}}))
                finally:
                    answer_graph.RAG_QA_PROMPT = original
                used_hash = digest([c.model_dump(mode='json') for c in state['chunks']])
                state.update(graph._generate(state, {'configurable': {}}))
                state.update(graph._validate(state, {'configurable': {}}))
                result = {'id': case['id'], 'variant': variant, 'status': 'completed',
                          'answer': state['answer'], 'raw_answer': state['raw_answer'],
                          'used_chunks_sha256': used_hash, 'usage': client.get_token_usage()['last'],
                          'seconds': round(time.perf_counter()-started, 3)}
            except Exception as exc:
                result = {'id': case['id'], 'variant': variant, 'status': 'failed',
                          'error': f'{type(exc).__name__}: {exc}',
                          'seconds': round(time.perf_counter()-started, 3)}
            write(output, result)
            if result['status'] != 'completed':
                print(f"{index+1}/{len(cases)} {case['id']} {variant}: FAILED {result['error']}", flush=True)
        print(f"{index+1}/{len(cases)} {case['stratum']} completed", flush=True)


def summary():
    cases = prepare()
    pairs = []
    for case in cases:
        variants = {}
        for variant in ('baseline', 'candidate'):
            path = OUT / variant / (case['id'] + '.json')
            variants[variant] = read(path) if path.exists() else {'status': 'pending'}
        same = (all(x['status'] == 'completed' for x in variants.values()) and
                variants['baseline']['used_chunks_sha256'] == variants['candidate']['used_chunks_sha256'])
        pairs.append({'case': case, 'same_evidence': same, **variants})
    write(OUT / 'paired_review.json', pairs)
    report = {'questions': len(cases), 'complete_same_evidence': sum(p['same_evidence'] for p in pairs),
              'incomplete': [p['case']['id'] for p in pairs if not p['same_evidence']],
              'semantic_scoring': 'requires review against retrieved evidence and reference'}
    write(OUT / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run', 'summary'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be non-negative')
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'run':
        run(args.limit)
    else:
        summary()
