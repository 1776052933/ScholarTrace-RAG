"""Local source-role acceptance, split into parse/retrieve/answer phases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/runs/provenance_pdf_20261107'
QUESTION = '这两篇论文各自研究什么任务？对比机器翻译与机器人灵巧操作，分别引用每篇论文的证据。'
IDS = ['repro-attention', 'repro-dexterous']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def parse():
    from rag_reader.docling_parser import DoclingPDFParser
    from rag_reader.provenance import source_role
    records = DoclingPDFParser().parse(ROOT / 'data/repro_acceptance_20261102/runtime/fixtures/dexterous.pdf')
    entries = [r for r in records if source_role(r['text'], r['metadata']) == 'bibliography']
    gupta = [r for r in entries if 'Learning dexterous manipulation for a soft robotic hand' in r['text']]
    assert gupta and all(r['section'] == 'References' for r in gupta)
    save(OUT / 'parsed.json', {'records': records, 'entries': len(entries), 'gupta': gupta})
    print(json.dumps({'records': len(records), 'entries': len(entries), 'gupta_section': gupta[0]['section']}), flush=True)


def retrieve():
    from rag_reader.config import RAGSettings
    from rag_reader.pipeline import RAGReader
    from rag_reader.provenance import source_role
    from evaluation.current_runtime import GenerationDisabled
    import rag_reader.retriever as module
    # The existing PDF index fingerprints this historical model path.
    settings = RAGSettings(embedding_model='/app/models/bge_m3_model', embedding_device='cpu',
        collection_name='repro_docling_20261102', vector_backend='milvus',
        milvus_uri='http://host.docker.internal:19533', retrieval_mode='hybrid', reranker_enabled=True,
        reranker_model='/workspace/data/m3/reranker_windows_model', evidence_strategy='balanced',
        pdf_parser_backend='docling')
    reader = RAGReader(settings=settings, llm_client=GenerationDisabled())
    original = module.answer_candidates
    for variant in ('unfiltered', 'filtered'):
        module.answer_candidates = (lambda q, rows: rows) if variant == 'unfiltered' else original
        try:
            graph = reader.qa.workflow
            state = graph._input(IDS, QUESTION)
            state['retrieval_question'] = QUESTION
            state.update(graph._retrieve(state, {}))
            if state['chunks']:
                state.update(graph._prepare(state, {}))
            serialized = {**state, 'chunks': [c.model_dump(mode='json') for c in state['chunks']]}
            save(OUT / f'{variant}_input.json', serialized)
            print(json.dumps({'variant': variant, 'chunks': len(state['chunks']),
                'roles': [source_role(c.text, c.metadata) for c in state['chunks']],
                'pages': [c.page for c in state['chunks']]}), flush=True)
        finally:
            module.answer_candidates = original


def answer():
    from rag_reader.answer_graph import AnswerGraph
    from rag_reader.llm_client import DeepSeekClient
    from rag_reader.schema import DocumentChunk
    client = DeepSeekClient(model='qwen3-4b-local', base_url='http://host.docker.internal:18510/v1', timeout=180)
    graph = AnswerGraph(None, lambda: client, None)
    for variant in ('unfiltered', 'filtered'):
        path = OUT / f'{variant}_answer.json'
        if path.exists():
            continue
        state = json.loads((OUT / f'{variant}_input.json').read_text(encoding='utf-8'))
        state['chunks'] = [DocumentChunk.model_validate(c) for c in state['chunks']]
        save(path, {'status': 'started'})
        try:
            if graph._route(state) == 'abstain':
                state.update(graph._abstain(state, {}))
            else:
                state.update(graph._generate(state, {}))
                state.update(graph._validate(state, {}))
            result = {'status': 'completed', 'answer': state['answer'], 'raw': state.get('raw_answer')}
        except Exception as exc:
            result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        save(path, result)
        print(json.dumps({'variant': variant, **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('phase', choices=['parse', 'retrieve', 'answer'])
    globals()[cli.parse_args().phase]()
