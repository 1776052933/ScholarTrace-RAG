"""Local paired diagnostic: frozen retrieved passages versus annotated evidence.

This does not change retrieval, prompts or scoring in the application. Empty
annotation evidence is a no-evidence model control, not an answerable oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / 'evaluation/runs/provenance_20261107'
OUTPUT = ROOT / 'evaluation/runs/answer_bottleneck_20261108'
DATA = ROOT / 'data/benchmarks/qasper/qasper-dev-v0.3.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_module():
    path = DATA.parent / 'qasper_evaluator.py'
    spec = importlib.util.spec_from_file_location('qasper_official', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_units(paper):
    yield 'Abstract', paper.get('abstract', ''), 'text'
    for section in paper.get('full_text', []):
        for text in section.get('paragraphs', []):
            yield section.get('section_name', ''), text, 'text'
    for item in paper.get('figures_and_tables', []):
        yield 'Figure/Table caption', item.get('caption', ''), 'caption'


def evidence_chunks(paper_id, paper, references):
    """Verify annotated excerpts against source, without copying reference answers."""
    normalize = lambda text: ' '.join(text.split())
    chunks, seen = [], set()
    for reference in references:
        if reference['type'] == 'none':
            continue
        for annotation in reference['evidence']:
            text = annotation.removeprefix('FLOAT SELECTED: ').strip()
            if not text or text in seen:
                continue
            matches = [(section, original, kind) for section, original, kind in source_units(paper)
                       if normalize(text) in normalize(original)]
            if not matches:
                raise ValueError(f'Annotated evidence not located in source: {paper_id}: {text[:80]}')
            section, _, kind = matches[0]
            seen.add(text)
            chunks.append({'chunk_id': f'{paper_id}-diagnostic-{len(chunks):05d}',
                'paper_id': paper_id, 'text': text, 'page': None, 'section': section,
                'metadata': {'title': paper['title'], 'item_type': kind, 'section': section}})
    return chunks


def prepare(out=OUTPUT):
    manifest = read(SOURCE_RUN / 'manifest.json')
    data = read(DATA)
    cases_by_id = {}
    for paper_id, paper in data.items():
        for qa in paper['qas']:
            cases_by_id[qa['question_id']] = (paper_id, paper, qa)
    gold = official_module().get_answers_and_evidence(data, False)
    cases, fingerprints = [], {}
    for qid in manifest['case_ids']:
        record_path = SOURCE_RUN / 'answer' / f'{qid}.json'
        stored = read(record_path)
        assert stored['status'] == 'completed'
        paper_id, paper, qa = cases_by_id[qid]
        references = gold[qid]
        all_none = all(ref['type'] == 'none' for ref in references)
        annotated = evidence_chunks(paper_id, paper, references)
        assert all_none == (not annotated)
        cases.append({'id': qid, 'paper_id': paper_id, 'question': qa['question'],
            'reference_types': sorted({ref['type'] for ref in references}),
            'all_unanswerable': all_none, 'references': references,
            'prior_answer': stored['answer'], 'retrieved': stored['retrieved'],
            'annotated': annotated,
            'control_kind': 'empty_evidence' if all_none else 'annotated_evidence'})
        fingerprints[qid] = digest(record_path)
    frozen = {'source_run': SOURCE_RUN.name, 'source_sha256': digest(DATA),
        'source_manifest_sha256': digest(SOURCE_RUN / 'manifest.json'),
        'source_predictions_sha256': fingerprints, 'script_sha256': digest(Path(__file__)),
        'case_ids': manifest['case_ids'], 'prompt': manifest['prompt'], 'max_tokens': 2048,
        'model': 'qwen3-4b-local', 'temperature': 0,
        'settings': 'same production AnswerGraph prepare/generate/validate; no retrieval rerun',
        'annotation_protocol': 'union of verified source excerpts from all answerable annotations; '
            'reference answers and question types excluded from model messages; '
            'empty-evidence controls excluded from answerable comparison'}
    if (out / 'manifest.json').exists():
        if read(out / 'manifest.json') != frozen or read(out / 'cases.json') != cases:
            raise ValueError('Diagnostic inputs changed; use a new output directory')
    else:
        save(out / 'manifest.json', frozen)
        save(out / 'cases.json', cases)
    print(json.dumps({'cases': len(cases), 'answerable': sum(not c['all_unanswerable'] for c in cases),
        'empty_evidence_controls': sum(c['all_unanswerable'] for c in cases),
        'source_captions': sum(any(x['metadata']['item_type'] == 'caption' for x in c['annotated'])
                               for c in cases)}, ensure_ascii=False), flush=True)
    return cases


def run(out=OUTPUT, base_url='http://host.docker.internal:18510/v1'):
    from rag_reader.answer_graph import AnswerGraph
    from rag_reader.llm_client import DeepSeekClient
    from rag_reader.prompts import RAG_QA_PROMPT
    from rag_reader.schema import DocumentChunk
    cases = prepare(out)
    assert read(out / 'manifest.json')['prompt'] == RAG_QA_PROMPT
    client = DeepSeekClient(model='qwen3-4b-local', base_url=base_url, timeout=180)
    graph = AnswerGraph(None, lambda: client, None)
    for index, case in enumerate(cases, 1):
        # Alternating order reduces a systematic warm-cache ordering advantage.
        variants = ('retrieved', 'annotated') if index % 2 else ('annotated', 'retrieved')
        for variant in variants:
            path = out / variant / f"{case['id']}.json"
            if path.exists():
                if read(path)['status'] != 'completed':
                    raise RuntimeError(f'Incomplete attempt needs inspection: {path}')
                continue
            row = {'id': case['id'], 'variant': variant, 'status': 'started'}
            save(path, row)
            start = time.monotonic()
            state = graph._input([case['paper_id']], case['question'], max_tokens=2048)
            state['retrieval_question'] = case['question']
            state['chunks'] = [DocumentChunk.model_validate(c) for c in case[variant]]
            state['evidence'] = {'sufficient': len(state['chunks']) >= 2}
            try:
                state.update(graph._prepare(state, {}))
                row['messages'] = state['messages']
                row['chunks'] = [c.model_dump(mode='json') for c in state['chunks']]
                # Empty evidence intentionally reaches the model to test refusal;
                # the normal no-evidence graph route would refuse without a call.
                state.update(graph._generate(state, {}))
                row['raw_answer'] = state['raw_answer']
                state.update(graph._validate(state, {}))
                row.update(status='completed', answer=state['answer'],
                           usage=client.get_token_usage()['last'])
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            row['seconds'] = time.monotonic() - start
            save(path, row)
            print(f"{index}/{len(cases)} {variant} {row['status']}", flush=True)
    score(out)


def score(out=OUTPUT):
    official = official_module()
    cases = read(out / 'cases.json')
    summary, per_case = {}, []
    for variant in ('retrieved', 'annotated'):
        predictions, statuses = {}, Counter()
        for case in cases:
            path = out / variant / f"{case['id']}.json"
            row = read(path) if path.exists() else {'status': 'pending'}
            statuses[row['status']] += 1
            if row['status'] == 'completed':
                predictions[case['id']] = row['answer']['answer']
        answerable = [c for c in cases if not c['all_unanswerable']]
        by_type = {}
        for case in answerable:
            value = max(official.token_f1_score(predictions.get(case['id'], ''), ref['answer'])
                        for ref in case['references'])
            kind = case['reference_types'][0]
            by_type.setdefault(kind, []).append(value)
        summary[variant] = {'statuses': dict(statuses), 'answerable_questions': len(answerable),
            'answerable_answer_f1': sum(sum(v) for v in by_type.values()) / len(answerable),
            'answer_f1_by_type': {k: sum(v) / len(v) for k, v in by_type.items()}}
    for case in cases:
        values = {'id': case['id'], 'question': case['question'],
                  'all_unanswerable': case['all_unanswerable'], 'references': case['references']}
        for variant in ('retrieved', 'annotated'):
            path = out / variant / f"{case['id']}.json"
            row = read(path) if path.exists() else {'status': 'pending'}
            text = row.get('answer', {}).get('answer', '')
            values[variant] = {'status': row['status'], 'answer': text,
                'f1': max(official.token_f1_score(text, ref['answer']) for ref in case['references']),
                'sources': row.get('answer', {}).get('sources', [])}
        per_case.append(values)
    save(out / 'score.json', summary)
    save(out / 'per_case.json', per_case)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=['prepare', 'run', 'score'])
    cli.add_argument('--out', type=Path, default=OUTPUT)
    cli.add_argument('--base-url', default='http://host.docker.internal:18510/v1')
    args = cli.parse_args()
    if args.phase == 'run':
        run(args.out, args.base_url)
    else:
        globals()[args.phase](args.out)
