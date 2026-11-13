"""Read-only diagnosis of two saved failures; no model, database, or network calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_reader.splitter import _is_section_heading, _section_blocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/runs/error_diagnosis_20261107'
QID = '1e775cf30784e6b1c2b573294a82e145a3f959bb'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def diagnose():
    frozen = ROOT / 'evaluation/runs/prompt_hybrid_80_dev_20261104'
    current = ROOT / 'evaluation/runs/current_qasper_regression_20261106_2048/results' / f'{QID}.json'
    old = frozen / 'candidate' / f'{QID}.json'
    case_path = frozen / 'cases.json'
    source = ROOT / 'data/benchmarks/qasper/qasper-dev-v0.3.json'
    multi_path = ROOT / 'data/repro_acceptance_20261102/runtime/current_qwen_20261106_v3.json'
    case = next(c for c in read(case_path) if c['id'] == QID)
    row = read(current)
    paper = read(source)[case['paper_id']]
    text = '\n\n'.join([paper['title'], paper.get('abstract', '')] + [
        value for section in paper['full_text']
        for value in [section['section_name'], *section['paragraphs']]])
    blocks = _section_blocks(text, '')
    target = [(section, block) for section, block in blocks
              if 'translate them back to english' in block]
    assert len(target) == 1
    multi = next(c['result']['answer'] for c in read(multi_path)['cases']
                 if c['name'] == 'multi_paper')
    old_chunks = read(frozen / 'retrieval' / f'{QID}.json')['chunks']
    report = {
        'scope': 'saved artifacts and current code only; no new generation or retrieval',
        'language_case': {
            'question': case['question'], 'paper_id': case['paper_id'],
            'reference_answers': [r['answer'] for r in case['references']],
            'reference_sections': [s['section_name'] for s in paper['full_text']
                if any('translate them back to english' in p for p in s['paragraphs'])],
            'old_answer': read(old)['answer'], 'current_answer': row['answer'],
            'retrieved_chunks': row['chunks'],
            'same_ordered_chunk_texts': [(c['chunk_id'], c['text']) for c in row['chunks']]
                == [(c['chunk_id'], c['text']) for c in old_chunks],
            'splitter_reproduction': {
                'assigned_section': target[0][0],
                'headings': {s['section_name']: _is_section_heading(s['section_name'])
                             for s in paper['full_text']},
            },
        },
        'multi_paper_case': {
            'saved_answer': multi,
            'review_citations': [c for c in multi['citations'] if c['paper_id'] == 'repro-dexterous'],
            'input_limit': 'The original complete multi-paper prompt and uncited chunks were not saved in this report. Citations prove these excerpts were supplied, not that they were the only excerpts.',
        },
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [current, old, case_path, source, multi_path,
                      ROOT / 'rag_reader/splitter.py', ROOT / 'rag_reader/answer_contracts.py',
                      ROOT / 'rag_reader/evidence_graph.py', ROOT / 'rag_reader/docling_parser.py']},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'same_ordered_chunk_texts': report['language_case']['same_ordered_chunk_texts'],
        'actual_source_sections': report['language_case']['reference_sections'],
        'splitter_assigned_section': target[0][0],
        'review_citation_count': len(report['multi_paper_case']['review_citations']),
        'report': str(OUT / 'diagnosis.json')}, ensure_ascii=False))


if __name__ == '__main__':
    diagnose()
