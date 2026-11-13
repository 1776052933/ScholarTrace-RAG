"""Conservative, model-free indicators for the frozen 80-question experiment.

Outputs are screening signals, not factual accuracy or citation-support labels.
Incomplete/ambiguous answers remain separate and are never counted as correct.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'evaluation/runs/prompt_hybrid_80_dev_20261104'
DATA = ROOT / 'data/benchmarks/qasper/qasper-dev-v0.3.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.casefold()))


def normalized(text):
    return ' '.join(text.split())


REFUSAL = re.compile(
    r'unanswerable|cannot (?:be )?(?:determine|ascertain|answer)|unable to (?:determine|answer)|'
    r'not (?:explicitly )?(?:stated|specified|provided|mentioned|available|indicated|clear)|'
    r'do(?:es)? not (?:explicitly )?(?:state|specify|mention|provide|indicate)|'
    r'insufficient (?:information|evidence)|not enough (?:information|evidence)|'
    r'no (?:explicit |sufficient )?(?:information|evidence|mention)|'
    r'无法(?:确定|判断|回答|得知)|不能(?:确定|判断|回答)|未(?:明确)?(?:说明|提及|提供)|'
    r'没有(?:明确)?(?:说明|提及|提供)|证据不足|信息不足', re.I)
PREFIX = re.compile(r'^\s*(?:["\'“”*\-]+\s*)?(yes|no)\b', re.I)


def classify(answer):
    text = str(answer or '').strip()
    refusal = bool(REFUSAL.search(text))
    match = PREFIX.search(text)
    decision = match.group(1).lower() if match else None
    if not decision:
        if re.match(r'^\s*(?:不是|否(?:定|[，。,:：\s]|$))', text):
            decision = 'no'
        elif re.match(r'^\s*(?:是的|是(?:[，。,:：\s]|$))', text):
            decision = 'yes'
    if refusal and decision:
        return 'mixed_or_ambiguous'
    if refusal:
        return 'refusal_signal'
    if decision:
        return decision
    return 'other_or_unclassified'


def main():
    cases, source = read(RUN/'cases.json'), read(DATA)
    spec = importlib.util.spec_from_file_location(
        'qasper_evaluator_for_prompt_80', ROOT/'data/benchmarks/qasper/qasper_evaluator.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    rows, groups = [], defaultdict(Counter)
    answer_f1 = defaultdict(list)
    for case in cases:
        qid = case['id']
        result = {'question_id': qid, 'paper_id': case['paper_id'],
                  'stratum': case['stratum'], 'question': case['question']}
        evidence_path = RUN/'retrieval'/f'{qid}.json'
        evidence = read(evidence_path) if evidence_path.exists() else None
        evidence_hit = None
        if evidence and case['stratum'] != 'none':
            paragraphs = [normalized(p) for s in source[case['paper_id']]['full_text'] for p in s['paragraphs']]
            gold = [e for ref in case['references'] for e in ref['evidence']
                    if e and 'FLOAT SELECTED' not in e and normalized(e) in paragraphs]
            scores = [len(words(e) & words(c['text'])) / len(words(e))
                      for e in gold for c in evidence['chunks'] if words(e)]
            evidence_hit = None if not scores else max(scores) >= 0.8
        result['evidence_hit_proxy'] = evidence_hit
        refs = {r['answer'].strip().casefold() for r in case['references']}
        gold_boolean = next(iter(refs)) if case['stratum'] == 'boolean' and len(refs) == 1 else None
        result['gold_boolean'] = gold_boolean
        variants = {}
        for variant in ('baseline', 'candidate'):
            path = RUN/variant/f'{qid}.json'
            item = read(path) if path.exists() else {'status': 'pending'}
            status = item['status']
            answer = item.get('answer', {}).get('answer', '')
            state = classify(answer) if status == 'completed' else 'not_completed'
            labels = {'status': status, 'answer': answer, 'classification': state,
                      'has_valid_source_id': bool(item.get('answer', {}).get('sources')),
                      'seconds': item.get('seconds')}
            if gold_boolean in {'yes', 'no'}:
                labels['prefix_agrees_with_gold'] = state == gold_boolean if state in {'yes', 'no'} else None
            groups[(variant, case['stratum'])]['total'] += 1
            groups[(variant, case['stratum'])][status] += 1
            if status == 'completed':
                answer_f1[(variant, case['stratum'])].append(max(
                    evaluator.token_f1_score(answer, reference['answer'])
                    for reference in case['references']))
                groups[(variant, case['stratum'])][state] += 1
                groups[(variant, case['stratum'])]['with_source_id'] += int(labels['has_valid_source_id'])
                if gold_boolean in {'yes', 'no'} and state in {'yes', 'no'}:
                    groups[(variant, case['stratum'])]['clear_boolean'] += 1
                    groups[(variant, case['stratum'])]['boolean_agrees_with_gold'] += int(state == gold_boolean)
            variants[variant] = labels
        result['same_evidence'] = (evidence is not None and
            all(v['status'] == 'completed' for v in variants.values()) and
            read(RUN/'baseline'/f'{qid}.json')['used_chunks_sha256'] ==
            read(RUN/'candidate'/f'{qid}.json')['used_chunks_sha256'])
        result.update(variants)
        rows.append(result)
    report = {'questions': len(cases), 'completed_pairs': sum(r['same_evidence'] for r in rows),
              'groups': {f'{k[0]}:{k[1]}': dict(v) for k, v in groups.items()},
              'raw_answer_f1_by_type': {
                  f'{variant}:{stratum}': sum(scores)/len(scores)
                  for (variant, stratum), scores in answer_f1.items() if scores},
              'raw_answer_f1_all': {
                  variant: sum(sum(scores) for (name, _), scores in answer_f1.items() if name == variant)
                  / sum(len(scores) for (name, _), scores in answer_f1.items() if name == variant)
                  for variant in ('baseline', 'candidate')},
              'evidence_proxy_hits': sum(r['evidence_hit_proxy'] is True for r in rows),
              'evidence_proxy_scorable': sum(r['evidence_hit_proxy'] is not None for r in rows),
              'notes': ['Stratified sample; not population prevalence.',
                        'Refusal and yes/no classification are conservative string signals requiring review.',
                        'Gold agreement, evidence overlap, and citation IDs are not factual correctness or semantic support.']}
    out = RUN/'screening'
    out.mkdir(parents=True, exist_ok=True)
    (out/'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
