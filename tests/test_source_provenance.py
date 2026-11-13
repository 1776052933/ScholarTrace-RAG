import json
from dataclasses import asdict
from types import SimpleNamespace as NS

from evaluation.current_datasets import qasper_source
from evaluation.current_runtime import CurrentRuntime
from rag_reader.answer_contracts import _fit_evidence
from rag_reader.cleaner import TextCleaner
from rag_reader.docling_parser import DoclingPDFParser
from rag_reader.provenance import bibliography_entry, needs_paper_body, source_role
from rag_reader.retriever import PaperRetriever
from rag_reader.schema import DocumentChunk
from rag_reader.splitter import DocumentSplitter, _section_blocks


ENTRY = ('Gupta, A., Eppner, C., Levine, S., and Abbeel, P. (2016). Learning dexterous '
         'manipulation for a soft robotic hand from human demonstrations. In: IROS. p. 3786-3793.')


def test_official_sections_survive_source_to_index(tmp_path):
    source = qasper_source('p', {'title': 'Example', 'abstract': 'An overview.',
        'full_text': [{'section_name': 'Proposed methodology ::: Data augmentation',
                      'paragraphs': ['Translate then translate back to English.']},
                     {'section_name': 'Unusual header', 'paragraphs': ['1 Results are preliminary.']}],
        'qas': [{'answer': 'SECRET_GOLD'}]})
    assert 'SECRET_GOLD' not in json.dumps(asdict(source))
    runtime = CurrentRuntime.__new__(CurrentRuntime)
    runtime.run_dir = tmp_path
    captured = []
    runtime.reader = NS(cleaner=TextCleaner(), splitter=DocumentSplitter(),
                        ingest_prepared=lambda prepared: captured.extend(prepared[1]),
                        _documents={}, _chunks={})
    runtime.verify_ids = lambda ids: bool(ids)
    runtime.index_sources([source])
    assert next(c for c in captured if 'translate back' in c.text).section == source.sections[2].name
    assert next(c for c in captured if 'preliminary' in c.text).section == 'Unusual header'
    assert all(c.page is None for c in captured)
    assert all(c.metadata['provenance_version'] == 2 for c in captured)


def test_plain_text_known_nested_heading_stops_old_section_bleed():
    blocks = _section_blocks('Related Work\nPrevious methods.\n\n'
        'Proposed methodology ::: Data augmentation\nTranslate back to English.', '')
    assert blocks[-1][0] == 'Proposed methodology ::: Data augmentation'


def test_docling_reference_continuation_does_not_relabel_neighboring_prose(tmp_path):
    pdf = tmp_path / 'paper.pdf'
    pdf.write_bytes(b'%PDF-1.4')
    def item(label, text):
        return NS(label=label, text=text, self_ref=text[:10], prov=[NS(page_no=15)])
    document = NS(iterate_items=lambda: [(item('SECTION_HEADER', 'FUNDING'), 1),
        (item('TEXT', ENTRY), 1), (item('TEXT', 'Our study received financial support.'), 1)])
    records = DoclingPDFParser(converter=NS(convert=lambda _: NS(document=document))).parse(pdf)
    assert records[1]['section'] == 'References'
    assert records[1]['metadata']['source_section'] == 'FUNDING'
    assert records[2]['section'] == 'FUNDING'
    assert source_role(records[1]['text'], records[1]['metadata']) == 'bibliography'
    assert not bibliography_entry('Gupta et al. (2016) developed a method. We compare with it.')


def test_prompt_retains_source_identity_within_budget_without_arbitrary_metadata():
    chunk = DocumentChunk(chunk_id='c', paper_id='p', text=ENTRY, section='FUNDING', page=15,
        metadata={'title': 'Robotic Hands: A Review', 'private_key': 'DO_NOT_FORWARD'})
    messages, used = _fit_evidence('Answer from sources.', {'question': 'What does this paper study?'},
                                   [], [chunk], 2000)
    payload = json.loads(messages[-1]['content'])['chunks'][0]
    assert payload['title'] == 'Robotic Hands: A Review'
    assert payload['source_type'] == 'bibliography' and payload['section'] == 'References'
    assert 'DO_NOT_FORWARD' not in json.dumps(messages)
    assert len(json.dumps(messages, ensure_ascii=False)) <= 2000
    assert used[0].text == ENTRY


def test_overview_filters_entries_before_reranking_but_citation_query_keeps_them():
    rows = [{'chunk_id': 'ref', 'text': ENTRY, 'score': .99, 'metadata': {'paper_id': 'p'}},
            {'chunk_id': 'head', 'text': '6.1 Machine Translation', 'score': .9, 'metadata': {'paper_id': 'p'}},
            {'chunk_id': 'body', 'text': 'We review reinforcement learning for manipulation.',
             'score': .5, 'metadata': {'paper_id': 'p'}}]
    seen = []
    class Ranker:
        def rerank(self, question, results, *, top_k):
            seen.append([r['chunk_id'] for r in results])
            return results[:top_k]
    retriever = PaperRetriever(embedding_model=NS(embed=lambda _: [1.]),
        vector_store=NS(search=lambda *args, **kwargs: rows), reranker=Ranker())
    overview = '这两篇论文各自研究什么任务？分别引用每篇论文的证据。'
    assert needs_paper_body(overview)
    assert [c.chunk_id for c in retriever.retrieve('p', overview, top_k=1)] == ['body']
    assert seen[-1] == ['body']
    assert [c.chunk_id for c in retriever.retrieve('p', 'Which papers are cited?', top_k=1)] == ['ref']
    assert len(seen[-1]) == 3
    assert not needs_paper_body('What language(s) is/are represented in the dataset?')


def test_reference_only_overview_has_no_usable_evidence():
    retriever = PaperRetriever(embedding_model=NS(embed=lambda _: [1.]),
        vector_store=NS(search=lambda *a, **kw: [
            {'chunk_id': 'r', 'text': ENTRY, 'metadata': {'paper_id': 'p'}}]))
    assert retriever.retrieve('p', 'What is the main contribution of this paper?') == []


def test_old_pdf_reference_variants_are_excluded_from_paper_overview():
    from rag_reader.provenance import answer_candidates
    rows = [
        {'text': 'Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation. 2014.',
         'metadata': {'section': 'References'}},
        {'text': 'van Hoof, H., Hermans, T., Neumann, G., and Peters, J. (2015). Learning robot inhand manipulation with tactile features. In: Humanoids.',
         'metadata': {'section': 'FUNDING'}},
        {'text': 'Citation: 1: Yu C and Wang P (2022) Dexterous Manipulation: A Review.',
         'metadata': {'section': ''}},
        {'text': 'We review reinforcement learning for dexterous manipulation.',
         'metadata': {'section': 'Introduction'}},
    ]
    assert answer_candidates('这篇论文研究什么任务？', rows) == rows[-1:]
    assert answer_candidates('Which papers are cited?', rows) == rows


def test_paper_overview_recovers_body_when_old_index_ranks_only_references():
    calls = []
    class Store:
        def search(self, vector, *, top_k, paper_id):
            calls.append((paper_id, top_k))
            if len(calls) == 1:
                return [{'chunk_id': 'ref', 'text': ENTRY,
                         'metadata': {'paper_id': paper_id, 'section': 'FUNDING'}}]
            return [{'chunk_id': 'body', 'text': 'Our main contribution is a review of robot manipulation.',
                     'metadata': {'paper_id': paper_id, 'section': 'INTRODUCTION'}}]
    retriever = PaperRetriever(embedding_model=NS(embed=lambda _: [1.]), vector_store=Store())
    found = retriever.retrieve('p', 'What is the main contribution of this paper?', top_k=2)
    assert [chunk.chunk_id for chunk in found] == ['body']
    assert len(calls) == 2
