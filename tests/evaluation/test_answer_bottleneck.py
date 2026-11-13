import json
import unittest

from evaluation.answer_bottleneck import evidence_chunks


class EvidenceDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.paper = {'title': 'Sample', 'abstract': 'Paper overview.',
            'full_text': [{'section_name': 'Dataset', 'paragraphs': ['The corpus is English.']}],
            'figures_and_tables': [{'caption': 'Table 1: Dataset sizes.'}]}

    def test_prompt_chunks_include_source_evidence_without_reference_answer(self):
        references = [{'type': 'extractive', 'answer': 'SECRET_ANSWER',
                       'evidence': ['The corpus is English.']}]
        chunks = evidence_chunks('p', self.paper, references)
        self.assertEqual(chunks[0]['section'], 'Dataset')
        self.assertEqual(chunks[0]['text'], 'The corpus is English.')
        self.assertNotIn('SECRET_ANSWER', json.dumps(chunks))
        self.assertIsNone(chunks[0]['page'])

    def test_missing_source_is_rejected_instead_of_forwarding_annotation(self):
        with self.assertRaises(ValueError):
            evidence_chunks('p', self.paper,
                [{'type': 'extractive', 'answer': 'x', 'evidence': ['A fabricated excerpt.']}])

    def test_unanswerable_evidence_stays_empty(self):
        self.assertEqual(evidence_chunks('p', self.paper,
            [{'type': 'none', 'answer': 'Unanswerable', 'evidence': []}]), [])

    def test_captions_are_verified_and_duplicate_annotation_evidence_is_deduplicated(self):
        reference = {'type': 'abstractive', 'answer': 'x',
                     'evidence': ['FLOAT SELECTED: Table 1: Dataset sizes.']}
        chunks = evidence_chunks('p', self.paper, [reference, reference])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['text'], self.paper['figures_and_tables'][0]['caption'])
        self.assertEqual(chunks[0]['metadata']['item_type'], 'caption')


if __name__ == '__main__':
    unittest.main()
