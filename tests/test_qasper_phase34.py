"""Contracts for paper-disjoint evaluation and bounded evidence selection."""
from __future__ import annotations

import json
import unittest

from evaluation.qasper_paper_holdout import BASELINE, MANIFEST, SOURCE, TRACKED, digest, freeze
from evaluation.qasper_phase3_model_selection import select_indices
from evaluation.qasper_phase3_selection import select


class QasperPhase34Tests(unittest.TestCase):
    def test_tracked_holdout_has_unique_papers_and_questions(self):
        frozen = json.loads(TRACKED.read_text(encoding="utf-8"))
        papers = frozen["paper_ids"]
        cases = frozen["cases"]
        self.assertEqual(len(papers), 30)
        self.assertEqual(set(papers), {case["paper_id"] for case in cases})
        self.assertEqual(len(papers), len(set(papers)))
        self.assertEqual(len(cases), len({case["question_id"] for case in cases}))

    @unittest.skipUnless(SOURCE.is_file() and BASELINE.is_file() and MANIFEST.is_file(),
                         "Optional historical QASPER data and run manifests are unavailable")
    def test_frozen_new_papers_are_disjoint_and_immutable(self):
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(freeze()["manifest_sha256"], digest(MANIFEST))
        self.assertEqual(expected["source_sha256"], digest(SOURCE))
        self.assertEqual(expected["baseline_manifest_sha256"], digest(BASELINE))
        original = json.loads(BASELINE.read_text(encoding="utf-8"))
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        old_ids = set(original["case_ids"])
        old_papers = {pid for pid, paper in data.items()
                      if any(q["question_id"] in old_ids for q in paper["qas"])}
        self.assertTrue(set(expected["paper_ids"]).isdisjoint(old_papers))
        self.assertEqual(len(expected["cases"]),
                         sum(len(data[pid]["qas"]) for pid in expected["paper_ids"]))

    def test_model_selector_rejects_duplicate_and_unbounded_indices(self):
        self.assertEqual(select_indices('{"indices":[7,4,2,1,5]}', 10), [1, 2, 4, 5, 7])
        for raw in ('{"indices":[1,2,3,4,4]}', '{"indices":[1,2,3,4,11]}',
                    '{"indices":[1,2,3,4,true]}'):
            with self.assertRaises(ValueError):
                select_indices(raw, 10)

    def test_heuristic_selectors_keep_five_unique_candidate_ids(self):
        ranked = [{"chunk_id": str(i), "section": str(i % 3),
                   "text": "metric score value" if i == 8 else "other context"}
                  for i in range(10)]
        for variant in ("baseline5", "lexical_tail5", "section_tail5"):
            selected = select("What is the metric score value?", ranked, variant)
            self.assertEqual(len({item["chunk_id"] for item in selected}), 5)
            self.assertTrue(all(item in ranked for item in selected))


if __name__ == "__main__":
    unittest.main()
