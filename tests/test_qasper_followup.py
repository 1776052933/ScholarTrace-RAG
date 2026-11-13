import json

import pytest

from evaluation.qasper_followup import parse_model_json


def test_short_answer_contract_preserves_model_decision_and_validates_sources():
    yes = json.dumps({"short_answer": "Yes", "explanation": "The paper reports only English results.",
                      "sources": ["paper-chunk-1"]})
    assert parse_model_json(yes, {"paper-chunk-1"}, "Does it use only English?")["short_answer"] == "Yes"
    with pytest.raises(ValueError, match="Binary question"):
        parse_model_json(yes.replace('"Yes"', '"The paper reports only English results"'),
                         {"paper-chunk-1"}, "Does it use only English?")
    with pytest.raises(ValueError, match="Unknown source"):
        parse_model_json(yes, {"other-chunk"}, "Does it use only English?")


def test_content_question_rejects_binary_or_verbose_abstention():
    def response(value):
        return json.dumps({"short_answer": value, "explanation": "Evidence reviewed.", "sources": []})

    with pytest.raises(ValueError, match="Content question"):
        parse_model_json(response("Yes"), set(), "Which datasets were used?")
    with pytest.raises(ValueError, match="exact Unanswerable"):
        parse_model_json(response("The size is not stated in the chunks."), set(),
                         "What is the corpus size?")
    assert parse_model_json(response("Unanswerable"), set(),
                            "What is the corpus size?")["short_answer"] == "Unanswerable"
