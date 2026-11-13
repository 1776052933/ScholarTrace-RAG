"""Conservative evidence-support decisions for an optional answer verifier.

This module validates a verifier's cited quotes. It never treats a retrieved
chunk count or a model's self-reported confidence as proof of factual support.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_REFUSAL = re.compile(
    r"^(?:the\s+(?:paper|provided|supplied|available|retrieved|given|specific|size|number|algorithm|"
    r"hyperparameters|answer|information|question)|it\s+is|there\s+is|"
    r"(?:we|i)\s+cannot|cannot|unable\s+to|not\s+(?:enough|specified))",
    re.I,
)
_MISSING = re.compile(
    r"\b(?:not (?:explicitly )?(?:stated|specified|provided|given|mentioned|available)|"
    r"(?:do|does) not (?:state|specify|provide|mention)|"
    r"cannot (?:determine|be determined)|insufficient (?:information|evidence)|"
    r"no (?:information|evidence|details?))\b",
    re.I,
)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?!\w|\.\d)")


def is_refusal(answer: str) -> bool:
    """Recognize explicit uncertainty, without treating an ordinary 'No' as refusal."""
    text = " ".join(answer.split()).strip()
    return (text == "Unanswerable"
            or text.startswith(("根据提供的论文片段无法确定", "根据当前引用片段无法确定",
                                "根据提供的证据无法确定", "无法根据提供的论文片段确定"))
            or normalize_refusal(text, []))


def normalize_refusal(answer: str, sources: Sequence[str]) -> bool:
    """Recognize an uncited missing-information sentence, never a bare 'No'."""
    text = " ".join(answer.split())
    return not sources and bool(_REFUSAL.search(text) and _MISSING.search(text))


def validate_verdict(verdict: Mapping[str, Any], cited_chunks: Mapping[str, str], *,
                     require_all_sources: bool = False) -> dict[str, Any]:
    """Accept 'supported' only with a verbatim quote from a cited chunk."""
    decision = verdict.get("decision")
    if decision not in {"supported", "unsupported", "uncertain"}:
        return {"supported": False, "reason": "invalid_decision"}
    if decision != "supported":
        return {"supported": False, "reason": decision}
    quotes = verdict.get("quotes")
    if not isinstance(quotes, list) or not quotes:
        return {"supported": False, "reason": "missing_quote"}
    valid = []
    for item in quotes:
        if not isinstance(item, dict):
            return {"supported": False, "reason": "invalid_quote"}
        chunk_id, quote = item.get("chunk_id"), item.get("quote")
        if chunk_id not in cited_chunks or not isinstance(quote, str):
            return {"supported": False, "reason": "unknown_quote_source"}
        clean_quote = " ".join(quote.split())
        clean_source = " ".join(cited_chunks[chunk_id].split())
        if len(clean_quote) < 8 or clean_quote not in clean_source:
            return {"supported": False, "reason": "quote_not_verbatim"}
        valid.append({"chunk_id": chunk_id, "quote": clean_quote})
    if require_all_sources and {item["chunk_id"] for item in valid} != set(cited_chunks):
        return {"supported": False, "reason": "missing_source_quote"}
    return {"supported": True, "reason": "quoted_support", "quotes": valid}


def strict_gate(answer: str, sources: Sequence[str], chunks: Sequence[Mapping[str, str]], *,
                paper_ids: Sequence[str], verdict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed on missing evidence, unverifiable claims or verifier uncertainty."""
    if is_refusal(answer):
        return {"action": "abstain", "reason": "model_abstained"}
    if not answer.strip():
        return {"action": "abstain", "reason": "empty_answer"}
    available = {chunk["chunk_id"]: chunk for chunk in chunks}
    if not sources or any(source not in available for source in sources):
        return {"action": "abstain", "reason": "missing_valid_citation"}
    if len(paper_ids) > 1 and {available[source]["paper_id"] for source in sources} != set(paper_ids):
        return {"action": "abstain", "reason": "missing_paper_citation"}
    if verdict is None:
        return {"action": "verify", "reason": "needs_verifier"}
    checked = validate_verdict(verdict, {source: available[source]["text"] for source in sources},
                               require_all_sources=True)
    if not checked["supported"]:
        return {"action": "abstain", "reason": checked["reason"]}
    quoted_numbers = {match.group() for item in checked["quotes"]
                      for match in _NUMBER.finditer(item["quote"])}
    answer_numbers = {match.group() for match in _NUMBER.finditer(answer)}
    if not answer_numbers.issubset(quoted_numbers):
        return {"action": "abstain", "reason": "unquoted_number"}
    return {"action": "answer", "reason": "verified_support"}


def gate(answer: str, sources: Sequence[str], chunks: Sequence[Mapping[str, str]],
         verdict: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if answer == "Unanswerable":
        return {"action": "abstain", "reason": "model_abstained"}
    if not answer.strip():
        return {"action": "retry", "reason": "empty_answer"}
    if normalize_refusal(answer, sources):
        return {"action": "abstain", "reason": "normalized_refusal"}
    available = {chunk["chunk_id"]: chunk["text"] for chunk in chunks}
    if not sources or any(source not in available for source in sources):
        return {"action": "retry", "reason": "missing_valid_citation"}
    if verdict is None:
        return {"action": "verify", "reason": "needs_verifier"}
    checked = validate_verdict(verdict, {source: available[source] for source in sources})
    return {"action": "answer" if checked["supported"] else "retry",
            "reason": checked["reason"], **({"quotes": checked["quotes"]} if checked["supported"] else {})}
