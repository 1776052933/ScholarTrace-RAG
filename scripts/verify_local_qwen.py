"""Exercise the project's real model client and RAG answer graph against local Qwen."""
from __future__ import annotations

from urllib.parse import urlparse

from app.config import ModelSettings
from rag_reader.llm_client import DeepSeekClient
from rag_reader.qa import PaperQA
from rag_reader.schema import DocumentChunk


PAPER_ID = "local-qwen-check"


class FixtureRetriever:
    """Two fixed passages isolate model integration from embedding and Milvus setup."""

    @staticmethod
    def retrieve(paper_id: str, query: str, *, top_k: int):
        return FixtureRetriever.retrieve_multiple([paper_id], query, top_k=top_k)

    @staticmethod
    def retrieve_multiple(paper_ids, query: str, *, top_k: int):
        if PAPER_ID not in paper_ids:
            return []
        return [
            DocumentChunk(
                chunk_id="local-qwen-check-1", paper_id=PAPER_ID,
                text="Table 4: Transformer (4 layers), semi-supervised WSJ23 F1: 92.7.",
                page=10, section="Results", metadata={"title": "Local model check"},
            ),
            DocumentChunk(
                chunk_id="local-qwen-check-2", paper_id=PAPER_ID,
                text="The evaluation reports F1 scores for WSJ23 in Table 4.",
                page=10, section="Results", metadata={"title": "Local model check"},
            ),
        ][:top_k]


def main() -> None:
    settings = ModelSettings.from_env()
    host = urlparse(settings.base_url).hostname
    if settings.model != "qwen3-4b-local" or host not in {
        "127.0.0.1", "localhost", "host.docker.internal"
    }:
        raise SystemExit("Refusing to send the verification prompt outside local Qwen")
    client = DeepSeekClient(**settings.client_kwargs(), timeout=settings.timeout_seconds)
    qa = PaperQA(FixtureRetriever(), client, top_k=2)
    question = "According to Table 4, what is the semi-supervised WSJ23 F1 for Transformer (4 layers)?"
    answer = qa.ask(PAPER_ID, question)
    if "92.7" not in answer.answer or "local-qwen-check-1" not in answer.sources:
        raise AssertionError(f"Local RAG answer was not correct and cited: {answer.to_dict()}")
    print(f"RAG answer: {answer.answer}; sources={answer.sources}")
    events = list(qa.stream([PAPER_ID], question, max_tokens=512))
    final_answers = [event["data"] for event in events if event["event"] == "answer"]
    deltas = [event for event in events if event["event"] == "delta"]
    if (len(final_answers) != 1 or not deltas or
            "92.7" not in final_answers[0]["answer"] or
            "local-qwen-check-1" not in final_answers[0]["sources"]):
        raise AssertionError("Local streamed RAG answer was not complete and cited")
    print(f"RAG stream: {len(deltas)} deltas; final sources={final_answers[0]['sources']}")
    print(f"Model: {settings.model}; URL: {settings.base_url}")
    print(f"Tokens: {client.get_token_usage()['total']}")


if __name__ == "__main__":
    main()
