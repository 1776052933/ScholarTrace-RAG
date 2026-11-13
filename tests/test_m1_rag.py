"""M1 regressions for page provenance, citations, and model failure."""

import json

import fitz
import pytest

from app.papers.schema import PaperMetadata, PaperResource
from rag_reader.embedding import EmbeddingLoadError, EmbeddingModel
from rag_reader.config import RAGSettings
from rag_reader.pipeline import RAGReader
from rag_reader.splitter import DocumentSplitter
from rag_reader.vector_store import VectorStore


class KeywordEmbedding:
    def embed(self, text):
        return [0.0, 1.0, 0.0] if "kiwi" in text.casefold() else [1.0, 0.0, 0.0]

    def embed_chunks(self, chunks):
        return [self.embed(chunk.text) for chunk in chunks]


class SourceEchoClient:
    def chat(self, messages, **_):
        payload = json.loads(messages[1]["content"])
        if "核验器" in messages[0]["content"]:
            return json.dumps({"decision": "supported", "quotes": [
                {"chunk_id": item["chunk_id"], "quote": item["text"]}
                for item in payload["cited_chunks"]
            ]})
        return json.dumps({
            "answer": "The kiwi evaluation improved.",
            "sources": [payload["chunks"][0]["chunk_id"]],
            "confidence": 0.9,
        })


def test_pdf_page_survives_ingestion_and_answer_even_with_text_sidecar(tmp_path):
    pdf = tmp_path / "paper.pdf"
    document = fitz.open()
    for content in (
        "INTRODUCTION\nThe first page describes apple orchard research.",
        "RESULTS\nThe second page reports kiwi evaluation improvement.",
    ):
        page = document.new_page()
        page.insert_text((72, 72), content)
    document.save(str(pdf))
    document.close()
    sidecar = tmp_path / "paper.txt"
    sidecar.write_text("A flattened sidecar should not replace PDF provenance.", encoding="utf-8")

    reader = RAGReader(
        embedding_model=KeywordEmbedding(),
        vector_store=VectorStore(persist_directory=str(tmp_path / "chroma")),
        splitter=DocumentSplitter(chunk_size=1000, chunk_overlap=50),
        llm_client=SourceEchoClient(),
        settings=RAGSettings(reranker_enabled=False),
    )
    chunks = reader.ingest_paper(PaperResource(
        paper=PaperMetadata(paper_id="paper-1", title="Two-page paper"),
        file_path=str(pdf), text_path=str(sidecar),
    ))
    assert {chunk.page for chunk in chunks} == {1, 2}
    answer = reader.ask("paper-1", "What kiwi result was found?")
    assert answer.sources == [answer.citations[0].chunk_id]
    assert answer.citations[0].paper_id == "paper-1"
    assert answer.citations[0].page == 2
    assert "kiwi" in answer.citations[0].excerpt.casefold()


def test_prepare_paper_does_not_write_index_until_ingest_prepared(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("A reproducible research finding about kiwi.", encoding="utf-8")
    store = VectorStore(persist_directory=str(tmp_path / "index"))
    reader = RAGReader(embedding_model=KeywordEmbedding(), vector_store=store,
                       settings=RAGSettings(reranker_enabled=False))
    resource = PaperResource(paper=PaperMetadata(paper_id="prepared-paper", title="Prepared paper"),
                             text_path=str(source))

    prepared = reader.prepare_paper(resource)
    assert prepared[0].paper_id == "prepared-paper"
    assert prepared[1]
    with pytest.raises(LookupError):
        reader.get_document("prepared-paper")
    assert store.search([1.0, 0.0, 0.0], top_k=1, paper_id="prepared-paper") == []

    chunks = reader.ingest_prepared(prepared)
    assert len(chunks) == len(prepared[1])
    assert reader.get_document("prepared-paper").content == prepared[0].content
    assert store.search([1.0, 0.0, 0.0], top_k=1, paper_id="prepared-paper")


def test_embedding_load_failure_is_explicit(monkeypatch):
    import sentence_transformers

    def fail(*_args, **_kwargs):
        raise OSError("model files missing")

    EmbeddingModel.clear_cache()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)
    with pytest.raises(EmbeddingLoadError, match="failed to load embedding model"):
        EmbeddingModel(model_name="missing-model").embed("test query")
    EmbeddingModel.clear_cache()


def test_config_and_error_contract_without_api_key(monkeypatch):
    from pydantic import ValidationError
    from rag_reader.config import RAGSettings
    from rag_reader.errors import error_info
    from rag_reader.pipeline import PaperNotIngestedError
    from rag_reader.prompt_adapter import build_messages

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = RAGSettings(chunk_size=120, chunk_overlap=20, retrieval_top_k=3)
    reader = RAGReader(settings=settings, embedding_model=KeywordEmbedding(),
                       vector_store=VectorStore())
    assert reader.qa.top_k == 3
    assert reader.splitter.chunk_size == 120
    assert reader.qa.llm_client.get_token_usage()["total"]["total_tokens"] == 0
    with pytest.raises(ValidationError):
        RAGSettings(chunk_size=50, chunk_overlap=50)

    messages = build_messages('Return JSON like {"answer": ""}', {"question": "kiwi"})
    assert messages[0]["content"] == 'Return JSON like {"answer": ""}'
    assert json.loads(messages[1]["content"])["question"] == "kiwi"
    error = PaperNotIngestedError("missing paper")
    assert isinstance(error, LookupError)
    assert error_info(error)["code"] == "not_found"


def test_persisted_index_rejects_other_embedding_and_legacy_vectors(tmp_path):
    from rag_reader.errors import IndexCompatibilityError
    from rag_reader.schema import DocumentChunk

    directory = str(tmp_path / "index")
    store = VectorStore(persist_directory=directory)
    store.ensure_embedding_compatible("model-a@revision-1", 3)
    store.add_documents([
        DocumentChunk(chunk_id="paper-a-chunk-1", paper_id="paper-a", text="kiwi")
    ], [[1.0, 0.0, 0.0]])
    reopened = VectorStore(persist_directory=directory)
    reopened.ensure_embedding_compatible("model-a@revision-1", 3)
    assert reopened.search([1.0, 0.0, 0.0], top_k=1)[0]["distance"] == pytest.approx(0.0)
    with pytest.raises(IndexCompatibilityError):
        reopened.ensure_embedding_compatible("model-b@revision-2", 3)
    with pytest.raises(IndexCompatibilityError):
        reopened.ensure_embedding_compatible("model-a@revision-1", 4)

    legacy = VectorStore(collection_name="legacy_collection", persist_directory=directory)
    legacy.add_documents([
        DocumentChunk(chunk_id="old-1", paper_id="paper-a", text="old")
    ], [[1.0, 0.0, 0.0]])
    with pytest.raises(IndexCompatibilityError, match="unversioned vectors"):
        legacy.ensure_embedding_compatible("model-a@revision-1", 3)
