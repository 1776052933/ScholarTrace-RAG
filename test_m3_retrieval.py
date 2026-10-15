from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_reader.cleaner import TextCleaner
from rag_reader.docling_parser import DoclingPDFParser
from rag_reader.errors import IndexCompatibilityError
from rag_reader.milvus_store import MilvusVectorStore
from rag_reader.reranker import CrossEncoderReranker
from rag_reader.retriever import PaperRetriever
from rag_reader.schema import DocumentChunk
from rag_reader.splitter import DocumentSplitter


class FakeMilvus:
    def __init__(self):
        self.description = None
        self.rows = {}
        self.calls = []
        self.requests = None

    def has_collection(self, name):
        return self.description is not None

    def prepare_index_params(self):
        return self

    def add_index(self, **kwargs):
        self.calls.append(("index", kwargs))

    def create_collection(self, *, collection_name, schema, index_params, consistency_level):
        self.description = schema.to_dict()["description"]
        self.schema = schema.to_dict()
        self.calls.append(("create", collection_name, consistency_level))

    def describe_collection(self, name):
        return {"description": self.description}

    def load_collection(self, name):
        self.calls.append(("load", name))

    def upsert(self, *, collection_name, data):
        for row in data:
            self.rows[row["chunk_id"]] = row
        return {"upsert_count": len(data)}

    def get(self, *, collection_name, ids, output_fields):
        return [{"chunk_id": value, "paper_id": self.rows[value]["paper_id"]}
                for value in ids if value in self.rows]

    def query(self, *, collection_name, filter, output_fields, limit):
        return [{"chunk_id": row["chunk_id"]} for row in self.rows.values()
                if row["paper_id"] in filter][:limit]

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [[self._hit(row, 0.9) for row in self.rows.values()
                 if not kwargs.get("filter") or row["paper_id"] in kwargs["filter"]][:kwargs["limit"]]]

    def hybrid_search(self, **kwargs):
        self.requests = kwargs["reqs"]
        self.calls.append(("hybrid", kwargs))
        return [[self._hit(row, 0.04) for row in self.rows.values()][:kwargs["limit"]]]

    def delete(self, *, collection_name, filter):
        if "paper_id ==" in filter:
            paper_id = filter.split("paper_id == ", 1)[1].strip('"')
            ids = [key for key, row in self.rows.items() if row["paper_id"] == paper_id]
        else:
            ids = [key for key in self.rows if key in filter]
        for key in ids:
            del self.rows[key]
        return {"delete_count": len(ids)}

    @staticmethod
    def _hit(row, score):
        return {"id": row["chunk_id"], "distance": score,
                "entity": {key: value for key, value in row.items() if key != "dense"}}


def test_milvus_schema_bm25_hybrid_version_and_readback():
    fake = FakeMilvus()
    store = MilvusVectorStore(client=fake, pipeline_signature="pages-v1")
    store.ensure_embedding_compatible("bge@rev1", 3)
    fields = {item["name"]: item for item in fake.schema["fields"]}
    assert fields["text"]["params"]["enable_analyzer"] is True
    assert fields["dense"]["params"]["dim"] == 3
    assert fake.schema["functions"][0]["output_field_names"] == ["sparse"]
    assert {call[1]["field_name"] for call in fake.calls if call[0] == "index"} == {"dense", "sparse"}
    chunk = DocumentChunk(chunk_id="p-chunk-1", paper_id="p", page=3, section="Methods",
                          text="retrieval augmented generation", metadata={"title": "A"})
    assert store.add_documents([chunk], [[0.1, 0.2, 0.3]]) == 1
    assert store.verify_documents([chunk])
    assert store.has_paper("p")
    result = store.search([0.1, 0.2, 0.3], query_text="retrieval", mode="hybrid", paper_id="p")
    assert result[0]["metadata"]["page"] == 3
    assert result[0]["metadata"]["section"] == "Methods"
    assert result[0]["metadata"]["retrieval_mode"] == "hybrid"
    assert len(fake.requests) == 2
    assert store.search([0.1, 0.2, 0.3], mode="dense")[0]["chunk_id"] == chunk.chunk_id
    lexical = MilvusVectorStore(client=fake, pipeline_signature="pages-v1")
    class NoDenseModel:
        def embed(self, _text):
            raise AssertionError("BM25-only must not run dense inference")
    assert PaperRetriever(
        NoDenseModel(), lexical, retrieval_mode="bm25"
    ).retrieve("p", "retrieval", top_k=1)[0].chunk_id == chunk.chunk_id
    assert store.delete(paper_id="p") == 1
    assert not store.has_paper("p")
    assert not store.verify_documents([chunk])
    other = MilvusVectorStore(client=fake, pipeline_signature="pages-v2")
    with pytest.raises(IndexCompatibilityError):
        other.ensure_embedding_compatible("bge@rev1", 3)
    with pytest.raises(IndexCompatibilityError):
        store.ensure_embedding_compatible("bge@rev2", 3)


def test_milvus_does_not_accept_wrong_dimension_or_missing_query_text():
    store = MilvusVectorStore(client=FakeMilvus())
    store.ensure_embedding_compatible("bge", 2)
    with pytest.raises(IndexCompatibilityError):
        store.add_documents([DocumentChunk(chunk_id="x", paper_id="p", text="sample")], [[1.0]])
    with pytest.raises(ValueError):
        store.search([1.0, 2.0], mode="hybrid")
    with pytest.raises(IndexCompatibilityError):
        store.search([1.0], mode="dense")


def test_retriever_sends_query_text_and_reranks_candidates():
    class Embedding:
        model_name = "fake"
        def embed(self, _text):
            return [1.0]
        def fingerprint(self):
            return "fake@v1"

    class Store:
        supports_hybrid = True
        def ensure_embedding_compatible(self, fingerprint, dimension):
            assert dimension == 1
        def search(self, vector, **kwargs):
            assert kwargs["query_text"] == "What is RAG?"
            assert kwargs["mode"] == "hybrid"
            assert kwargs["top_k"] == 3
            return [
                {"chunk_id": "a", "text": "unrelated", "score": 0.1,
                 "metadata": {"paper_id": "p", "page": 1}},
                {"chunk_id": "b", "text": "RAG uses retrieved evidence", "score": 0.05,
                 "metadata": {"paper_id": "p", "page": 2}},
            ]

    class Model:
        def predict(self, pairs):
            assert len(pairs) == 2
            return [-2.0, 2.0]

    retriever = PaperRetriever(Embedding(), Store(), retrieval_mode="hybrid",
                               candidate_k=3, reranker=CrossEncoderReranker(model=Model()))
    chunks = retriever.retrieve("p", "What is RAG?", top_k=1)
    assert [item.chunk_id for item in chunks] == ["b"]
    assert chunks[0].page == 2
    assert chunks[0].metadata["first_stage_score"] == 0.05


def test_docling_parser_preserves_page_section_and_table(tmp_path: Path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    def item(label, text, page, ref, markdown=None):
        obj = SimpleNamespace(label=label, text=text, self_ref=ref,
                              prov=[SimpleNamespace(page_no=page)])
        if markdown is not None:
            obj.export_to_markdown = lambda _document: markdown
        return obj
    document = SimpleNamespace(iterate_items=lambda: [
        (item("SECTION_HEADER", "2 Methods", 2, "/head"), 1),
        (item("TEXT", "Our method compares models.", 2, "/text"), 1),
        (item("TABLE", "", 3, "/table", "| A | B |\n|---|---|\n| 1 | 2 |"), 1),
    ])
    parser = DoclingPDFParser(converter=SimpleNamespace(
        convert=lambda _path: SimpleNamespace(document=document)))
    records = parser.parse(pdf)
    assert [(r["page"], r["section"]) for r in records] == [
        (2, "2 Methods"), (2, "2 Methods"), (3, "2 Methods"),
    ]
    cleaned = TextCleaner().clean_page_records(records)
    assert cleaned[2]["text"].startswith("| A | B |")
    chunks = DocumentSplitter(chunk_size=100, chunk_overlap=10).split_pages("p", cleaned)
    assert any(chunk.page == 3 and "| A | B |" in chunk.text for chunk in chunks)
    assert all(chunk.section == "2 Methods" for chunk in chunks)



def test_milvus_env_uses_separate_default_collection(monkeypatch):
    from rag_reader.config import RAGSettings
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "milvus")
    monkeypatch.delenv("RAG_COLLECTION_NAME", raising=False)
    assert RAGSettings.from_env().collection_name == "paper_chunks_m3"
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "chroma")
    assert RAGSettings.from_env().collection_name == "paper_chunks_m1"
