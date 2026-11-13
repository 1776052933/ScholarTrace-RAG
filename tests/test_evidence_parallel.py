"""Bounded multi-paper search shares one embedding and preserves paper order."""
from threading import Lock, get_ident
from time import sleep

from rag_reader.evidence_graph import EvidenceGraph
from rag_reader.retriever import PaperRetriever
from rag_reader.schema import DocumentChunk


class Embedding:
    def __init__(self):
        self.calls = 0

    def embed_query(self, question):
        self.calls += 1
        return [1.0, 0.0]

    def fingerprint(self):
        return "test-embedding"


class Store:
    supports_hybrid = True

    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = Lock()
        self.queries = []

    def ensure_embedding_compatible(self, fingerprint, dimension):
        assert fingerprint == "test-embedding" and dimension == 2

    def search(self, vector, *, paper_id, top_k, **kwargs):
        assert vector == [1.0, 0.0]
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            sleep(0.06 if paper_id == "a" else 0.02)
            self.queries.append(paper_id)
            return [{"chunk_id": f"{paper_id}-{index}", "text": f"{paper_id} evidence {index}",
                     "metadata": {"paper_id": paper_id, "page": index}, "score": 0.9}
                    for index in range(1, top_k + 1)]
        finally:
            with self.lock:
                self.active -= 1


def test_balanced_retrieval_overlaps_milvus_queries_and_embeds_once():
    embedding, store = Embedding(), Store()
    retriever = PaperRetriever(embedding_model=embedding, vector_store=store)
    chunks, report = EvidenceGraph(retriever, top_k=5, strategy="balanced").run(["a", "b"], "compare")
    assert store.peak == 2
    assert embedding.calls == 1
    assert [chunk.paper_id for chunk in chunks] == ["a"] * 3 + ["b"] * 3
    assert report["sufficient"] is True


def test_expansion_batches_missing_papers_once_and_keeps_scope():
    class Retriever:
        def __init__(self):
            self.batches = []

        def retrieve_multiple(self, ids, question, *, top_k):
            return [DocumentChunk(chunk_id="a-1", paper_id="a", text="a evidence", page=1)]

        def retrieve_per_paper(self, ids, question, *, top_k, max_workers):
            self.batches.append((ids, max_workers))
            return {paper_id: [DocumentChunk(chunk_id=f"{paper_id}-{index}", paper_id=paper_id,
                                             text=f"{paper_id} evidence {index}", page=index)
                               for index in (1, 2)] for paper_id in ids}

    retriever = Retriever()
    chunks, report = EvidenceGraph(retriever, strategy="coverage", top_k=2).run(
        ["a", "b", "c"], "compare")
    assert retriever.batches == [(["a", "b", "c"], 3)]
    assert [chunk.paper_id for chunk in chunks] == ["a", "a", "b", "b", "c", "c"]
    assert report["sufficient"] is True


def test_reranking_stays_on_caller_thread_after_parallel_search():
    caller = get_ident()

    class Reranker:
        def __init__(self):
            self.calls = []

        def rerank(self, question, results, *, top_k):
            self.calls.append((get_ident(), results[0]["metadata"]["paper_id"]))
            return results[:top_k]

    reranker = Reranker()
    retriever = PaperRetriever(embedding_model=Embedding(), vector_store=Store(), reranker=reranker)
    output = retriever.retrieve_per_paper(["a", "b"], "compare", top_k=2)
    assert list(output) == ["a", "b"]
    assert reranker.calls == [(caller, "a"), (caller, "b")]
