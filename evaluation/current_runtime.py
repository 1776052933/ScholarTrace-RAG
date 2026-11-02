"""Evaluation adapter over the same RAGReader and graphs used by QuestionService."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from app.config import ModelSettings
from app.papers.schema import PaperMetadata, PaperResource
from rag_reader.config import RAGSettings
from rag_reader.embedding import embedding_fingerprint
from rag_reader.evidence_graph import EvidenceGraph
from rag_reader.llm_client import DeepSeekClient
from rag_reader.pipeline import RAGReader
from rag_reader.schema import DocumentChunk
from rag_reader.store_factory import create_vector_store

from .current_datasets import Case, Source


class GenerationDisabled:
    def chat(self, *args, **kwargs):
        raise RuntimeError("Generation is disabled for this evaluation phase")

    stream_chat = chat


class GraphCapture(BaseCallbackHandler):
    """Observe the actual graph output, including evidence after prompt trimming."""
    def __init__(self):
        self.chunks = []
        self.evidence = {}
        self.nodes = []
        self.model_calls = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        if kwargs.get("name"):
            self.nodes.append(kwargs["name"])

    def on_chat_model_start(self, *args, **kwargs):
        self.model_calls += 1

    def on_chain_end(self, outputs, **kwargs):
        if isinstance(outputs, Mapping) and "chunks" in outputs and "answer" in outputs:
            self.chunks = outputs["chunks"]
            self.evidence = outputs.get("evidence", {})


class CurrentRuntime:
    def __init__(self, settings: RAGSettings, run_dir: Path, *, generation=False):
        if not settings.collection_name.startswith("eval_current_"):
            raise ValueError("Evaluation must use its isolated eval_current_ collection")
        if settings.vector_backend == "chroma" and settings.retrieval_mode != "dense":
            raise ValueError("Chroma cannot execute BM25/hybrid; choose Milvus")
        self.settings = settings
        self.run_dir = run_dir
        self.generation = generation
        store = create_vector_store(settings, chroma_directory=str(run_dir / "chroma"))
        # Construct the provider only for the explicitly enabled answer phase.
        self.client = GenerationDisabled()
        if generation:
            provider = ModelSettings.from_env().model_copy(update={
                "model": settings.llm_model, "base_url": settings.llm_base_url,
                "timeout_seconds": settings.llm_timeout_seconds,
            })
            self.client = DeepSeekClient(**provider.client_kwargs(), timeout=provider.timeout_seconds)
        self.reader = RAGReader(settings=settings, vector_store=store, llm_client=self.client)

    def ensure_new_index(self):
        store = self.reader.vector_store
        exists = (store.client.has_collection(store.collection_name)
                  if self.settings.vector_backend == "milvus" else store.collection.count() > 0)
        if exists:
            raise ValueError("Unowned evaluation index already exists; use a new run directory")

    def verify_ids(self, ids: list[str]) -> bool:
        store = self.reader.vector_store
        for start in range(0, len(ids), 128):
            batch = ids[start:start + 128]
            if self.settings.vector_backend == "milvus":
                rows = store.client.get(collection_name=store.collection_name,
                                        ids=batch, output_fields=["chunk_id"])
                found = {row["chunk_id"] for row in rows}
            else:
                found = set(store.collection.get(ids=batch, include=[])["ids"])
            if found != set(batch):
                return False
        return bool(ids)

    def index_sources(self, sources: list[Source]) -> dict[str, list[str]]:
        if not sources:
            raise ValueError("Cannot index an empty source batch")
        if all(s.representation == "title_abstract" for s in sources):
            chunks = [DocumentChunk(chunk_id=f"{s.paper_id}-abstract", paper_id=s.paper_id,
                                    text=s.text, metadata={"title": s.title}) for s in sources]
            vectors = self.reader.embedding_model.embed_chunks(chunks)
            store = self.reader.vector_store
            store.ensure_embedding_compatible(embedding_fingerprint(self.reader.embedding_model),
                                               len(vectors[0]))
            if store.add_documents(chunks, vectors) != len(chunks):
                raise RuntimeError("Incomplete abstract index write")
            result = {c.paper_id: [c.chunk_id] for c in chunks}
        else:
            result = {}
            for source in sources:
                key = hashlib.sha256(source.paper_id.encode()).hexdigest()
                path = self.run_dir / "documents" / f"{key}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source.text, encoding="utf-8")
                chunks = self.reader.ingest_paper(PaperResource(
                    paper=PaperMetadata(paper_id=source.paper_id, title=source.title),
                    text_path=str(path),
                ))
                result[source.paper_id] = [c.chunk_id for c in chunks]
                # These are only per-process source caches, not persistent indices.
                self.reader._documents.pop(source.paper_id, None)
                self.reader._chunks.pop(source.paper_id, None)
        if not self.verify_ids([key for ids in result.values() for key in ids]):
            raise RuntimeError("Indexed chunks cannot be read back")
        return result

    def run_case(self, case: Case, dataset: str, *, answer=False) -> dict:
        if answer and not self.generation:
            raise RuntimeError("Generation is disabled")
        capture = GraphCapture()
        config = {"callbacks": [capture]}
        if dataset == "litsearch":
            if answer:
                raise ValueError("LitSearch is a retrieval task")
            chunks = self.reader.retriever.retrieve_all(
                case.question, top_k=self.settings.retrieval_top_k)
            return {"retrieved": [c.model_dump(mode="json") for c in chunks],
                    "selected_papers": list(dict.fromkeys(c.paper_id for c in chunks))}
        ids = list(case.paper_ids)
        candidates = []
        if not ids:
            candidates = self.reader.retriever.retrieve_all(
                case.question, top_k=max(20, self.settings.retrieval_candidate_k))
            ids = list(dict.fromkeys(c.paper_id for c in candidates))[:5]
        if not ids:
            return {"status": "no_evidence", "selected_papers": [], "retrieved": [],
                    "model_calls": 0}
        if len(ids) > 5:
            raise ValueError("Current application supports at most five papers per answer")
        if answer:
            qa = self.reader.qa
            response = (qa.ask(ids[0], case.question, config=config) if len(ids) == 1
                        else qa.ask_many(ids, case.question, config=config))
            chunks, evidence = capture.chunks, capture.evidence
            result = {"answer": response.model_dump(mode="json"),
                      "usage": (self.client.get_token_usage().get("last", {})
                                if capture.model_calls and hasattr(self.client, "get_token_usage") else {}),
                      "model_calls": capture.model_calls}
        else:
            chunks, evidence = EvidenceGraph(
                self.reader.retriever, top_k=self.settings.retrieval_top_k
            ).run(ids, case.question, config=config)
            result = {"model_calls": 0}
        return {**result, "selected_papers": ids,
                "candidate_chunks": [c.model_dump(mode="json") for c in candidates],
                "retrieved": [c.model_dump(mode="json") for c in chunks],
                "evidence": evidence, "graph_nodes": capture.nodes}
