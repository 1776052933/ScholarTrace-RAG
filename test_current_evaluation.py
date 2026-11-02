"""Offline adapter contracts; no downloaded models, database or provider needed."""
import json
from types import SimpleNamespace as NS

import pytest

from evaluation.current_datasets import Case, load_dataset
from evaluation.current_runtime import CurrentRuntime
from evaluation.run_current import execute, parser, public_settings, settings_for
from rag_reader.config import RAGSettings
from rag_reader.qa import PaperQA
from test_langchain_architecture import Retrieval, Transport


@pytest.fixture
def args(tmp_path, monkeypatch):
    for key in ("RAG_VECTOR_BACKEND", "RAG_RETRIEVAL_MODE", "MILVUS_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "data"
    (root / "qasper").mkdir(parents=True)
    paper = {"title": "Source", "abstract": "Source abstract",
             "full_text": [{"section_name": "Method", "paragraphs": ["Six layers."]}],
             "qas": [{"question_id": f"q{i}", "question": "How many layers?",
                      "answers": [{"answer": {"free_form_answer": "SECRET_GOLD"}}]}
                     for i in range(3)]}
    (root / "qasper/qasper-test-v0.3.json").write_text(json.dumps({"a": paper}))
    return parser().parse_args(["qasper", "--data-root", str(root),
                                "--run-dir", str(tmp_path / "run")])


class FakeRuntime:
    def __init__(self):
        self.ids = set()
        self.calls = []
        self.new_checks = 0

    def factory(self, *args, **kwargs):
        return self

    def ensure_new_index(self):
        self.new_checks += 1

    def verify_ids(self, ids):
        return bool(ids) and set(ids) <= self.ids

    def index_sources(self, sources):
        result = {s.paper_id: [f"{s.paper_id}-1"] for s in sources}
        self.ids.update(i for ids in result.values() for i in ids)
        return result

    def run_case(self, case, dataset, *, answer):
        self.calls.append((case.case_id, answer))
        return {"model_calls": int(answer), "retrieved": []}


def test_plan_is_read_only_and_settings_are_isolated(args, monkeypatch):
    monkeypatch.setenv("MILVUS_TOKEN", "secret-token")
    monkeypatch.setenv("RAG_COLLECTION_NAME", "production")
    report = execute(args, runtime_factory=lambda *a, **k: pytest.fail("runtime on plan"))
    assert report["settings"]["vector_backend"] == "milvus"
    assert report["settings"]["collection_name"].startswith("eval_current_")
    assert "secret-token" not in json.dumps(report)
    assert not args.run_dir.exists()


def test_qasper_gold_never_enters_sources_or_questions(args):
    ds = load_dataset("qasper", args.data_root)
    assert "SECRET_GOLD" not in str(list(ds.sources()))
    assert "SECRET_GOLD" not in str(ds.cases)
    assert "SECRET_GOLD" in str(ds.references)


def test_index_resume_and_answer_budget(args):
    runtime = FakeRuntime()
    args.phase = "index"
    assert execute(args, runtime_factory=runtime.factory)["papers"] == 1
    assert execute(args, runtime_factory=runtime.factory)["papers"] == 1
    assert runtime.new_checks == 1
    args.phase = "retrieve"
    assert execute(args, runtime_factory=runtime.factory)["statuses"] == {"completed": 3}
    args.phase = "answer"
    with pytest.raises(ValueError, match="allow-model"):
        execute(args, runtime_factory=runtime.factory)
    args.allow_model_calls, args.max_answers = True, 1
    assert execute(args, runtime_factory=runtime.factory)["statuses"] == {"completed": 1, "pending": 2}
    assert execute(args, runtime_factory=runtime.factory)["statuses"] == {"completed": 2, "pending": 1}
    assert sum(answer for _, answer in runtime.calls) == 2


def test_failed_attempt_is_not_replayed(args):
    runtime = FakeRuntime()
    args.phase = "index"
    execute(args, runtime_factory=runtime.factory)
    def fail(*a, **k):
        raise RuntimeError("must not persist sensitive exception text")
    runtime.run_case = fail
    args.phase, args.allow_model_calls, args.max_answers = "answer", True, 3
    first = execute(args, runtime_factory=runtime.factory)
    assert first["statuses"] == {"failed": 3}
    assert execute(args, runtime_factory=runtime.factory)["new_attempts"] == 0
    assert "sensitive" not in "".join(p.read_text() for p in (args.run_dir / "answer").glob("*.json"))


def test_missing_vectors_and_changed_configuration_stop_resume(args):
    runtime = FakeRuntime()
    args.phase = "index"
    execute(args, runtime_factory=runtime.factory)
    args.phase, args.top_k = "retrieve", 9
    with pytest.raises(ValueError, match="changed"):
        execute(args, runtime_factory=runtime.factory)
    args.top_k = None
    runtime.ids.clear()
    with pytest.raises(ValueError, match="vectors are missing"):
        execute(args, runtime_factory=runtime.factory)


def test_unowned_directory_is_rejected(args):
    args.run_dir.mkdir()
    (args.run_dir / "important.txt").write_text("keep")
    args.phase = "index"
    with pytest.raises(ValueError, match="Unowned"):
        execute(args, runtime_factory=lambda *a, **k: pytest.fail("must stop first"))
    assert (args.run_dir / "important.txt").read_text() == "keep"


def test_unsupported_backend_and_credential_urls(args):
    args.backend, args.retrieval_mode = "chroma", "hybrid"
    with pytest.raises(ValueError, match="requires Milvus"):
        settings_for(args)
    with pytest.raises(ValueError, match="credentials"):
        public_settings(RAGSettings(milvus_uri="http://user:password@localhost:19530"))


def test_scholarqa_requires_independent_corpus(args, tmp_path):
    folder = args.data_root / "scholarqabench/pinned/data/scholarqa_cs"
    folder.mkdir(parents=True)
    (folder / "qa_metadata_all.jsonl").write_text(json.dumps(
        {"idx": 1, "question": "Compare methods", "key_ingredients": "SECRET_GOLD"}))
    assert not load_dataset("scholarqa-cs", args.data_root).ready
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"paper_id": "p", "title": "T", "text": "source",
                                 "answer": "SECRET_GOLD"}))
    ds = load_dataset("scholarqa-cs", args.data_root, corpus=corpus, provenance="fixture")
    with pytest.raises(ValueError, match="only paper_id"):
        list(ds.sources())
    corpus.write_text(json.dumps({"paper_id": "p", "title": "T", "text": "source"}))
    assert list(ds.sources())[0].text == "source"


def test_unsupported_dataset_options_fail_instead_of_being_ignored(args, tmp_path):
    with pytest.raises(ValueError, match="only to ScholarQA-CS"):
        load_dataset("qasper", args.data_root, corpus=tmp_path / "unused.jsonl")
    with pytest.raises(ValueError, match="only to QASPER"):
        load_dataset("scholarqa-cs", args.data_root, split="train")


def test_current_answer_adapter_observes_real_graph():
    runtime = CurrentRuntime.__new__(CurrentRuntime)
    runtime.settings = RAGSettings()
    runtime.generation = True
    runtime.client = Transport()
    retrieval = Retrieval()
    runtime.reader = NS(retriever=retrieval, qa=PaperQA(retrieval, runtime.client))
    result = runtime.run_case(Case("q", "How many layers?", ("a",)), "qasper", answer=True)
    assert result["answer"]["sources"] == ["a-1"]
    assert result["retrieved"]
    assert result["model_calls"] == 1
    assert {"retrieve", "prepare", "generate", "validate"} <= set(result["graph_nodes"])
    before = len(runtime.client.inputs)
    assert runtime.run_case(Case("q", "layers?", ("a",)), "qasper")["model_calls"] == 0
    assert len(runtime.client.inputs) == before


def test_litsearch_source_protocol(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    corpus = args.data_root / "litsearch/corpus_clean"
    query = args.data_root / "litsearch/query"
    corpus.mkdir(parents=True)
    query.mkdir()
    pq.write_table(pa.Table.from_pylist([{"corpusid": 1, "title": "T", "abstract": "A"}]),
                   corpus / "part.parquet")
    pq.write_table(pa.Table.from_pylist([{"query": "find", "corpusids": [1]}]),
                   query / "full-00000-of-00001.parquet")
    ds = load_dataset("litsearch", args.data_root)
    assert list(ds.sources())[0].representation == "title_abstract"
    assert len(ds.cases) == 1


@pytest.mark.parametrize("representation", ["full_text", "title_abstract"])
def test_runtime_indexes_and_reads_back_real_local_store(tmp_path, monkeypatch, representation):
    from evaluation import current_runtime
    from evaluation.current_datasets import Source
    from rag_reader.pipeline import RAGReader
    from test_m1_rag import KeywordEmbedding
    monkeypatch.setattr(current_runtime, "RAGReader",
                        lambda **kw: RAGReader(embedding_model=KeywordEmbedding(), **kw))
    runtime = CurrentRuntime(RAGSettings(vector_backend="chroma",
                                         collection_name="eval_current_fixture"), tmp_path)
    runtime.ensure_new_index()
    result = runtime.index_sources([Source("a", "Title", "Six layers using kiwi.", representation)])
    assert runtime.verify_ids(result["a"])
    record = runtime.run_case(Case("q", "kiwi", ("a",)), "qasper")
    assert record["retrieved"][0]["paper_id"] == "a"
    assert record["model_calls"] == 0
