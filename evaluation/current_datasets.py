"""Source-only adapters for the current RAG runner; references stay outside RAG."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterator


@dataclass(frozen=True)
class SourceSection:
    name: str
    text: str


@dataclass(frozen=True)
class Source:
    paper_id: str
    title: str
    text: str
    representation: str = "full_text"
    sections: tuple[SourceSection, ...] = ()


def qasper_source(paper_id: str, paper: dict) -> Source:
    """Carry source section boundaries directly; never inspect QA annotations."""
    sections = [SourceSection('Title', paper['title'])]
    if paper.get('abstract'):
        sections.append(SourceSection('Abstract', paper['abstract']))
    sections.extend(SourceSection(str(s.get('section_name') or ''),
                                  '\n\n'.join(s.get('paragraphs') or []))
                    for s in paper.get('full_text', []) if s.get('paragraphs'))
    return Source(paper_id, paper['title'], '\n\n'.join(s.text for s in sections),
                  sections=tuple(sections))


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    paper_ids: tuple[str, ...] = ()


@dataclass
class Dataset:
    name: str
    cases: list[Case]
    sources: Callable[[], Iterator[Source]]
    files: list[Path]
    references: dict
    protocol: dict
    ready: bool = True


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_dataset(name: str, root: Path, *, split="test", limit=0, seed=42,
                 corpus_limit=0, corpus: Path | None = None, provenance="",
                 case_ids_file: Path | None = None) -> Dataset:
    if limit < 0 or corpus_limit < 0:
        raise ValueError("limits must be non-negative; zero means all")
    if name != "qasper" and split != "test":
        raise ValueError("--split applies only to QASPER")
    if name != "scholarqa-cs" and (corpus is not None or provenance):
        raise ValueError("--corpus-jsonl and --corpus-provenance apply only to ScholarQA-CS")
    if name != "litsearch" and corpus_limit:
        raise ValueError("--corpus-limit is only supported for LitSearch")
    if case_ids_file is not None and (name != "qasper" or limit):
        raise ValueError("--case-ids-file requires QASPER and replaces --limit")
    if name == "qasper":
        from .run_m0_qasper import select_questions
        path = root / "qasper" / f"qasper-{split}-v0.3.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if case_ids_file is None:
            selected = select_questions(data, limit, seed)
        else:
            ids = json.loads(case_ids_file.read_text(encoding="utf-8"))
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(qid, str) or not qid for qid in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Expected a non-empty JSON list of unique question IDs")
            available = {q["question_id"]: (p, q) for p, q in select_questions(data, 0, seed)}
            if set(ids) - available.keys():
                raise ValueError("Case IDs are absent from the selected QASPER split")
            selected = [available[qid] for qid in ids]
        ids = sorted({paper_id for paper_id, _ in selected})
        cases = [Case(q["question_id"], q["question"], (p,)) for p, q in selected]

        def sources():
            for paper_id in ids:
                paper = data[paper_id]
                yield qasper_source(paper_id, paper)

        return Dataset(name, cases, sources, [path, *([case_ids_file] if case_ids_file else [])],
                       {q["question_id"]: q.get("answers", []) for _, q in selected},
                       {"split": split, "seed": seed, "limit": limit,
                        "source": "structured_full_text", "papers": len(ids),
                        "available_questions": sum(len(p.get("qas", [])) for p in data.values()),
                        "pdf_pages": False})
    if name == "litsearch":
        import pyarrow.parquet as pq
        from .litsearch import load_queries
        from .run_m0_litsearch import corpus_rows
        files = sorted((root / "litsearch" / "corpus_clean").glob("*.parquet"))
        if not files:
            raise ValueError("LitSearch corpus shards are missing")
        query_file = root / "litsearch" / "query" / "full-00000-of-00001.parquet"
        queries = load_queries(query_file)
        selected = queries[:limit] if limit else queries
        available = sum(pq.ParquetFile(p).metadata.num_rows for p in files)

        def sources():
            from itertools import islice
            for row in islice(corpus_rows(files), corpus_limit or available):
                if row.get("corpusid") is None:
                    raise ValueError("LitSearch row has no corpusid")
                title = str(row.get("title") or "").strip()
                abstract = str(row.get("abstract") or "").strip()
                # Official utils.get_clean_title_abstract template also retains
                # records whose title and abstract are both missing. Never fill
                # these from gold answers or silently shrink the corpus.
                yield Source(f"litsearch-{int(row['corpusid'])}", title or str(row["corpusid"]),
                             f"Title: {title}\nAbstract: {abstract}", "title_abstract")

        return Dataset(name, [Case(q["query_id"], q["query"]) for q in selected],
                       sources, [query_file, *files],
                       {q["query_id"]: q["gold_corpus_ids"] for q in selected},
                       {"source": "fixed_title_abstract_corpus", "limit": limit,
                        "corpus_limit": corpus_limit, "available_papers": available,
                        "target_papers": min(corpus_limit or available, available),
                        "available_questions": len(queries), "live_search": False,
                        "text_template": "Title: {title}\\nAbstract: {abstract}",
                        "empty_source_policy": "retain template, do not impute content"})
    if name != "scholarqa-cs":
        raise ValueError(f"Unknown dataset: {name}")
    matches = sorted((root / "scholarqabench").glob("*/data/scholarqa_cs/qa_metadata_all.jsonl"))
    if len(matches) != 1:
        raise ValueError("Expected one pinned ScholarQA-CS snapshot")
    rows = list(jsonl(matches[0]))
    selected = rows[:limit] if limit else rows
    cases = [Case(f"scholarqa-cs-{row['idx']}", row["question"]) for row in selected]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate ScholarQA-CS IDs")
    if corpus is not None and not provenance.strip():
        raise ValueError("--corpus-provenance is required with --corpus-jsonl")

    def sources():
        if corpus is None:
            raise ValueError("ScholarQA-CS requires a separate source-only corpus JSONL")
        seen = set()
        for row in jsonl(corpus):
            if set(row) != {"paper_id", "title", "text"}:
                raise ValueError("Corpus rows must contain only paper_id, title, text (no answers/rubrics)")
            if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
                raise ValueError("Corpus fields must be non-empty strings")
            if row["paper_id"] in seen:
                raise ValueError("Duplicate corpus paper_id")
            seen.add(row["paper_id"])
            yield Source(**row)

    # Keep the evaluation material out of both Case and Source.
    references = {case.case_id: {k: row.get(k) for k in ("key_ingredients", "sources")}
                  for case, row in zip(cases, selected)}
    return Dataset(name, cases, sources, [matches[0], *([corpus] if corpus else [])],
                   references, {"source": "explicit_external_corpus",
                                "corpus_provenance": provenance, "limit": limit,
                                "available_questions": len(rows),
                                "max_selected_papers": 5,
                                "official_benchmark_protocol": False},
                   ready=corpus is not None)
