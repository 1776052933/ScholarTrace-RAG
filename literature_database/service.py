"""供后续 RAG 模块调用的论文资源数据库服务。"""

from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from .availability import AvailabilityChecker
from .database import DatabaseManager
from .downloader import DownloadFunction, PaperDownloadCoordinator
from .models import PaperFileModel, PaperModel
from .paper_manager import PaperManager
from .repository import FileRepository
from .schema import (
    PaperAvailability,
    PaperImportance,
    PaperMetadata,
    PaperResource,
    UploadRequest,
)
from .upload_queue import UploadQueueManager


class LiteratureService:
    """注册精排论文、准备 RAG 文档并暴露人工上传提醒。"""

    def __init__(
        self,
        database: Optional[DatabaseManager] = None,
        *,
        create_tables: bool = True,
        download_function: Optional[DownloadFunction] = None,
    ) -> None:
        self.database = database or DatabaseManager()
        self.download_function = download_function
        self.last_download_results: List[Dict[str, Any]] = []
        if create_tables:
            self.database.create_tables()

    def register_ranked_papers(
        self,
        ranking_result: Any,
        *,
        scores: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """注册 Ranking Agent 输出，并返回每篇论文的资源准备状态。"""
        self.last_download_results = []
        candidates = _paper_values(ranking_result)
        score_map = scores or _score_map(ranking_result)
        states = []
        with self.database.get_session() as session:
            manager = PaperManager(session)
            for paper in candidates:
                paper_id = _value(paper, "paper_id")
                score = _final_score(score_map.get(str(paper_id))) if paper_id else None
                states.append(manager.register_paper(paper, rank_score=score))
            session.commit()
        return states

    def download_registered_papers(
        self,
        paper_ids: Iterable[str],
        *,
        output_dir: str = "./papers",
        engine: str = "academic_oa",
        timeout: int = 30,
        retry_failed: bool = False,
    ) -> List[Dict[str, Any]]:
        """执行已注册论文的下载任务并持久化结果。"""
        self.last_download_results = []
        with self.database.get_session() as session:
            coordinator = PaperDownloadCoordinator(
                session,
                download_function=self.download_function,
                output_dir=output_dir,
                engine=engine,
                timeout=timeout,
            )
            results = coordinator.download_many(
                paper_ids,
                retry_failed=retry_failed,
            )
            self.last_download_results = list(results)
            session.commit()
            return results

    def prepare_for_reader(
        self,
        ranking_result: Any,
        *,
        scores: Optional[Mapping[str, Any]] = None,
        target_count: Optional[int] = None,
        output_dir: str = "./papers",
        engine: str = "academic_oa",
        timeout: int = 30,
        retry_failed: bool = False,
    ) -> List[PaperResource]:
        """注册、下载并返回具有本地 PDF、可供 RAGReader 摄取的资源。"""
        if target_count is None:
            states = self.register_ranked_papers(ranking_result, scores=scores)
            paper_ids = [
                str(state["paper_id"])
                for state in states
                if state.get("paper_id") not in {None, ""}
            ]
            results = self.download_registered_papers(
                paper_ids,
                output_dir=output_dir,
                engine=engine,
                timeout=timeout,
                retry_failed=retry_failed,
            )
            selected_ids = [
                str(item["paper_id"])
                for item in results
                if item.get("success") and item.get("paper_id")
            ]
        else:
            target_count = max(0, int(target_count))
            candidates = _paper_values(ranking_result)
            score_map = scores or _score_map(ranking_result)
            selected_ids = []
            all_results: List[Dict[str, Any]] = []
            cursor = 0
            while len(selected_ids) < target_count and cursor < len(candidates):
                remaining = target_count - len(selected_ids)
                batch = candidates[cursor:cursor + remaining]
                cursor += len(batch)
                states = self.register_ranked_papers(
                    {"papers": batch},
                    scores=score_map,
                )
                batch_ids = [
                    str(state["paper_id"])
                    for state in states
                    if state.get("paper_id") not in {None, ""}
                ]
                batch_results = self.download_registered_papers(
                    batch_ids,
                    output_dir=output_dir,
                    engine=engine,
                    timeout=timeout,
                    retry_failed=retry_failed,
                )
                all_results.extend(batch_results)
                selected_ids.extend(
                    str(item["paper_id"])
                    for item in batch_results
                    if item.get("success") and item.get("paper_id")
                )
            self.last_download_results = all_results

        return self._resources(selected_ids[:target_count], require_full=False)

    def ingest_into_reader(
        self,
        resources: Iterable[PaperResource],
        reader: Any,
        *,
        text_output_dir: str = "./papers/text",
    ) -> List[Dict[str, Any]]:
        """摄取本地 PDF，并把解析文本及 Embedding 完成状态回写数据库。"""
        ingest = getattr(reader, "ingest_paper", None)
        get_document = getattr(reader, "get_document", None)
        if not callable(ingest) or not callable(get_document):
            raise TypeError("reader must provide ingest_paper() and get_document()")

        output_dir = Path(text_output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for resource in resources:
            if not isinstance(resource, PaperResource):
                raise TypeError("resources must contain only PaperResource values")
            paper_id = resource.paper.paper_id
            try:
                chunks = ingest(resource)
                document = get_document(paper_id)
                text_path = (output_dir / _text_filename(paper_id)).resolve()
                text_path.write_text(document.content, encoding="utf-8")
                status = self._record_reader_output(paper_id, text_path)
                results.append({
                    "paper_id": paper_id,
                    "success": True,
                    "chunk_count": len(chunks),
                    "text_path": str(text_path),
                    "status": status.value,
                    "error": None,
                })
            except Exception as exc:
                results.append({
                    "paper_id": paper_id,
                    "success": False,
                    "chunk_count": 0,
                    "text_path": None,
                    "status": resource.availability.value,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return results

    def prepare_and_ingest(
        self,
        ranking_result: Any,
        reader: Any,
        *,
        scores: Optional[Mapping[str, Any]] = None,
        target_count: Optional[int] = None,
        output_dir: str = "./papers",
        text_output_dir: str = "./papers/text",
        engine: str = "academic_oa",
        timeout: int = 30,
        retry_failed: bool = False,
    ) -> List[Dict[str, Any]]:
        """完成“注册→下载→RAG 摄取→数据库状态回写”端到端流程。"""
        resources = self.prepare_for_reader(
            ranking_result,
            scores=scores,
            target_count=target_count,
            output_dir=output_dir,
            engine=engine,
            timeout=timeout,
            retry_failed=retry_failed,
        )
        remove_paper = getattr(reader, "remove_paper", None)
        if callable(remove_paper):
            for item in self.last_download_results:
                if item.get("success") or not item.get("paper_id"):
                    continue
                remove_paper(str(item["paper_id"]))
        return self.ingest_into_reader(
            resources,
            reader,
            text_output_dir=text_output_dir,
        )

    def prepare_documents(
        self,
        ranking_result: Any,
        *,
        scores: Optional[Mapping[str, Any]] = None,
    ) -> List[PaperResource]:
        """注册候选论文并返回当前可以直接进入 RAG 的论文资源。"""
        states = self.register_ranked_papers(ranking_result, scores=scores)
        paper_ids = [str(state["paper_id"]) for state in states]
        return self._resources(paper_ids, require_full=True)

    def _resources(
        self,
        paper_ids: Iterable[str],
        *,
        require_full: bool,
    ) -> List[PaperResource]:
        resources = []
        with self.database.get_session() as session:
            checker = AvailabilityChecker(session)
            for paper_id in dict.fromkeys(value for value in paper_ids if value and value != "None"):
                model = session.get(PaperModel, paper_id)
                if model is None:
                    continue
                availability = checker.check(paper_id)
                files = list(session.scalars(
                    select(PaperFileModel)
                    .where(PaperFileModel.paper_id == paper_id)
                    .order_by(PaperFileModel.id)
                ))
                pdf_path = _first_existing_path(files, {"pdf", "fulltext", "full_text"})
                if require_full:
                    if availability is not PaperAvailability.FULL_AVAILABLE:
                        continue
                elif pdf_path is None:
                    continue
                resources.append(PaperResource(
                    paper=_metadata(model),
                    availability=availability,
                    file_path=pdf_path,
                    text_path=_first_existing_path(files, {"text", "txt", "parsed_text"}),
                    embedding_status=model.embedding_status,
                ))
        return resources

    def _record_reader_output(
        self,
        paper_id: str,
        text_path: Path,
    ) -> PaperAvailability:
        with self.database.get_session() as session:
            model = session.get(PaperModel, paper_id)
            if model is None:
                raise LookupError(f"paper not found: {paper_id}")
            files = FileRepository(session)
            normalized = str(text_path)
            existing = next(
                (
                    item
                    for item in files.list_by_paper(paper_id)
                    if item.file_type.casefold() in {"text", "txt", "parsed_text"}
                    and str(Path(item.file_path).expanduser().resolve()) == normalized
                ),
                None,
            )
            if existing is None:
                files.add_file(
                    paper_id,
                    normalized,
                    status="PARSED",
                    file_type="text",
                )
            else:
                existing.status = "PARSED"
            model.embedding_status = "COMPLETED"
            status = AvailabilityChecker(session).check(paper_id)
            model.availability = status.value
            session.commit()
            return status

    def get_missing_core_papers(self) -> List[UploadRequest]:
        """返回所有等待用户上传全文的 CORE 论文。"""
        with self.database.get_session() as session:
            requests = UploadQueueManager(session).list_pending()
            return [item for item in requests if item.priority is PaperImportance.CORE]


def _paper_values(value: Any) -> List[Any]:
    if isinstance(value, Mapping):
        papers = value.get("papers")
        if papers is None:
            papers = [value]
    elif hasattr(value, "papers"):
        papers = value.papers
    else:
        papers = value
    if isinstance(papers, (str, bytes)) or not isinstance(papers, Iterable):
        raise TypeError("ranking_result must contain an iterable papers field")
    return list(papers)


def _score_map(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        scores = value.get("scores") or value.get("last_scores")
    else:
        scores = getattr(value, "scores", None) or getattr(value, "last_scores", None)
    return scores if isinstance(scores, Mapping) else {}


def _final_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    result = _value(value, "final_score")
    return float(result) if result is not None else None


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _metadata(model: PaperModel) -> PaperMetadata:
    return PaperMetadata(
        paper_id=model.paper_id,
        title=model.title,
        authors=model.authors,
        abstract=model.abstract,
        doi=model.doi,
        arxiv_id=model.arxiv_id,
        year=model.year,
        venue=model.venue,
        citation_count=model.citation_count,
        rank_score=model.rank_score,
    )


def _first_existing_path(
    files: Iterable[PaperFileModel],
    accepted_types: set[str],
) -> Optional[str]:
    for item in files:
        if (
            item.file_type.casefold() in accepted_types
            and item.status.casefold() in {"available", "completed", "downloaded", "parsed", "success"}
            and Path(item.file_path).is_file()
        ):
            return item.file_path
    return None


def _text_filename(paper_id: str) -> str:
    digest = sha256(paper_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.txt"
