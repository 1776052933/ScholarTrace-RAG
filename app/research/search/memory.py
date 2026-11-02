"""保存自主搜索 Agent 的多轮检索记忆。"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .schema import AgentState, Paper, PaperMetadata, SearchIteration, SearchResult
from .searcher import deduplicate_papers, to_internal_paper


StringValues = Union[str, Sequence[str]]
PaperValue = Union[PaperMetadata, Mapping[str, Any]]


class SearchMemory:
    """记录已搜索查询、轮次、论文以及关键词调整历史。"""

    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        search_round: int,
        *,
        queries: StringValues = (),
        papers: Optional[Iterable[PaperValue]] = None,
        failed_keywords: StringValues = (),
        added_keywords: StringValues = (),
    ) -> Dict[str, Any]:
        """增加一轮搜索记忆，并返回本轮可 JSON 序列化的摘要。

        ``papers`` 同时接受 ``PaperMetadata`` 和字典，方便搜索工具、Agent
        或持久化层直接写入。论文会按 DOI 或标题自动去重。
        """
        if isinstance(search_round, bool) or not isinstance(search_round, int) or search_round < 1:
            raise ValueError("search_round must be a positive integer")

        round_queries = _strings(queries)
        round_failed = _strings(failed_keywords)
        round_added = _strings(added_keywords)
        parsed_papers = [_paper(value) for value in (papers or [])]
        new_paper_count = self.add_papers(parsed_papers)

        self.searched_queries = _unique([*self.searched_queries, *round_queries])
        self.failed_keywords = _unique([*self.failed_keywords, *round_failed])
        self.added_keywords = _unique([*self.added_keywords, *round_added])
        self.state.current_round = max(self.state.current_round, search_round)

        entry = {
            "search_round": search_round,
            "queries": round_queries,
            "paper_count": len(parsed_papers),
            "new_paper_count": new_paper_count,
            "failed_keywords": round_failed,
            "added_keywords": round_added,
        }
        self.search_rounds.append(entry)
        return dict(entry)

    def get_history(self) -> Dict[str, Any]:
        """返回完整、可直接 JSON 序列化的搜索历史快照。"""
        return {
            "current_round": self.state.current_round,
            "searched_queries": list(self.searched_queries),
            "search_rounds": [dict(item) for item in self.search_rounds],
            "papers_found": [paper.model_dump(mode="json") for paper in self.papers],
            "failed_keywords": list(self.failed_keywords),
            "added_keywords": list(self.added_keywords),
            "plans": [plan.model_dump(mode="json") for plan in self.state.plans],
            "evaluations": [item.model_dump(mode="json") for item in self.state.evaluations],
            "search_results": [item.model_dump(mode="json") for item in self.state.search_history],
        }

    def clear(self) -> None:
        """清空全部搜索记忆，使实例可以安全用于一个新查询。"""
        self.searched_queries: List[str] = []
        self.search_rounds: List[Dict[str, Any]] = []
        self.papers: List[Paper] = []
        self.failed_keywords: List[str] = []
        self.added_keywords: List[str] = []
        self.iterations: List[SearchIteration] = []
        self.state = AgentState()

    # 下列方法保留给现有 Pipeline 使用，并同步维护统一记忆。
    def add_papers(self, papers: Iterable[PaperMetadata]) -> int:
        before = len(self.papers)
        self.papers = deduplicate_papers([*self.papers, *papers])
        self.state.papers_found = list(self.papers)
        return len(self.papers) - before

    def add_search_result(self, result: SearchResult) -> None:
        self.state.search_history.append(result)
        queries = list(result.query_sources) or _split_queries(result.query_used)
        self.searched_queries = _unique([*self.searched_queries, *queries])

        matched = {
            query.casefold()
            for paper in result.papers
            for query in paper.matched_queries
            if query.strip()
        }
        failed = [query for query in queries if query.casefold() not in matched]
        self.failed_keywords = _unique([*self.failed_keywords, *failed])

    def add_iteration(self, iteration: SearchIteration) -> None:
        self.iterations.append(iteration)
        self.state.current_round = iteration.round_number
        self.state.plans.append(iteration.plan)
        self.state.evaluations.append(iteration.evaluation)
        queries = [query.query for query in iteration.plan.queries]
        new_keywords = iteration.reflection.add_keywords
        self.searched_queries = _unique([*self.searched_queries, *queries])
        self.added_keywords = _unique([*self.added_keywords, *new_keywords])
        self.search_rounds.append(
            {
                "search_round": iteration.round_number,
                "queries": queries,
                "paper_count": iteration.new_paper_count,
                "new_paper_count": iteration.new_paper_count,
                "total_paper_count": iteration.total_paper_count,
                "failed_keywords": [
                    query for query in queries if query.casefold() in {item.casefold() for item in self.failed_keywords}
                ],
                "added_keywords": list(new_keywords),
            }
        )


def _paper(value: PaperValue) -> Paper:
    metadata = value if isinstance(value, PaperMetadata) else PaperMetadata.model_validate(dict(value))
    return metadata if isinstance(metadata, Paper) else to_internal_paper(metadata)


def _strings(values: StringValues) -> List[str]:
    if isinstance(values, str):
        values = [values]
    return _unique(values)


def _split_queries(value: str) -> List[str]:
    return _unique(part.strip() for part in value.replace(";", "；").split("；"))


def _unique(values: Iterable[str]) -> List[str]:
    seen, result = set(), []
    for raw in values:
        value = str(raw).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result
