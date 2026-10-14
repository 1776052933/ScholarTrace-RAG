"""search_agent 的 Pydantic 公共数据契约。"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Priority = Literal["high", "medium", "low"]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


class SearchQuery(SchemaModel):
    """一个可以提交给指定论文数据源的搜索语句。"""

    query: str = Field(min_length=1)
    source: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    priority: Priority = "medium"


class SearchPlan(SchemaModel):
    """Agent 在某一搜索轮次执行的完整计划。"""

    original_query: str = Field(min_length=1)
    search_round: int = Field(ge=1)
    queries: List[SearchQuery] = Field(min_length=1)
    strategy: str


class PaperMetadata(SchemaModel):
    """跨论文数据库统一后的论文元数据。"""

    title: str = Field(min_length=1)
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    publication_date: Optional[str] = None
    venue: str = ""
    citation_count: Optional[int] = Field(default=None, ge=0)
    url: str = ""
    source: str = Field(min_length=1)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None
    pdf_url: Optional[str] = None
    matched_queries: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    citation_seed_ids: List[str] = Field(default_factory=list)
    citation_directions: List[str] = Field(default_factory=list)
    citation_parent_ids: List[str] = Field(default_factory=list)
    citation_relations: List[Dict[str, Any]] = Field(default_factory=list)
    citation_depth: int = Field(default=0, ge=0, le=2)


class SearchResult(SchemaModel):
    """一个搜索语句产生的论文集合。"""

    papers: List[PaperMetadata] = Field(default_factory=list)
    total: int = Field(ge=0)
    query_used: str
    query_sources: Dict[str, List[str]] = Field(default_factory=dict)


class SearchEvaluation(SchemaModel):
    """当前论文结果的相关性、质量和覆盖度评估。"""

    relevance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    missing_information: List[str] = Field(default_factory=list)
    need_refine: bool


class AgentState(SchemaModel):
    """自主搜索 Agent 的可序列化运行状态。"""

    current_round: int = Field(default=0, ge=0)
    search_history: List[SearchResult] = Field(default_factory=list)
    papers_found: List[PaperMetadata] = Field(default_factory=list)
    plans: List[SearchPlan] = Field(default_factory=list)
    evaluations: List[SearchEvaluation] = Field(default_factory=list)


# 以下模型用于 Agent 内部编排，并建立在上述公共契约之上。
class SearchSubQuery(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)
    query: str = Field(min_length=1)
    purpose: str = "论文检索"
    query_type: str = "method_search"
    priority: Priority = "medium"
    english_query: str = ""
    keywords: List[str] = Field(default_factory=list)


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)
    original_query: str = Field(min_length=1)
    need_decompose: bool = False
    reason: str = ""
    sub_queries: List[SearchSubQuery] = Field(min_length=1)


class SearchTask(SearchQuery):
    id: str
    sub_query_index: int = Field(ge=0)
    keywords: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)


class SearchStrategy(SearchPlan):
    """包含内部任务字段的可执行搜索计划。"""

    queries: List[SearchTask] = Field(min_length=1)


class Paper(PaperMetadata):
    paper_id: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Reflection(SchemaModel):
    should_continue: bool
    reason: str
    add_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)
    source_adjustments: List[str] = Field(default_factory=list)


class SearchIteration(SchemaModel):
    round_number: int
    plan: SearchPlan
    new_paper_count: int
    total_paper_count: int
    evaluation: SearchEvaluation
    reflection: Reflection


class SearchAgentResult(SchemaModel):
    original_query: str
    rounds_completed: int
    stopped_reason: str
    papers: List[PaperMetadata] = Field(default_factory=list)
    iterations: List[SearchIteration] = Field(default_factory=list)
    final_evaluation: SearchEvaluation
