"""Ranking Agent 的 Pydantic 输入、论文、评分和输出数据结构。"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


RelevanceLevel = Literal[
    "highly_relevant",
    "relevant",
    "partially_relevant",
]


class SchemaModel(BaseModel):
    """Ranking Agent 公共模型基类。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """返回适合 JSON 序列化的字典。"""
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        """返回 JSON 字符串。"""
        return self.model_dump_json(**kwargs)


class QueryContext(SchemaModel):
    """论文精排所需的用户查询上下文。"""

    original_query: str = Field(min_length=1, description="用户原始科研查询")
    research_intent: str = Field(min_length=1, description="用户的研究目标")
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="查询理解阶段抽取的关键实体",
    )
    method_constraints: List[str] = Field(
        default_factory=list,
        description="方法或模型约束",
    )
    domain_constraints: List[str] = Field(
        default_factory=list,
        description="研究领域或应用场景约束",
    )
    task_type: str = Field(default="", description="科研任务类型")
    keywords: List[str] = Field(
        default_factory=list,
        description="查询关键词及扩展关键词",
    )


class Paper(SchemaModel):
    """Ranking Agent 接收的统一论文元数据。"""

    paper_id: str = Field(min_length=1, description="论文唯一标识")
    title: str = Field(min_length=1, description="论文标题")
    abstract: str = Field(default="", description="论文摘要")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    year: Optional[int] = Field(default=None, description="发表年份")
    venue: str = Field(default="", description="期刊、会议或预印本平台")
    citation_count: int = Field(default=0, ge=0, description="引用次数")
    url: str = Field(default="", description="论文访问地址")
    doi: Optional[str] = Field(default=None, description="数字对象唯一标识符")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv 论文标识")
    openalex_id: Optional[str] = Field(default=None, description="OpenAlex 论文标识")
    pdf_url: str = Field(default="", description="已知 PDF 直链")
    source: str = Field(default="", description="主要元数据来源")
    sources: List[str] = Field(default_factory=list, description="合并后的元数据来源")
    keywords: List[str] = Field(default_factory=list, description="论文关键词")
    citation_parent_ids: List[str] = Field(default_factory=list, description="引文追踪父节点")
    citation_directions: List[str] = Field(default_factory=list, description="引文追踪方向")
    citation_relations: List[Dict[str, Any]] = Field(default_factory=list, description="父子发现关系")
    citation_depth: int = Field(default=0, ge=0, le=2, description="引文追踪层级")
    relation_keys: List[str] = Field(default_factory=list, description="关系图身份键")


class PaperScore(SchemaModel):
    """单篇论文在各精排维度上的归一化评分。"""

    semantic_score: float = Field(ge=0.0, le=1.0, description="语义相关性评分")
    method_score: float = Field(ge=0.0, le=1.0, description="方法约束匹配评分")
    task_score: float = Field(ge=0.0, le=1.0, description="研究任务匹配评分")
    time_score: float = Field(ge=0.0, le=1.0, description="论文时效性评分")
    quality_score: float = Field(ge=0.0, le=1.0, description="论文质量评分")
    llm_score: float = Field(ge=0.0, le=1.0, description="大语言模型综合判断评分")
    final_score: float = Field(ge=0.0, le=1.0, description="加权后的最终评分")
    relevance_level: RelevanceLevel = Field(description="论文相关等级")


class PaperGraphPoint(SchemaModel):
    """论文在查询中心三维关系图中的展示坐标。"""

    paper_id: str = Field(min_length=1, description="论文唯一标识")
    title: str = Field(min_length=1, description="论文标题")
    rank: int = Field(ge=1, description="最终精排排名")
    x: float = Field(description="三维坐标 X")
    y: float = Field(description="三维坐标 Y")
    z: float = Field(description="三维坐标 Z")
    distance: float = Field(ge=0.0, description="与查询中心的距离")
    final_score: float = Field(ge=0.0, le=1.0, description="最终相关性得分")
    relevance_level: RelevanceLevel = Field(description="自适应相关等级")
    llm_evaluated: bool = Field(description="是否经过 DeepSeek 二阶段评价")
    citation_depth: int = Field(default=0, ge=0, le=2, description="引文追踪层级")
    relation_keys: List[str] = Field(default_factory=list, description="关系图身份键")


class PaperGraphEdge(SchemaModel):
    """论文发现关系；研究目标使用固定 source_id。"""

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    direction: str = "search"
    depth: int = Field(default=0, ge=0, le=2)


class RankingResult(SchemaModel):
    """论文精排完成后的输出。papers 按相关程度降序排列。"""

    papers: List[Paper] = Field(default_factory=list, description="排序后的论文列表")
    total: int = Field(ge=0, description="输出论文总数")
    ranking_time: float = Field(ge=0.0, description="精排耗时，单位为秒")
    scores: Dict[str, PaperScore] = Field(default_factory=dict, description="全部去重论文的评分")
    graph_points: List[PaperGraphPoint] = Field(default_factory=list, description="三维关系图数据")
    graph_edges: List[PaperGraphEdge] = Field(default_factory=list, description="去重后的论文发现关系")
    llm_evaluated_count: int = Field(default=0, ge=0, description="DeepSeek 二阶段评价数量")
    highly_relevant_count: int = Field(default=0, ge=0, description="自适应高度相关论文数量")
    adaptive_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="高度相关自适应阈值")
