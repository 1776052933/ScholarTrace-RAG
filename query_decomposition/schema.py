"""科研查询拆解模块的 Pydantic 输入与输出契约。"""

from enum import Enum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


QueryType = Literal[
    "method_search",
    "comparison",
    "application",
    "dataset_search",
    "evaluation",
    "trend_analysis",
]
Priority = Literal["high", "medium", "low"]


class SchemaModel(BaseModel):
    """为所有数据模型提供统一 JSON 序列化接口。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


class InputEntity(BaseModel):
    """查询理解结果中的实体；允许携带 importance 等附加属性。"""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class QueryUnderstandingResult(BaseModel):
    """query_decomposition 接收的查询理解 JSON。"""

    # 允许并保留 time_constraints、sub_queries 等上游扩展字段，保证模块间传递兼容。
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    original_query: str = Field(min_length=1)
    research_intent: str = Field(min_length=1)
    entities: List[InputEntity] = Field(default_factory=list)
    method_constraints: List[str] = Field(default_factory=list)
    domain_constraints: List[str] = Field(default_factory=list)
    dataset_constraints: List[str] = Field(default_factory=list)
    task_type: str = ""
    keywords: List[str] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)


# 兼容该模块早期代码使用的输入名称。
QueryAnalysisInput = QueryUnderstandingResult


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ComplexityReport(SchemaModel):
    """复杂度分析器使用的内部结果。"""

    level: ComplexityLevel
    score: int = Field(ge=0)
    need_decompose: bool
    reasons: List[str] = Field(default_factory=list)


class SubQuery(SchemaModel):
    """能够独立执行的论文检索子查询。"""

    query: str = Field(min_length=1, description="子查询文本")
    purpose: str = Field(min_length=1, description="该子查询在整体研究中的作用")
    query_type: QueryType = Field(description="子查询类型")
    priority: Priority = Field(default="medium", description="执行优先级")
    english_query: str = Field(default="", description="适合国际论文数据库的英文检索式")
    keywords: List[str] = Field(default_factory=list, description="后续论文检索关键词")

    @property
    def search_keywords(self) -> List[str]:
        """``keywords`` 的语义化只读别名，供后续论文检索调用。"""
        return self.keywords


class DecompositionResult(SchemaModel):
    """科研查询自动拆解的最终输出。"""

    original_query: str = Field(min_length=1)
    need_decompose: bool
    reason: str
    sub_queries: List[SubQuery] = Field(default_factory=list)
