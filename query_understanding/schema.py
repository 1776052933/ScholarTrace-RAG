"""科研查询理解模块的数据结构定义。

本文件使用 Pydantic 描述查询理解各阶段交换的数据。使用统一的数据模型有三点好处：

1. 在对象创建时自动检查字段类型和取值范围，尽早发现上游解析错误；
2. 为查询解析器、改写器、拆解器以及其他 Agent 提供稳定的数据契约；
3. 可以直接生成 JSON 兼容字典或 JSON 字符串，便于持久化及跨 Agent 传递。

``QueryAnalysis`` 是推荐给新调用方使用的完整查询分析模型。文件后半部分的
``ResearchEntity``、``QueryConstraints`` 和 ``QueryUnderstandingResult`` 等模型
用于兼容当前查询理解流水线，避免已有调用方因数据结构升级而失效。
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 科研实体类型采用 Literal 而不是任意字符串，使非法类型能在模型构造阶段被拒绝。
# method：研究方法；task：研究任务；domain：学科/业务领域；dataset：数据集；
# model：具体模型；application：实际应用或落地场景。
EntityType = Literal["method", "task", "domain", "dataset", "model", "application"]

# 实体重要程度。调用方可依据该字段决定检索词权重或是否保留实体。
Importance = Literal["high", "medium", "low"]


class SchemaModel(BaseModel):
    """所有查询理解数据模型的公共基类。

    公共配置会禁止未声明字段，并自动清理字符串两端空白。禁止额外字段可以防止
    字段拼写错误被静默接受；例如将 ``keywords`` 误写为 ``keyword`` 时会立即报错。
    """

    model_config = ConfigDict(
        # 上游传入模型中不存在的字段时抛出 ValidationError。
        extra="forbid",
        # 对所有字符串字段统一执行 strip，避免空格影响检索和去重。
        str_strip_whitespace=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """返回可直接交给 ``json.dumps`` 的字典。

        ``mode="json"`` 会将枚举等 Python 对象转换为对应的 JSON 基础类型。
        """
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        """将模型序列化为 JSON 字符串。

        ``kwargs`` 会原样传递给 Pydantic，例如 ``to_json(indent=2)`` 可输出
        便于阅读的缩进格式。
        """
        return self.model_dump_json(**kwargs)


class Entity(SchemaModel):
    """从用户查询中抽取出的关键科研实体。

    一个实体不仅记录原始名称，还记录语义类型和重要程度。检索 Agent 可以用类型
    生成字段化检索条件，用重要程度控制召回与精确率之间的平衡。
    """

    # 实体在查询中的名称或规范化后的标准名称，例如“图神经网络”或“CIFAR-10”。
    name: str = Field(min_length=1, description="实体名称")
    # 类型只能取 EntityType 定义的六种值之一。
    type: EntityType = Field(description="实体类型")
    # 未显式指定时按中等重要度处理，避免普通实体被过度加权。
    importance: Importance = Field(default="medium", description="实体重要性")


class TimeConstraint(SchemaModel):
    """用户对论文发表时间提出的限制。

    ``start_year`` 和 ``end_year`` 均允许为空，以表达开放区间。例如“2020 年以来”
    只需填写 ``start_year=2020``；“截至 2022 年”只需填写 ``end_year=2022``。
    ``description`` 保留用户原始时间表达，供结果解释或后续重新解析使用。
    """

    # 起始年份（包含该年）。限定在合理的科研文献年份范围内。
    start_year: Optional[int] = Field(default=None, ge=1800, le=2200, description="起始年份（含）")
    # 结束年份（包含该年）。为空表示不设置上界。
    end_year: Optional[int] = Field(default=None, ge=1800, le=2200, description="结束年份（含）")
    # 用户原始描述，例如“近五年”“2020 年以来”。
    description: Optional[str] = Field(default=None, description="原始时间限制描述")

    @model_validator(mode="after")
    def validate_year_range(self) -> "TimeConstraint":
        """确保闭合时间区间按时间正序排列。"""
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            raise ValueError("start_year must not be greater than end_year")
        return self


class QueryAnalysis(SchemaModel):
    """用户科研查询的完整结构化分析结果。

    该模型覆盖“原始输入 → 意图与约束 → 子查询 → 扩展检索式 → 关键词”的完整
    查询理解链路。列表字段均使用 ``default_factory=list``，保证不同模型实例不会
    共享可变列表，同时允许上游只填写已经识别出的信息。
    """

    # 完整保留用户输入，用于审计、结果展示和失败回退。
    original_query: str = Field(min_length=1, description="用户原始输入")
    # 对用户研究目标的简洁自然语言概括，例如“比较两类目标检测方法的性能”。
    research_intent: str = Field(min_length=1, description="研究目标")
    # 从原始查询中识别出的结构化实体集合。
    entities: List[Entity] = Field(default_factory=list, description="关键科研实体")
    # 必须使用、排除或比较的方法约束。
    method_constraints: List[str] = Field(default_factory=list, description="方法限制")
    # 学科、行业或应用场景方面的约束。
    domain_constraints: List[str] = Field(default_factory=list, description="领域限制")
    # 明确指定或需要检索的数据集约束。
    dataset_constraints: List[str] = Field(default_factory=list, description="数据集限制")
    # 任务类别，例如 classification、generation 或 semantic_segmentation。
    task_type: Optional[str] = Field(default=None, description="科研任务类型")
    # 时间限制；没有时间要求时为 None。
    time_constraints: Optional[TimeConstraint] = Field(default=None, description="论文时间限制")
    # 复杂问题拆解后得到的、可以单独执行检索的子查询。
    sub_queries: List[str] = Field(default_factory=list, description="拆解后的子查询")
    # 通过同义词、缩写或英文术语扩展得到的完整查询表达。
    expanded_queries: List[str] = Field(default_factory=list, description="扩展后的检索查询")
    # 适合直接提交给搜索引擎或文献数据库的核心检索词。
    keywords: List[str] = Field(default_factory=list, description="检索关键词")


# 向后兼容的流水线数据契约。它们同样是 Pydantic 模型，
# 旧调用方可以继续使用，新 Agent 可直接使用 QueryAnalysis。
class QueryIntent(str, Enum):
    """当前流水线支持的标准研究意图枚举。"""

    # 寻找某个主题的综述或系统性文献回顾。
    LITERATURE_REVIEW = "literature_review"
    # 比较多个方法、模型或算法。
    METHOD_COMPARISON = "method_comparison"
    # 检索当前最佳方法或最先进结果。
    STATE_OF_THE_ART = "state_of_the_art"
    # 寻找适用于目标任务的数据集或基准。
    DATASET_DISCOVERY = "dataset_discovery"
    # 寻找解决具体科研问题的技术方案。
    PROBLEM_SOLUTION = "problem_solution"
    # 分析研究主题随时间的变化或发展方向。
    TREND_ANALYSIS = "trend_analysis"
    # 寻找可用于复现论文实验的信息。
    REPLICATION = "replication"
    # 无法归入上述类别时使用的通用检索意图。
    GENERAL_SEARCH = "general_search"


class ResearchEntity(SchemaModel):
    """流水线内部使用的实体表示，保留原文本、规范名称和别名。"""

    text: str = Field(min_length=1, description="查询中出现的实体原文")
    entity_type: str = Field(min_length=1, description="解析器识别出的实体类型")
    normalized: Optional[str] = Field(default=None, description="规范化实体名称")
    aliases: List[str] = Field(default_factory=list, description="实体别名或同义表达")


class QueryConstraints(SchemaModel):
    """流水线解析出的各类显式检索约束。"""

    methods: List[str] = Field(default_factory=list, description="方法约束")
    domains: List[str] = Field(default_factory=list, description="领域约束")
    datasets: List[str] = Field(default_factory=list, description="数据集约束")
    time: Optional[TimeConstraint] = Field(default=None, description="时间约束")
    # 用于承载暂未标准化但仍需保留的约束，避免丢失用户信息。
    other: Dict[str, Any] = Field(default_factory=dict, description="其他扩展约束")


class SubQuery(SchemaModel):
    """从复杂查询中拆解出的单个可执行检索单元。"""

    id: str = Field(min_length=1, description="子查询唯一标识")
    query: str = Field(min_length=1, description="子查询文本")
    purpose: str = Field(min_length=1, description="该子查询在整体研究中的目的")
    # 保存前置子查询 ID，供编排 Agent 按依赖顺序执行。
    depends_on: List[str] = Field(default_factory=list, description="依赖的子查询 ID")
    keywords: List[str] = Field(default_factory=list, description="子查询检索关键词")


class QueryUnderstandingResult(SchemaModel):
    """现有查询理解流水线 ``run`` 方法返回的完整结果。"""

    original_query: str = Field(description="用户原始查询")
    normalized_query: str = Field(description="清理和规范化后的查询")
    language: str = Field(description="查询语言，如 zh、en 或 mixed")
    intent: QueryIntent = Field(description="标准化研究意图")
    entities: List[ResearchEntity] = Field(description="解析得到的科研实体")
    constraints: QueryConstraints = Field(description="方法、领域、数据集及时间约束")
    complexity: str = Field(description="查询复杂度：simple、compound 或 complex")
    subqueries: List[SubQuery] = Field(description="拆解后的结构化子查询")
    rewritten_query: str = Field(description="面向学术检索改写后的查询")
    expanded_keywords: List[str] = Field(description="扩展后的检索关键词")
    # 置信度严格限制在 0 到 1 之间。
    confidence: float = Field(ge=0.0, le=1.0, description="查询解析置信度")
    warnings: List[str] = Field(default_factory=list, description="回退或异常警告")
    # 记录解析引擎等辅助信息，便于追踪结果来源。
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展元数据")
