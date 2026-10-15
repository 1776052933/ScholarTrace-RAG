"""RAG Reader 的论文文档、切片、知识和回答数据契约。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """提供严格校验和统一 JSON 序列化能力的模型基类。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """返回可直接 JSON 序列化的字典。"""
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        """返回 JSON 字符串。"""
        return self.model_dump_json(**kwargs)


class PaperDocument(SchemaModel):
    """从 Literature Database 加载后的完整论文文档。"""

    paper_id: str = Field(min_length=1, description="论文唯一标识")
    title: str = Field(min_length=1, description="论文标题")
    file_path: Optional[str] = Field(default=None, description="PDF 或原始全文路径")
    text_path: Optional[str] = Field(default=None, description="已解析纯文本路径")
    content: str = Field(default="", description="加载后的论文全文内容")


class DocumentChunk(SchemaModel):
    """用于向量化和检索的论文文本切片。"""

    chunk_id: str = Field(min_length=1, description="切片唯一标识")
    paper_id: str = Field(min_length=1, description="所属论文标识")
    text: str = Field(min_length=1, description="切片正文")
    page: Optional[int] = Field(default=None, ge=1, description="来源页码")
    section: str = Field(default="", description="所属论文章节")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="扩展检索元数据")


class PaperKnowledge(SchemaModel):
    """从论文全文中抽取的结构化科研知识。"""

    title: str = Field(min_length=1)
    research_problem: str = ""
    method: str = ""
    model: str = ""
    dataset: str = ""
    experiment: str = ""
    metrics: str = ""
    results: str = ""
    advantages: str = ""
    limitations: str = ""
    future_work: str = ""


class EvidenceCitation(SchemaModel):
    """A validated, locatable source excerpt returned with an answer."""

    paper_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    title: str = ""
    page: Optional[int] = Field(default=None, ge=1)
    section: str = ""
    excerpt: str = Field(min_length=1)
    retrieval_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class RAGAnswer(SchemaModel):
    """基于论文检索证据生成的回答。"""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    sources: List[str] = Field(default_factory=list, description="引用的论文或切片标识")
    citations: List[EvidenceCitation] = Field(default_factory=list, description="可定位原文证据")
    confidence: float = Field(ge=0.0, le=1.0, description="回答置信度")
