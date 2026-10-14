"""论文资源数据库的 Pydantic 数据契约。"""

from enum import Enum
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
        """返回可直接进行 JSON 序列化的字典。"""
        return self.model_dump(mode="json")

    def to_json(self, **kwargs: Any) -> str:
        """返回 JSON 字符串。"""
        return self.model_dump_json(**kwargs)


class PaperMetadata(SchemaModel):
    """数据库保存的统一论文元数据。"""

    paper_id: str = Field(min_length=1, description="论文唯一标识")
    title: str = Field(min_length=1, description="论文标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    abstract: str = Field(default="", description="论文摘要")
    doi: Optional[str] = Field(default=None, description="数字对象唯一标识符")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv 论文标识")
    year: Optional[int] = Field(default=None, ge=1000, description="发表年份")
    venue: str = Field(default="", description="期刊、会议或预印本平台")
    citation_count: int = Field(default=0, ge=0, description="引用次数")
    rank_score: float = Field(default=0.0, ge=0.0, le=1.0, description="精排综合分数")


class PaperImportance(str, Enum):
    """论文重要等级。"""

    CORE = "CORE"
    IMPORTANT = "IMPORTANT"
    NORMAL = "NORMAL"


class PaperAvailability(str, Enum):
    """论文全文资源的当前可用状态。"""

    FULL_AVAILABLE = "FULL_AVAILABLE"
    METADATA_ONLY = "METADATA_ONLY"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    USER_UPLOAD_REQUIRED = "USER_UPLOAD_REQUIRED"


class PaperResource(SchemaModel):
    """论文元数据及其本地全文、文本和向量处理状态。"""

    paper: PaperMetadata
    availability: PaperAvailability = PaperAvailability.METADATA_ONLY
    file_path: Optional[str] = Field(default=None, description="本地全文文件路径")
    text_path: Optional[str] = Field(default=None, description="解析后文本文件路径")
    embedding_status: str = Field(default="pending", min_length=1, description="向量生成状态")


class UploadRequest(SchemaModel):
    """需要用户人工上传全文的提醒任务。"""

    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    priority: PaperImportance = PaperImportance.NORMAL
