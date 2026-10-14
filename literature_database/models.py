"""论文资源数据库的 SQLAlchemy 2.0 ORM 模型。"""

from typing import List, Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .schema import PaperAvailability, PaperImportance


class Base(DeclarativeBase):
    """所有 literature_database ORM 模型的声明式基类。"""


class PaperModel(Base):
    """论文元数据表。"""

    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    authors: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    abstract: Mapped[str] = mapped_column(Text, nullable=False, default="")
    doi: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True, index=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True, index=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    venue: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rank_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)
    availability: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=PaperAvailability.METADATA_ONLY.value,
        index=True,
    )
    embedding_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        index=True,
    )

    files: Mapped[List["PaperFileModel"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    download_tasks: Mapped[List["DownloadTaskModel"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    upload_requests: Mapped[List["UploadQueueModel"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"PaperModel(paper_id={self.paper_id!r}, title={self.title!r})"


class PaperFileModel(Base):
    """论文全文或解析文本的本地文件记录。"""

    __tablename__ = "paper_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="AVAILABLE", index=True)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, default="pdf")

    paper: Mapped[PaperModel] = relationship(back_populates="files")

    __table_args__ = (
        Index("ix_paper_files_paper_type", "paper_id", "file_type"),
    )


class DownloadTaskModel(Base):
    """论文全文下载任务及失败重试记录。"""

    __tablename__ = "download_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    paper: Mapped[PaperModel] = relationship(back_populates="download_tasks")


class UploadQueueModel(Base):
    """缺少核心论文全文时的人工上传队列。"""

    __tablename__ = "upload_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("papers.paper_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    importance: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=PaperImportance.NORMAL.value,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING", index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    paper: Mapped[PaperModel] = relationship(back_populates="upload_requests")
