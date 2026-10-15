"""Stable HTTP request and response contracts for the M2 application."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    max_papers: int = Field(default=3, ge=1, le=20)
    max_search_rounds: int = Field(default=1, ge=1, le=10)
    download_papers: bool = True
    enable_rag: bool = True
    summarize: bool = True
    extract_knowledge: bool = True
    question: str | None = None
    use_query_llm: bool = True
    enable_citation_tracking: bool = True
    publication_date_to: str | None = None
    download_engine: Literal["academic_oa", "auto", "arxiv", "openalex", "crossref", "scholar"] = "academic_oa"
    download_timeout: int = Field(default=30, ge=10, le=300)
    retry_failed_downloads: bool = False

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("query must not be empty")
        return value

    @model_validator(mode="after")
    def check_dependencies(self) -> "ResearchOptions":
        if self.enable_rag and not self.download_papers:
            raise ValueError("enable_rag requires download_papers")
        if not self.enable_rag and (self.summarize or self.extract_knowledge or self.question):
            raise ValueError("summary, extraction and question require enable_rag")
        if self.publication_date_to:
            from datetime import date
            date.fromisoformat(self.publication_date_to)
        return self


JobStatus = Literal["queued", "running", "cancelling", "completed", "failed", "cancelled", "interrupted"]


class JobSnapshot(BaseModel):
    job_id: str
    query: str
    status: JobStatus
    stage: str | None = None
    progress: int = Field(ge=0, le=100)
    result: dict[str, Any] | None = None
    error: str | None = None
    retryable: bool = False
    logs: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    elapsed_seconds: float = 0.0


class JobListItem(BaseModel):
    job_id: str
    query: str
    status: JobStatus
    stage: str | None = None
    progress: int = Field(ge=0, le=100)
    error: str | None = None
    retryable: bool = False
    created_at: datetime
    updated_at: datetime
    elapsed_seconds: float = 0.0


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str = Field(min_length=1, max_length=255)
    question: str = Field(min_length=1, max_length=2000)


class QuestionResponse(BaseModel):
    answer_id: str
    paper_id: str
    answer: dict[str, Any]


class PaperSummary(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    availability: str
    embedding_status: str
