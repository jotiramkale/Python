"""Pydantic request and response contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    name: str
    pages: int
    chunks: int
    size_bytes: int


class SourceChunk(BaseModel):
    text: str
    page: int | str
    score: float


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[dict[str, object]]
