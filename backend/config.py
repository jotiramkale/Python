"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    chroma_path: str
    max_pdf_size_mb: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    chunk_size = int(os.getenv("PDF_CHUNK_SIZE", "700"))
    chunk_overlap = int(os.getenv("PDF_CHUNK_OVERLAP", "120"))
    retrieval_k = int(os.getenv("PDF_RETRIEVAL_K", "5"))
    if chunk_size <= chunk_overlap:
        raise ValueError("PDF_CHUNK_SIZE must exceed PDF_CHUNK_OVERLAP")
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=retrieval_k,
        chroma_path=os.getenv("CHROMA_PATH", "./data/chroma"),
        max_pdf_size_mb=int(os.getenv("MAX_PDF_SIZE_MB", "25")),
    )
