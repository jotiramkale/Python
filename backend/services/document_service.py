"""PDF ingestion and document catalogue service."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.config import get_settings
from backend.database.chroma_store import ChromaStore
from backend.rag.embeddings import get_embedding_model
from backend.models.schemas import DocumentSummary


class DocumentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaStore()
        self.catalog_path = Path(self.settings.chroma_path).parent / "documents.json"
        self.documents: dict[str, DocumentSummary] = self._load_catalog()

    def _load_catalog(self) -> dict[str, DocumentSummary]:
        if not self.catalog_path.exists():
            return {}
        try:
            raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            return {key: DocumentSummary(**value) for key, value in raw.items()}
        except (OSError, ValueError):
            return {}

    def _save_catalog(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog_path.write_text(
            json.dumps({key: value.model_dump() for key, value in self.documents.items()}),
            encoding="utf-8",
        )

    def ingest(self, content: bytes, file_name: str) -> DocumentSummary:
        max_size = self.settings.max_pdf_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise ValueError(f"PDF must be smaller than {self.settings.max_pdf_size_mb} MB.")
        if not file_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        reader = PdfReader(BytesIO(content))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks: list[str] = []
        pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            for chunk in splitter.split_text(page.extract_text() or ""):
                chunks.append(chunk)
                pages.append(page_number)
        if not chunks:
            raise ValueError("No selectable text was found in this PDF.")

        document_id = uuid4().hex
        model = get_embedding_model(self.settings.embedding_model)
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        self.store.add_document(document_id, chunks, embeddings, pages)
        summary = DocumentSummary(
            id=document_id,
            name=file_name,
            pages=len(reader.pages),
            chunks=len(chunks),
            size_bytes=len(content),
        )
        self.documents[document_id] = summary
        self._save_catalog()
        return summary

    def list_documents(self) -> list[DocumentSummary]:
        return list(self.documents.values())

    def get(self, document_id: str) -> DocumentSummary:
        if document_id not in self.documents:
            raise KeyError("Document not found.")
        return self.documents[document_id]

    def delete(self, document_id: str) -> None:
        self.get(document_id)
        self.store.delete(document_id)
        del self.documents[document_id]
        self._save_catalog()
