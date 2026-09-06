"""Persistent ChromaDB document storage and retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from backend.config import get_settings


class ChromaStore:
    def __init__(self) -> None:
        settings = get_settings()
        Path(settings.chroma_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_path)

    def create_collection(self, document_id: str) -> Any:
        return self.client.get_or_create_collection(name=f"document_{document_id}")

    def add_document(
        self,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        pages: list[int],
    ) -> None:
        collection = self.create_collection(document_id)
        collection.add(
            ids=[str(index) for index in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"page": page} for page in pages],
        )

    def query(self, document_id: str, embedding: list[float], k: int) -> list[dict[str, Any]]:
        result = self.create_collection(document_id).query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            {
                "text": text,
                "page": metadata.get("page", "?"),
                "score": max(0.0, 1.0 - float(distance)),
            }
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

    def delete(self, document_id: str) -> None:
        try:
            self.client.delete_collection(f"document_{document_id}")
        except Exception:
            pass
