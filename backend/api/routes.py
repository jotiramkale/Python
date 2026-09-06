"""REST API routes for documents and chat."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.schemas import ChatRequest, ChatResponse, DocumentSummary

router = APIRouter(prefix="/api")


def build_router(document_service, chat_service) -> APIRouter:
    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/documents", response_model=list[DocumentSummary])
    def list_documents() -> list[DocumentSummary]:
        return document_service.list_documents()

    @api.post("/documents", response_model=DocumentSummary)
    async def upload_document(file: UploadFile = File(...)) -> DocumentSummary:
        try:
            return document_service.ingest(await file.read(), file.filename or "document.pdf")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, str]:
        try:
            document_service.delete(document_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"status": "deleted"}

    @api.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            return chat_service.answer(
                request.document_id, request.question, request.session_id
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.delete("/sessions/{session_id}")
    def clear_session(session_id: str) -> dict[str, str]:
        chat_service.clear(session_id)
        return {"status": "cleared"}

    return api
