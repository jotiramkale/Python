"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import build_router
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService

BASE_DIR = Path(__file__).resolve().parent.parent
document_service = DocumentService()
chat_service = ChatService(document_service)
app = FastAPI(title="PDF Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(build_router(document_service, chat_service))
app.mount("/static/css", StaticFiles(directory=BASE_DIR / "frontend" / "css"), name="styles")
app.mount("/static/js", StaticFiles(directory=BASE_DIR / "frontend" / "js"), name="scripts")
app.mount("/", StaticFiles(directory=BASE_DIR / "frontend" / "html", html=True), name="frontend")
