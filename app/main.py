"""FastAPI application entry point.

Mounts all API routers and serves the static frontend.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, ingest, retrieve, health
from app.utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup and shutdown logic."""
    logger.info("Protocol Assistant starting up...")
    try:
        from app.core.dependencies import get_vector_store, get_embedding_service
        from app.core.config import settings
        from app.rag.ingestion import ingest_documents

        vector_store = get_vector_store()
        if vector_store.count() == 0:
            logger.info("ChromaDB collection is empty. Performing automatic initial ingestion...")
            embedding_service = get_embedding_service()
            ingest_documents(
                data_dir=settings.data_dir,
                embedding_service=embedding_service,
                vector_store=vector_store,
                force_reingest=False,
            )
            logger.info("Initial ingestion complete.")
    except Exception as e:
        logger.warning(f"Startup ingestion check skipped or failed: {e}")
    yield
    logger.info("Protocol Assistant shutting down...")


app = FastAPI(
    title="Protocol Assistant",
    description=(
        "AI-powered RAG chatbot for Web & API protocols. "
        "Covers core IETF RFCs for HTTP, TLS, OAuth, JWT, and WebSockets."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(retrieve.router, tags=["Retrieval"])
app.include_router(health.router, tags=["Health"])

# Serve static frontend
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chat frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "name": "Protocol Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
