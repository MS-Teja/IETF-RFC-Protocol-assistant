"""POST /ingest — Document ingestion endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.dependencies import get_embedding_service, get_vector_store
from app.models.schemas import IngestRequest, IngestResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.rag.ingestion import ingest_documents
from app.utils.logging import logger

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest = IngestRequest(),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    """Trigger document ingestion.

    Loads PDFs from the data directory, preprocesses, chunks, embeds,
    and stores them in the vector store.

    Status codes:
    - 200: Successful ingestion or already ingested
    - 500: Ingestion failed
    """
    data_dir = settings.data_dir

    logger.info("Ingest request: data_dir=%s, force=%s", data_dir, request.force_reingest)

    try:
        result = ingest_documents(
            data_dir=data_dir,
            embedding_service=embedding_service,
            vector_store=vector_store,
            force_reingest=request.force_reingest,
        )

        return IngestResponse(
            status=result["status"],
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            message=result["message"],
        )
    except Exception as e:
        logger.error("Ingestion error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {e}",
        )
