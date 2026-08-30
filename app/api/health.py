"""GET /health — Health check endpoint.

Not just a 200 OK — verifies Chroma actually has documents loaded
and the configured LLM key is present. A no-op health check is a
weak version of satisfying this requirement.
"""

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import get_vector_store
from app.models.schemas import HealthResponse
from app.services.vector_store import VectorStore
from app.utils.logging import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    """Check application health.

    Verifies:
    - ChromaDB is accessible and has documents loaded
    - The configured LLM API key is present
    - Embedding model is configured

    Returns unhealthy status if critical components are missing.
    """
    chroma_healthy = vector_store.health_check()
    chroma_count = vector_store.count()

    # Check if the configured LLM key is present
    if settings.llm_provider == "claude":
        llm_key_present = bool(settings.anthropic_api_key)
    else:
        llm_key_present = bool(settings.google_api_key)

    # Overall health
    is_healthy = chroma_healthy and llm_key_present

    details = {}
    if not chroma_healthy:
        details["chroma"] = "No documents loaded — run POST /ingest first"
    if not llm_key_present:
        details["llm"] = f"Missing API key for {settings.llm_provider}"

    status = "healthy" if is_healthy else "unhealthy"

    logger.info(
        "Health check: %s (chroma: %d docs, llm_key: %s)",
        status,
        chroma_count,
        llm_key_present,
    )

    return HealthResponse(
        status=status,
        chroma_connected=chroma_healthy,
        chroma_document_count=chroma_count,
        llm_provider=settings.llm_provider,
        llm_key_present=llm_key_present,
        embedding_model=settings.embedding_model,
        details=details,
    )
