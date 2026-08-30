"""GET /retrieve — Raw retrieval endpoint (no LLM).

Returns top-k chunks with similarity scores for debugging and evaluation.
The assessment doc (Section 9) explicitly lists "retrieval-related operations"
as a separate endpoint category from chat.
"""

from fastapi import APIRouter, Depends, Query

from app.core.config import settings
from app.core.dependencies import get_embedding_service, get_vector_store
from app.models.schemas import RetrievedChunk, RetrieveResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.logging import logger

router = APIRouter()


@router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    query: str = Query(..., min_length=1, max_length=2000, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> RetrieveResponse:
    """Retrieve raw chunks matching a query — no LLM generation.

    Useful for:
    - Debugging retrieval quality
    - Running the eval script
    - Inspecting what context the LLM would receive

    Status codes:
    - 200: Results returned (may be empty if no matches)
    - 400: Invalid query parameters
    """
    logger.info("Retrieve request: '%s' (top_k=%d)", query, top_k)

    query_embedding = embedding_service.embed_query(query)
    raw_chunks = vector_store.query(query_embedding, top_k=top_k)

    results = [
        RetrievedChunk(
            content=chunk.get("content", ""),
            metadata=chunk.get("metadata", {}),
            score=round(chunk.get("score", 0.0), 4),
            label=chunk.get("metadata", {}).get("label", ""),
        )
        for chunk in raw_chunks
    ]

    return RetrieveResponse(
        query=query,
        results=results,
        total_results=len(results),
    )
