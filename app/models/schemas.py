"""Pydantic models for API request/response schemas.

All data structures used in API communication are defined here.
Pydantic v2 is used for validation and serialization.
"""

from pydantic import BaseModel, Field


# --- Shared Models ---

class Document(BaseModel):
    """A processed document chunk with metadata."""
    content: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from vector store retrieval."""
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float = 0.0
    label: str = ""  # e.g., "[RFC 9110, §15.5.1 Bad Request]"


# --- Chat ---

class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    model: str | None = Field(default=None, description="Optional model override")


class Citation(BaseModel):
    """A source citation for a response."""
    document: str  # e.g., "RFC 9110"
    section: str  # e.g., "15.5.1"
    content_preview: str  # first ~100 chars of the chunk
    score: float


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model_used: str = ""
    query_used: str = ""  # The (possibly rewritten) query that was actually embedded


# --- Ingest ---

class IngestRequest(BaseModel):
    """Request body for POST /ingest."""
    file_paths: list[str] = Field(default_factory=list)
    force_reingest: bool = False


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    status: str
    documents_processed: int
    chunks_created: int
    message: str = ""


# --- Retrieve ---

class RetrieveRequest(BaseModel):
    """Query parameters for GET /retrieve."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveResponse(BaseModel):
    """Response body for GET /retrieve."""
    query: str
    results: list[RetrievedChunk]
    total_results: int


# --- Health ---

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str  # "healthy" or "unhealthy"
    chroma_connected: bool
    chroma_document_count: int
    llm_provider: str
    llm_key_present: bool
    embedding_model: str
    details: dict = Field(default_factory=dict)
