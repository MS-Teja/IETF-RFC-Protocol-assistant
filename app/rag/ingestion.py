"""Document ingestion for IETF RFCs.

Loads plaintext RFC files from disk, preprocesses them, chunks them,
and ingests them into the vector store.
"""

import hashlib
from pathlib import Path

from app.core.config import settings
from app.rag.preprocessing import preprocess_text
from app.rag.chunking import Chunk, rfc_chunker
from app.utils.logging import logger


class RFCLoader:
    """Loader for IETF RFC documents from local text files."""

    def load(self, rfc_number: int) -> list[Chunk]:
        """Load an RFC text file and return a list of chunks."""
        cache_path = Path(settings.data_dir) / "rfcs" / f"rfc{rfc_number}.txt"
        
        if not cache_path.exists():
            raise FileNotFoundError(f"RFC not found in cache: {cache_path}")
            
        logger.info("Extracting text from RFC: %s", cache_path)
        raw_text = cache_path.read_text(encoding="utf-8", errors="ignore")
        
        doc_type = f"rfc{rfc_number}"
        cleaned_text = preprocess_text(raw_text, doc_type=doc_type)
        chunks = rfc_chunker(cleaned_text, doc_type=doc_type)
        
        logger.info("RFC %s loaded: %d chunks", rfc_number, len(chunks))
        return chunks


def _generate_chunk_id(chunk: Chunk, index: int) -> str:
    """Generate a deterministic unique ID for a chunk.

    Uses doc_type + section_id + content hash for deduplication.
    """
    content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:8]
    return f"{chunk.doc_type}_{chunk.section_id}_{content_hash}_{index}"


def ingest_documents(
    data_dir: str,
    embedding_service,
    vector_store,
    force_reingest: bool = False,
) -> dict:
    """Full ingestion pipeline for RFCs.

    Args:
        data_dir: Directory containing the data (not heavily used since we rely on config.rfc_numbers).
        embedding_service: EmbeddingService instance for generating embeddings.
        vector_store: VectorStore instance for storing chunks.
        force_reingest: If True, clear existing data and re-ingest.

    Returns:
        Dict with ingestion stats.
    """
    if force_reingest:
        logger.info("Force re-ingestion: clearing existing data")
        vector_store.clear()

    # Check if already ingested
    if not force_reingest and vector_store.count() > 0:
        logger.info(
            "Vector store already has %d documents. Use force_reingest=True to re-ingest.",
            vector_store.count(),
        )
        return {
            "status": "skipped",
            "documents_processed": 0,
            "chunks_created": 0,
            "message": f"Already ingested ({vector_store.count()} chunks in store)",
        }

    all_chunks: list[Chunk] = []
    docs_processed = 0

    loader = RFCLoader()
    
    for rfc_number in settings.rfc_numbers:
        try:
            chunks = loader.load(rfc_number)
            all_chunks.extend(chunks)
            docs_processed += 1
        except Exception as e:
            logger.error("Failed to load RFC %s: %s", rfc_number, e)

    if not all_chunks:
        return {
            "status": "error",
            "documents_processed": 0,
            "chunks_created": 0,
            "message": "No RFC documents found or processed successfully",
        }

    # Generate embeddings
    logger.info("Generating embeddings for %d chunks...", len(all_chunks))
    texts = [chunk.content for chunk in all_chunks]
    embeddings = embedding_service.embed(texts)

    # Store in vector store
    ids = [_generate_chunk_id(chunk, i) for i, chunk in enumerate(all_chunks)]
    documents = texts
    metadatas = [chunk.metadata for chunk in all_chunks]

    vector_store.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        "Ingestion complete: %d RFCs → %d chunks stored",
        docs_processed,
        len(all_chunks),
    )

    return {
        "status": "success",
        "documents_processed": docs_processed,
        "chunks_created": len(all_chunks),
        "message": f"Successfully ingested {docs_processed} RFCs into {len(all_chunks)} chunks",
    }
