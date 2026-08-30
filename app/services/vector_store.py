"""Vector store interface and implementations.

Defines the abstract interface for vector storage and retrieval,
with a concrete implementation using ChromaDB's PersistentClient.
"""

from abc import ABC, abstractmethod

import chromadb

from app.utils.logging import logger


class VectorStore(ABC):
    """Abstract interface for vector storage and retrieval.

    To swap the vector DB (assessment Section 8: "Replace the vector database"),
    create a new subclass with the same add()/query() contract
    and register it in dependencies.py.
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents with their embeddings and metadata to the store."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Query the store for similar documents.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: content, metadata, score.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the store is healthy and accessible."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        ...


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store using PersistentClient (embedded mode).

    Runs in-process — no separate server needed. Data persisted to disk.
    Uses chromadb.PersistentClient (NOT the deprecated Client(Settings(...)) API).
    """

    COLLECTION_NAME = "protocol_documents"

    def __init__(self, path: str = "./chroma_data") -> None:
        logger.info("Initializing ChromaDB PersistentClient at: %s", path)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (documents: %d)",
            self.COLLECTION_NAME,
            self.collection.count(),
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents to the ChromaDB collection."""
        if not ids:
            return

        # ChromaDB handles batching internally, but we chunk to avoid
        # memory issues with very large document sets
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info("Added %d documents to ChromaDB", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Query ChromaDB for similar documents.

        Returns results sorted by similarity (highest first).
        Scores are cosine similarity values (0-1, higher = more similar).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        chunks = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            # cosine_similarity = 1 - cosine_distance
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            chunks.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": similarity,
            })

        return chunks

    def count(self) -> int:
        """Return document count in the collection."""
        return self.collection.count()

    def health_check(self) -> bool:
        """Verify ChromaDB is accessible and has documents."""
        try:
            count = self.collection.count()
            return count > 0
        except Exception as e:
            logger.error("ChromaDB health check failed: %s", e)
            return False

    def clear(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection cleared")
