"""Embedding service interface and implementations.

Defines the abstract interface for embedding text into vectors,
with a concrete implementation using sentence-transformers.
"""

from abc import ABC, abstractmethod

import os

# Suppress HuggingFace Hub network checks (we already have the model cached locally)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

from app.utils.logging import logger


class EmbeddingService(ABC):
    """Abstract interface for text embedding.

    To swap the embedding model (assessment Section 8: "Replace the embedding model"),
    create a new subclass and register it in dependencies.py.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            A single embedding vector.
        """
        ...


class SentenceTransformerEmbedder(EmbeddingService):
    """Embedding service using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default — 384-dimensional embeddings,
    512-token context limit. Chunks must stay under ~400 words.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()
