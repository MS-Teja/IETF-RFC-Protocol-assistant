"""Structured logging utility.

Logs retrieval details (chunk IDs, similarity scores) for every query —
required by the assessment's evaluation/logging architecture component.
"""

import logging
import json
import sys
from datetime import datetime, timezone

from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Configure application-wide structured logging."""
    logger = logging.getLogger("protocol_assistant")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()


def log_retrieval(
    query: str,
    rewritten_query: str,
    chunks: list[dict],
    threshold: float,
    passed_threshold: bool,
) -> None:
    """Log retrieval details for audit trail.

    A compliance/legal tool should be able to produce its own audit trail.
    Logs chunk IDs, similarity scores, and threshold decisions.
    """
    retrieval_record = {
        "event": "retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_query": query,
        "rewritten_query": rewritten_query,
        "threshold": threshold,
        "passed_threshold": passed_threshold,
        "num_chunks_retrieved": len(chunks),
        "chunks": [
            {
                "doc_type": c.get("metadata", {}).get("doc_type", "unknown"),
                "section": c.get("metadata", {}).get("section_id", "unknown"),
                "score": round(c.get("score", 0.0), 4),
            }
            for c in chunks
        ],
    }
    logger.info("Retrieval: %s", json.dumps(retrieval_record))


def log_generation(
    model_used: str,
    prompt_length: int,
    response_length: int,
    fallback_used: bool,
) -> None:
    """Log generation details."""
    generation_record = {
        "event": "generation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": model_used,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "fallback_used": fallback_used,
    }
    logger.info("Generation: %s", json.dumps(generation_record))
