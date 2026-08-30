"""RAG pipeline — orchestrates all 9 stages.

This is the core orchestration module that ties together:
1. Query processing (contextualize follow-ups)
2. Embedding (vectorize the query)
3. Retrieval (similarity search + threshold gate)
4. Context construction (label injection)
5. Generation (LLM with fallback)

Ingestion stages (1-5 of the full pipeline) are handled in ingestion.py.
This module handles the query-time stages (6-9).
"""

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService, LLMError, generate_with_fallback
from app.rag.query_processing import contextualize_query
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models.schemas import ChatResponse, Citation, ChatMessage
from app.utils.logging import logger, log_retrieval, log_generation


class RAGPipeline:
    """Orchestrates the query-time RAG pipeline.

    Receives pre-wired services via dependency injection — never imports
    concrete classes directly.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        primary_llm: LLMService,
        fallback_llm: LLMService | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm

    def query(
        self,
        user_query: str,
        history: list[ChatMessage] | None = None,
        top_k: int | None = None,
        override_llm: LLMService | None = None,
    ) -> ChatResponse:
        """Execute the full query-time RAG pipeline.

        Stages:
        6. Query processing — contextualize follow-ups
        7. Retrieval — similarity search + threshold gate
        8. Context construction — label injection
        9. Generation — LLM with fallback

        Args:
            user_query: The user's question.
            history: Conversation history for query contextualization.
            top_k: Override for retrieval_top_k setting.

        Returns:
            ChatResponse with answer, citations, and metadata.
        """
        history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in (history or [])
        ]

        # Stage 6: Query processing
        rewritten_query = contextualize_query(user_query, history_dicts)

        # Stage 7: Retrieval
        query_embedding = self.embedding_service.embed_query(rewritten_query)
        k = top_k or settings.retrieval_top_k
        raw_chunks = self.vector_store.query(query_embedding, top_k=k)

        # Threshold gate — filter chunks below similarity threshold
        threshold = settings.similarity_threshold
        filtered_chunks = [
            chunk for chunk in raw_chunks
            if chunk.get("score", 0.0) >= threshold
        ]

        passed_threshold = len(filtered_chunks) > 0

        # Log retrieval for audit trail
        log_retrieval(
            query=user_query,
            rewritten_query=rewritten_query,
            chunks=raw_chunks,
            threshold=threshold,
            passed_threshold=passed_threshold,
        )

        # If nothing clears the threshold, respond without forcing the LLM
        if not passed_threshold:
            logger.info(
                "No chunks passed threshold (%.2f). Returning 'not covered' response.",
                threshold,
            )
            return ChatResponse(
                answer=(
                    "I cannot find relevant information about this in the ingested documents. "
                    "This topic may not be covered by the retrieved RFCs, or the relevant "
                    "sections may not have been indexed."
                ),
                citations=[],
                model_used="none (threshold not met)",
                query_used=rewritten_query,
            )

        # Stage 8: Context construction (with label injection + conversation history)
        user_prompt = build_user_prompt(user_query, filtered_chunks, history=history_dicts)

        # Stage 9: Generation (with fallback or override)
        try:
            if override_llm:
                response_text = override_llm.generate(SYSTEM_PROMPT, user_prompt)
                model_used = override_llm.model_name
                fallback_used = False
            elif self.fallback_llm:
                response_text, model_used, fallback_used = generate_with_fallback(
                    primary=self.primary_llm,
                    fallback=self.fallback_llm,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            else:
                response_text = self.primary_llm.generate(SYSTEM_PROMPT, user_prompt)
                model_used = self.primary_llm.model_name
                fallback_used = False

            # Log generation
            log_generation(
                model_used=model_used,
                prompt_length=len(user_prompt),
                response_length=len(response_text),
                fallback_used=fallback_used,
            )

        except LLMError as e:
            logger.error("All LLM providers failed: %s", e)
            raise  # Let the route handler convert this to a proper HTTP error

        # Build response with citations
        return ChatResponse(
            answer=response_text,
            citations=_build_citations(filtered_chunks),
            model_used=model_used,
            query_used=rewritten_query,
        )


def _build_citations(chunks: list[dict]) -> list[Citation]:
    """Build citation objects from retrieved chunks."""
    citations = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        # For RFCs, the doc_type is something like "rfc9110". Let's capitalize it.
        doc_type = metadata.get("doc_type", "unknown")
        if doc_type.startswith("rfc"):
            document = doc_type.upper()
        else:
            document = doc_type

        # Section/rule/clause identifier
        section = metadata.get("section_id", "Unknown section")

        # Content preview (first ~100 chars)
        content = chunk.get("content", "")
        preview = content[:100] + "..." if len(content) > 100 else content

        citations.append(Citation(
            document=document,
            section=section,
            content_preview=preview,
            score=round(chunk.get("score", 0.0), 4),
        ))

    return citations
