"""POST /chat — Full RAG pipeline endpoint."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_rag_pipeline
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import RAGPipeline
from app.utils.logging import logger
from app.services.llm_service import ClaudeLLM, GeminiLLM, LLMError

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """Process a chat query through the full RAG pipeline.

    Accepts a user query and optional conversation history.
    Returns an AI-generated response with source citations.

    Status codes:
    - 200: Successful response (including "not covered" responses)
    - 400: Invalid request (handled by Pydantic validation)
    - 502: Both LLM providers failed
    """
    logger.info("Chat request: '%s' (history: %d messages, model: %s)", request.query, len(request.history), request.model)

    try:
        override_llm = None
        if request.model:
            if request.model.startswith("claude"):
                override_llm = ClaudeLLM(model_name=request.model)
            elif request.model.startswith("gemini"):
                override_llm = GeminiLLM(model_name=request.model)
            else:
                logger.warning("Unknown model override: %s", request.model)

        response = pipeline.query(
            user_query=request.query,
            history=request.history,
            override_llm=override_llm,
        )
        return response

    except LLMError as e:
        # Parse the error string and return a clean, structured error
        error_str = str(e)
        if "503" in error_str or "UNAVAILABLE" in error_str:
            detail = "The selected model is currently experiencing high demand. Try again shortly or switch to a different model."
        elif "401" in error_str or "authentication" in error_str.lower():
            detail = "API key is invalid or missing for the selected model."
        elif "429" in error_str or "rate" in error_str.lower():
            detail = "Rate limit exceeded. Please wait a moment before retrying."
        elif "timeout" in error_str.lower():
            detail = "The model took too long to respond. Try again or switch models."
        else:
            detail = "The AI model failed to generate a response. Try again or switch models."

        logger.error("LLM error returned to client: %s (raw: %s)", detail, error_str)
        raise HTTPException(status_code=502, detail=detail)

    except Exception as e:
        logger.error("Unexpected chat pipeline error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again.",
        )
