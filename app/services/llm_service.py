"""LLM service interface and implementations.

Defines the abstract interface for LLM-based text generation,
with concrete implementations for Claude (Anthropic) and Gemini (Google).
Includes a fallback wrapper for resilience.
"""

from abc import ABC, abstractmethod

from anthropic import Anthropic, APIError, APITimeoutError

from google import genai
from google.genai import types

from app.core.config import settings
from app.utils.logging import logger


class LLMService(ABC):
    """Abstract interface for LLM generation.

    To swap the LLM (assessment Section 8: "Replace the language model"),
    create a new subclass and register it in dependencies.py.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        ...

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts.

        Args:
            system_prompt: The system instruction for the model.
            user_prompt: The user's query with retrieved context.

        Returns:
            The generated text response.

        Raises:
            LLMError: If generation fails.
        """
        ...


class LLMError(Exception):
    """Raised when LLM generation fails."""
    pass


class ClaudeLLM(LLMService):
    """Claude LLM via Anthropic API (v1.x SDK).

    Primary choice for compliance/legal domain because Claude tends to
    hold grounding instructions more reliably — important for a chatbot
    that must refuse to answer beyond what's retrieved.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for ClaudeLLM")
        self.client = Anthropic(api_key=self._api_key)
        self._model_name = model_name or "claude-sonnet-5"
        logger.info("ClaudeLLM initialized with model: %s", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Claude via Anthropic API."""
        try:
            message = self.client.messages.create(
                model=self._model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            return message.content[0].text
        except (APIError, APITimeoutError) as e:
            logger.error("Claude API error: %s", e)
            raise LLMError(f"Claude API error: {e}") from e
        except Exception as e:
            logger.error("Unexpected ClaudeLLM error: %s", e)
            raise LLMError(f"ClaudeLLM error: {e}") from e


class GeminiLLM(LLMService):
    """Gemini LLM via Google GenAI SDK (google-genai, NOT deprecated google-generativeai).

    Secondary/fallback provider. Uses client.models.generate_content()
    with types.GenerateContentConfig for system instructions.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self._api_key = api_key or settings.google_api_key
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is required for GeminiLLM")
        self.client = genai.Client(api_key=self._api_key)
        self._model_name = model_name or "gemini-3.7-flash"
        logger.info("GeminiLLM initialized with model: %s", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Gemini via Google GenAI SDK."""
        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2048,
                    temperature=0.3,
                ),
            )
            return response.text
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise LLMError(f"Gemini API error: {e}") from e


def generate_with_fallback(
    primary: LLMService,
    fallback: LLMService,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str, bool]:
    """Generate with automatic fallback to secondary provider.

    If the primary LLM fails or times out, retries once against
    the fallback. This turns the dual-provider setup into actual
    resilience, not just a config toggle.

    Args:
        primary: The primary LLM service.
        fallback: The fallback LLM service.
        system_prompt: System instruction.
        user_prompt: User query with context.

    Returns:
        Tuple of (response_text, model_used, fallback_was_used).

    Raises:
        LLMError: If both primary and fallback fail.
    """
    try:
        response = primary.generate(system_prompt, user_prompt)
        return response, primary.model_name, False
    except LLMError as primary_error:
        logger.warning(
            "Primary LLM (%s) failed: %s. Trying fallback (%s)...",
            primary.model_name,
            primary_error,
            fallback.model_name,
        )
        try:
            response = fallback.generate(system_prompt, user_prompt)
            return response, fallback.model_name, True
        except LLMError as fallback_error:
            logger.error(
                "Both LLMs failed. Primary: %s, Fallback: %s",
                primary_error,
                fallback_error,
            )
            raise LLMError(
                f"Both LLM providers failed. Primary ({primary.model_name}): "
                f"{primary_error}. Fallback ({fallback.model_name}): {fallback_error}"
            ) from fallback_error
