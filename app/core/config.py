"""Application configuration using pydantic-settings.

All configuration is driven by environment variables, with sensible defaults
for local development. This is the single source of truth for all config values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Provider
    llm_provider: str = "claude"  # "claude" or "gemini"
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # ChromaDB
    chroma_path: str = "./chroma_data"

    # Retrieval
    similarity_threshold: float = 0.3
    retrieval_top_k: int = 5

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Logging
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000

    # Data
    data_dir: str = "./data"
    rfc_numbers: list[int] = [9110, 9111, 9112, 9113, 9114, 8446, 6749, 7519, 6455]


settings = Settings()
