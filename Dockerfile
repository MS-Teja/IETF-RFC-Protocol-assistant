# Stage 1: Build environment
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
# Install CPU-only PyTorch first to prevent downloading 1GB+ of NVIDIA CUDA binaries
RUN pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.12-slim

# Create a non-root user with a home directory
RUN groupadd -r appuser && useradd -r -m -g appuser appuser

WORKDIR /app

# Copy python dependencies from builder with proper ownership
COPY --chown=appuser:appuser --from=builder /root/.local /home/appuser/.local

# Ensure local bin is on PATH and set env vars
ENV PATH=/home/appuser/.local/bin:$PATH
ENV HF_HUB_OFFLINE=1

# Copy application code and set ownership
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser static/ ./static/

# We DO NOT copy .env.example to .env in the image.
# Production containers should receive config via environment variables
# or mounted volumes, not baked-in files.

# Set explicit HuggingFace cache directory and create it with proper permissions
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache && chown appuser:appuser /app/hf_cache

# Create chroma directory with proper permissions
RUN mkdir -p chroma_data && chown appuser:appuser chroma_data
VOLUME /app/chroma_data

# Switch to non-root user
USER appuser

# Pre-download embedding model into cache and pre-ingest RFCs into ChromaDB
RUN HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -c "from app.core.config import settings; from app.core.dependencies import get_embedding_service, get_vector_store; from app.rag.ingestion import ingest_documents; ingest_documents(settings.data_dir, get_embedding_service(), get_vector_store())"

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
