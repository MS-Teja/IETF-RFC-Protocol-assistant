# Stage 1: Build environment
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime environment
FROM python:3.13-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

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

# Create chroma directory with proper permissions
RUN mkdir -p chroma_data && chown appuser:appuser chroma_data
VOLUME /app/chroma_data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
