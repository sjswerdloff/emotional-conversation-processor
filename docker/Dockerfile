# Multi-stage Dockerfile for Emotional Conversation Processor using uv

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    # hadolint ignore=DL3013
    && pip install --no-cache-dir uv

# Create app directory
WORKDIR /app

# Copy requirements and pyproject.toml for dependency installation
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies with uv (much faster than pip)
RUN uv pip install --system -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ECP_ENV=production

# Install runtime dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # hadolint ignore=DL3013
    && pip install --no-cache-dir uv

# Create non-root user
RUN groupadd -r ecp && useradd -r -g ecp ecp

# Create app directory and set ownership
WORKDIR /app
RUN chown -R ecp:ecp /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=ecp:ecp . .

# Create necessary directories
RUN mkdir -p /app/data/raw/conversations /app/data/processed/embeddings /app/data/exports \
    /app/logs/application /app/logs/tests \
    /app/models \
    && chown -R ecp:ecp /app

# Switch to non-root user
USER ecp

# Download required models during build (optional, can be done at runtime)
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" || true
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')" || true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || true

# Expose ports (if running API)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/health_check.py --qdrant-host ${QDRANT_HOST:-localhost} || exit 1

# Default command
CMD ["python", "scripts/process_conversation.py", "--help"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies with uv
# hadolint ignore=DL3013
RUN uv pip install --system pytest pytest-cov pytest-mock jupyter ipython ruff mypy

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER ecp

# Override default command for development
CMD ["bash"]
