# ================================================================================
# ⚡ FLAWLESS GENESIS ORCHESTRATOR - Dockerfile
# ================================================================================
# Multi-stage production build with security best practices
# ================================================================================

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# Stage 2: Production image
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user
RUN groupadd -r genesis && useradd -r -g genesis genesis

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY --chown=genesis:genesis . .

# Switch to non-root user
USER genesis

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start server
CMD ["python", "-m", "uvicorn", "flawless_api:app", "--host", "0.0.0.0", "--port", "8080"]
