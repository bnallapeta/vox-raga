# Build stage for dependencies
FROM --platform=$BUILDPLATFORM python:3.11-slim as deps

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    pkg-config \
    curl \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip --version \
    && gcc --version

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only pure Python packages that don't have binary components
RUN pip3 install --no-cache-dir setuptools wheel

# Builder stage for target architecture
FROM --platform=$TARGETPLATFORM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    cmake \
    pkg-config \
    curl \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip --version \
    && gcc --version

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install CPU-only PyTorch first (faster to build)
RUN pip3 install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies directly on the target architecture
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage
FROM --platform=$TARGETPLATFORM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/tts_models && \
    chown -R appuser:appuser /app /tmp/tts_models

# Copy application code
COPY --chown=appuser:appuser . /app/

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MODEL_DOWNLOAD_ROOT=/tmp/tts_models \
    LOG_LEVEL=INFO

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 