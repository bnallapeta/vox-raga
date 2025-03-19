# Select the appropriate base image (GPU version)
FROM ghcr.io/coqui-ai/tts:v0.22.0

# Set up working directory
WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN adduser --disabled-password --gecos "" appuser

# Copy application code
COPY src/ /app/src/

# Create model and cache directories with proper permissions
RUN mkdir -p /app/models /app/cache && \
    chown -R appuser:appuser /app/models /app/cache

# Environment variables
ENV PYTHONPATH=/app \
    SERVER_PORT=8888 \
    MODEL_DOWNLOAD_ROOT=/app/models \
    SERVER_CACHE_DIR=/app/cache \
    MODEL_NAME=tts_models/en/vctk/vits \
    SERVER_LOG_LEVEL=info

# Set Python optimization flags for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run FastAPI application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8888"] 