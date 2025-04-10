# GPU version of Coqui TTS
# NOTE: This image is ONLY compatible with x86_64/amd64 architecture and requires NVIDIA GPU support.

FROM ghcr.io/coqui-ai/tts:v0.22.0

# Set up working directory
WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
# Install dependencies but exclude TTS as it's already in the base image
RUN grep -v "TTS==" requirements.txt > requirements_no_tts.txt && \
    pip install --no-cache-dir -r requirements_no_tts.txt && \
    # Fix the TTS package by reinstalling it properly with pip
    pip uninstall -y TTS && \
    pip install --no-cache-dir TTS==0.22.0 && \
    # Verify TTS is installed correctly
    python3 -c "import TTS; print(f'TTS {TTS.__version__} installed successfully')"

# Create model and cache directories
RUN mkdir -p /app/models /app/cache && \
    chmod -R 777 /app/models /app/cache

# Copy application code and models
COPY src/ /app/src/
COPY models/ /app/models/

# Create the entrypoint script
RUN PYTHON_BIN=$(which python3) && \
    echo '#!/bin/bash' > /app/entrypoint.sh && \
    echo 'set -e' >> /app/entrypoint.sh && \
    echo 'echo "Starting Vox-Raga TTS service..."' >> /app/entrypoint.sh && \
    echo "# Save the original PATH but remove /usr/local/bin to avoid TTS command" >> /app/entrypoint.sh && \
    echo 'export CLEAN_PATH=$(echo $PATH | sed "s|/usr/local/bin:||g")' >> /app/entrypoint.sh && \
    echo "# Temporarily use the clean PATH" >> /app/entrypoint.sh && \
    echo 'export PATH="$CLEAN_PATH"' >> /app/entrypoint.sh && \
    echo 'echo "Modified PATH: $PATH"' >> /app/entrypoint.sh && \
    echo "# Use the direct path to the Python binary" >> /app/entrypoint.sh && \
    echo "PYTHON_BIN=$PYTHON_BIN" >> /app/entrypoint.sh && \
    echo 'echo "Using Python binary: $PYTHON_BIN"' >> /app/entrypoint.sh && \
    echo '# List model directory contents for debugging' >> /app/entrypoint.sh && \
    echo 'echo "Available models in /app/models:"' >> /app/entrypoint.sh && \
    echo 'find /app/models -type f | sort' >> /app/entrypoint.sh && \
    echo "# Start the application with the clean PATH" >> /app/entrypoint.sh && \
    echo 'exec env PATH="$CLEAN_PATH" "$PYTHON_BIN" -m uvicorn src.main:app --host 0.0.0.0 --port ${SERVER_PORT:-8888}' >> /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Environment variables - add /root to PYTHONPATH to fix TTS import issues
ENV PYTHONPATH=/app:/root \
    SERVER_PORT=8888 \
    MODEL_DOWNLOAD_ROOT=/app/models \
    SERVER_CACHE_DIR=/app/cache \
    MODEL_NAME=tts_models/multilingual/multi-dataset/xtts_v2 \
    MODEL_DEVICE=cuda \
    SERVER_LOG_LEVEL=info

# Set Python optimization flags for better performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Use the custom entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 