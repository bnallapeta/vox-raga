# Vox-Raga: Text-to-Speech Service with Coqui TTS

This README contains instructions for building and deploying the vox-raga TTS service using Docker and Kubernetes.

## Overview

Vox-Raga is a Text-to-Speech (TTS) service that leverages Coqui TTS models to convert text to natural-sounding speech. The service is built with FastAPI and packaged as a Docker container for easy deployment.

## Multi-Architecture Docker Implementation

The project includes support for building Docker images for different architectures and use cases:

- **CPU Version**: Uses the `coqui-ai/tts-cpu` base image, suitable for environments without GPUs
- **GPU Version**: Uses the `coqui-ai/tts` base image, optimized for NVIDIA GPU acceleration

### Prerequisites

- Docker installed on your system with buildx support
- Git to clone this repository
- (Optional) Access to a container registry for pushing images

### Building the Docker Images

You can build the Docker images using the provided Makefile, which handles cross-platform builds:

```bash
# Initialize Docker buildx (only needed once)
make docker-buildx-init

# Build CPU Docker image (works on amd64 architecture)
make docker-buildx-cpu

# Build GPU Docker image (for amd64 architecture with NVIDIA GPUs)
make docker-buildx-gpu

# To build and push to a registry (replace REGISTRY in Makefile first)
make docker-push-cpu
make docker-push-gpu
```

### Cross-Platform Building on Mac

If you're using a Mac with Apple Silicon (ARM64), the buildx commands will create images for AMD64 architecture that will work on your x86_64 Kubernetes cluster.

### Running the Docker Container Locally

Using the Makefile:

```bash
# Run the Docker container (CPU version)
make docker-run
```

Or manually with Docker:

```bash
docker run -p 8888:8888 \
  -e SERVER_PORT=8888 \
  -e MODEL_NAME=tts_models/en/vctk/vits \
  -e MODEL_DEVICE=cpu \
  -e SERVER_LOG_LEVEL=info \
  --name vox-raga-container \
  your-registry/vox-raga:0.0.1-cpu
```

### Environment Variables

The Docker container supports the following environment variables:

- `SERVER_PORT`: Port the server listens on (default: 8888)
- `MODEL_NAME`: TTS model to use (default: tts_models/en/vctk/vits)
- `MODEL_DEVICE`: Device to use for inference (cpu or cuda, default depends on image)
- `MODEL_DOWNLOAD_ROOT`: Directory where models are stored (default: /app/models)
- `SERVER_CACHE_DIR`: Directory for caching (default: /app/cache)
- `SERVER_LOG_LEVEL`: Logging level (default: info)

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster running on x86_64/amd64 architecture
- kubectl configured to connect to your cluster
- For GPU usage: NVIDIA GPU operator or device plugin installed on your cluster

### Deploying to Kubernetes

The project provides separate deployment configurations for CPU and GPU usage:

```bash
# For CPU-only deployment
kubectl apply -f k8s/vox-raga-cpu-deployment.yaml

# For GPU-accelerated deployment (requires NVIDIA GPUs)
kubectl apply -f k8s/vox-raga-deployment.yaml
```

### Customizing the Deployment

Both deployment files include commented sections for:

- Node affinity to target specific nodes (with or without GPUs)
- Resource requests and limits
- Environment variable configuration

## API Usage

Once deployed, you can use the API to convert text to speech:

### Text-to-Speech API

```bash
curl -X POST "http://localhost:8888/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the text to speech system.",
    "options": {
      "language": "en",
      "voice": "default",
      "speed": 1.0,
      "format": "wav"
    }
  }' \
  --output speech.wav
```

### Health Check API

```bash
curl http://localhost:8888/health
```

## Architecture

The service is built using:

- [Coqui TTS](https://github.com/coqui-ai/TTS) as the base Docker image
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- Kubernetes for orchestration and scaling

### Model Loading Strategy

The service is designed to:

1. First check for models in the /app/models directory
2. Check standard Coqui TTS model paths in the container
3. Use Coqui's built-in model loading capabilities as a fallback

## Development

### Local Development Setup

```bash
# Set up development environment
make setup-dev

# Run development server
make dev
```

### Running Tests

```bash
make test
```

## Troubleshooting

### Common Issues

1. **Architecture mismatch**: Make sure you're using buildx to create images for the correct architecture (amd64)
2. **GPU not detected**: Ensure the NVIDIA device plugin is installed on your Kubernetes cluster
3. **Permission issues**: The init container should set appropriate permissions on volumes
4. **Models not found**: Check that the model exists in the Coqui TTS repository or in your models volume

## License

See the LICENSE file for details. 