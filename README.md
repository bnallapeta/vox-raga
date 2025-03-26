# VoxRaga

A high-performance Text-to-Speech (TTS) service built with FastAPI and Coqui TTS.

## Overview

VoxRaga delivers natural-sounding speech synthesis with support for multiple languages and voices. Built for running locally and on Kubernetes clusters, it offers a RESTful API that integrates seamlessly with existing speech processing pipelines.

## Features

- High-quality speech synthesis using state-of-the-art neural models
- Multi-language and multi-voice support
- Adjustable speech parameters (speed, pitch, format)
- REST API with JSON interface
- GPU-accelerated inference
- Kubernetes-ready containerization
- Prometheus metrics and health monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Kubernetes cluster with GPU nodes (for cloud deployment)
- Docker or Podman
- espeak or espeak-ng (for phonemization)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vox-raga.git
cd vox-raga

# Setup development environment
make setup-dev

# Run development server
make dev
```

### Local Deployment

```bash
# Build Docker image
make build

# Run locally
make run
```

## Cloud Deployment

### Push to Azure Container Registry

```bash
# Login to ACR
make acr-login

# Build and push in one step
make acr-push
```

### Deploy to Kubernetes with KServe

VoxRaga is deployed as a KServe InferenceService, which provides scaling, monitoring, and routing capabilities.

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/inferenceservice.yaml

# Check deployment status
kubectl get inferenceservices
```

The deployment creates a KServe InferenceService that automatically scales based on demand and provides a RESTful endpoint for clients to consume.

## Configuration

VoxRaga is configured through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_PORT` | Port to bind to | `8888` |
| `MODEL_NAME` | TTS model name | `tts_models/multilingual/multi-dataset/xtts_v2` |
| `MODEL_DEVICE` | Compute device | `cuda` |
| `SERVER_LOG_LEVEL` | Logging level | `info` |
| `MODEL_DOWNLOAD_ROOT` | Model storage location | `/app/models` |
| `SERVER_CACHE_DIR` | Cache directory | `/app/cache` |

## API Usage

### Synthesize Speech

```bash
curl -X POST http://localhost:8888/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the text to speech system.",
    "options": {
      "language": "en",
      "voice": "p225",
      "speed": 1.0,
      "format": "wav"
    }
  }' --output test.wav
```

### List Available Voices

```bash
curl -X GET http://localhost:8888/voices
```

### List Available Languages

```bash
curl -X GET http://localhost:8888/languages
```

## Testing

VoxRaga includes comprehensive test suites:

```bash
# Run all tests
make test

# Try sample client
cd samples
python test_tts.py --list-voices
python test_tts.py --voice p225 --format wav
```

## Performance Optimization

For optimal performance:

- Enable hardware acceleration where available
- Set `MODEL_COMPUTE_TYPE=float16` for faster inference
- Consider models with lower latency for real-time applications

## Monitoring

VoxRaga exposes Prometheus metrics at the `/metrics` endpoint for monitoring:

- Request latency and throughput
- Model inference time
- Cache hit/miss rates
- Resource utilization

## License

This project is licensed under the MIT License - see the LICENSE file for details.