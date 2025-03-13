# Text-to-Speech (TTS) Service Implementation Guide

## Project Overview

### Vision
Create a modular, scalable, and high-performance speech-to-speech translation pipeline that can operate in both cloud and edge environments. The system will convert spoken language from one language to another in near real-time with high accuracy and natural-sounding output.

### Complete Pipeline Components
The pipeline consists of three independent but interconnected services:

1. **Speech-to-Text (ASR)** - Kube-Whisperer
   * Converts spoken audio into text
   * Uses OpenAI's Whisper models with various size options
   * Handles multiple audio formats and languages

2. **Text Translation** - Translation Service
   * Translates text from source to target language
   * Uses NLLB-200 models with various size options
   * Supports 100+ languages with high accuracy

3. **Text-to-Speech (TTS)** - This Service
   * Converts translated text into natural-sounding speech
   * Uses Coqui TTS with multi-language, multi-speaker support
   * Provides voice customization and emotion control

### Pipeline Flow
1. User submits audio in language A
2. Kube-Whisperer converts audio to text in language A
3. Translation Service converts text from language A to language B
4. TTS Service converts text in language B to audio in language B
5. User receives audio in language B

### Key Features of TTS Service
- High-quality, natural-sounding speech synthesis
- Support for 40+ languages
- Multiple voice options per language
- Adjustable speech parameters (speed, pitch, etc.)
- Emotion and style control
- REST and WebSocket APIs
- Kubernetes-native deployment
- Scalable architecture
- Comprehensive monitoring and observability

### Technical Architecture
- Microservices architecture with each component as an independent service
- Kubernetes-native deployment
- FastAPI for all service APIs
- Docker containers for packaging
- Prometheus for metrics
- Structured logging
- GPU acceleration with CPU fallback
- Multi-architecture support (AMD64/ARM64)

## Implementation Plan

### P0 - Foundation (Must Have)

#### Project Setup
- [ ] Create GitHub repository for TTS Service
- [ ] Initialize Python project with proper structure
- [ ] Set up development environment (Python 3.11+, virtual env)
- [ ] Create initial README.md with project overview
- [ ] Set up .gitignore for Python projects
- [ ] Create LICENSE file (MIT recommended)

#### Core Service Framework
- [ ] Initialize FastAPI application
- [ ] Set up basic project structure:
  ```
  tts-service/
  ├── src/
  │   ├── __init__.py
  │   ├── main.py
  │   ├── config.py
  │   ├── models/
  │   ├── api/
  │   ├── utils/
  │   └── logging_setup.py
  ├── tests/
  ├── requirements.txt
  ├── Dockerfile
  ├── Makefile
  ├── README.md
  └── .gitignore
  ```
- [ ] Create requirements.txt with core dependencies:
  ```
  fastapi==0.109.2
  uvicorn[standard]==0.27.1
  pydantic==2.6.1
  pydantic-settings==2.1.0
  python-multipart==0.0.9
  TTS==0.21.1
  torch==2.2.0
  numpy==1.26.4
  prometheus-client==0.19.0
  structlog==24.1.0
  soundfile==0.12.1
  librosa==0.10.1
  ```

#### Model Integration
- [ ] Implement Coqui TTS model loading functionality
- [ ] Create model configuration class with validation
- [ ] Implement model selection (base, multilingual, etc.)
- [ ] Add device selection (CPU/GPU)
- [ ] Implement model caching mechanism
- [ ] Create text-to-speech function with proper error handling
- [ ] Add support for basic voice parameters (speed, pitch)

#### Basic API Endpoints
- [ ] Implement health check endpoint (`/health`)
- [ ] Implement readiness check endpoint (`/ready`)
- [ ] Implement liveness check endpoint (`/live`)
- [ ] Create basic TTS endpoint (`/synthesize`)
- [ ] Implement configuration endpoint (`/config`)
- [ ] Add CORS middleware
- [ ] Create endpoint to list available voices (`/voices`)

#### Audio Processing
- [ ] Implement audio format conversion (WAV, MP3, OGG)
- [ ] Add audio quality configuration options
- [ ] Implement proper audio streaming
- [ ] Create audio caching mechanism
- [ ] Add support for SSML markup (basic)

#### Containerization
- [ ] Create multi-stage Dockerfile
- [ ] Optimize container size and layer caching
- [ ] Set up proper user permissions (non-root)
- [ ] Configure environment variables
- [ ] Add health checks to container
- [ ] Implement multi-architecture build support (AMD64/ARM64)
- [ ] Fix binary extension compatibility issues for cross-platform deployment

#### Basic Testing
- [ ] Set up pytest framework
- [ ] Create unit tests for core functionality
- [ ] Implement API tests
- [ ] Set up test fixtures
- [ ] Create audio quality assessment tests

#### Documentation
- [ ] Document API endpoints
- [ ] Create basic usage examples
- [ ] Document configuration options
- [ ] Add deployment instructions
- [ ] Document supported languages and voices

### P1 - Production Readiness

#### Enhanced API Features
- [ ] Implement batch synthesis endpoint (`/batch_synthesize`)
- [ ] Add language and voice validation
- [ ] Implement voice selection options
- [ ] Add synthesis options (emotion, style, etc.)
- [ ] Create async synthesis endpoint
- [ ] Implement WebSocket endpoint for streaming synthesis

#### Voice Management
- [ ] Implement voice selection by language
- [ ] Add voice cloning capabilities (basic)
- [ ] Create voice customization options
- [ ] Implement voice style transfer
- [ ] Add emotion control parameters

#### Monitoring & Observability
- [ ] Set up Prometheus metrics
  - Request count
  - Latency metrics
  - Error rates
  - Resource utilization
  - Audio generation time
- [ ] Implement structured logging
- [ ] Add request ID tracking
- [ ] Create detailed health checks
  - Model health
  - GPU health
  - System resources
  - Temporary directory

#### Performance Optimization
- [ ] Implement model quantization options
- [ ] Add compute type selection (int8, float16, float32)
- [ ] Optimize batch processing
- [ ] Implement caching for frequent synthesis requests
- [ ] Add thread/worker configuration
- [ ] Implement audio chunk processing for long texts

#### Kubernetes Deployment
- [ ] Create Kubernetes deployment YAML
- [ ] Set up resource requests/limits
- [ ] Configure liveness/readiness probes
- [ ] Add horizontal pod autoscaling
- [ ] Create service and ingress definitions
- [ ] Implement KServe InferenceService deployment
- [ ] Configure proper image selection for different architectures
- [ ] Set up secrets for container registry access

#### Multi-Architecture Support
- [ ] Implement proper multi-stage build process for cross-platform compatibility
- [ ] Separate pure Python packages from binary extensions in build process
- [ ] Ensure binary extensions are built on the target platform
- [ ] Test deployment on both AMD64 and ARM64 architectures
- [ ] Document multi-architecture build and deployment process
- [ ] Create Makefile targets for multi-architecture builds

#### CI/CD Pipeline
- [ ] Set up GitHub Actions workflow
- [ ] Implement automated testing
- [ ] Configure Docker image building
- [ ] Set up image publishing to container registry
- [ ] Add version tagging

#### Security Enhancements
- [ ] Implement input validation
- [ ] Add rate limiting to prevent abuse
- [ ] Add basic authentication for API access
- [ ] Configure proper file permissions
- [ ] Set up security context for Kubernetes
- [ ] Implement content filtering for inappropriate text

#### Testing
- [ ] Add unit tests for all components
- [ ] Add integration tests for API endpoints
- [ ] Add performance tests
- [ ] Add load tests
- [ ] Add audio quality tests
- [ ] Add documentation for API endpoints

### P2 - Advanced Features

#### Advanced Voice Features
- [ ] Implement advanced SSML support
- [ ] Add prosody control (emphasis, breaks, etc.)
- [ ] Implement voice cloning from audio samples
- [ ] Add multi-speaker synthesis in a single request
- [ ] Create voice style mixing capabilities
- [ ] Implement adaptive voice selection based on content

#### Integration with Pipeline
- [ ] Create API client for ASR service
- [ ] Implement API client for Translation service
- [ ] Add pipeline orchestration endpoints
- [ ] Implement proper error handling across services
- [ ] Create end-to-end examples
- [ ] Add support for streaming through the entire pipeline

#### Advanced Audio Processing
- [ ] Implement audio post-processing options
- [ ] Add background music/sound mixing
- [ ] Create audio effects library
- [ ] Implement adaptive audio quality based on network conditions
- [ ] Add audio normalization and enhancement
- [ ] Create audio watermarking capabilities

#### Advanced Monitoring
- [ ] Create Grafana dashboards
- [ ] Set up alerting rules
- [ ] Implement distributed tracing
- [ ] Add detailed performance metrics
- [ ] Create operational runbooks
- [ ] Implement audio quality monitoring

#### Edge Deployment
- [ ] Optimize for resource-constrained environments
- [ ] Implement model distillation options
- [ ] Create lightweight deployment configurations
- [ ] Add offline mode support
- [ ] Implement progressive model loading
- [ ] Create edge-specific voice models

### P3 - Enhancements & Optimizations

#### Advanced Language Features
- [ ] Add support for low-resource languages
- [ ] Implement dialect handling
- [ ] Add gender-aware synthesis
- [ ] Create formality level options
- [ ] Implement context-aware prosody
- [ ] Add support for code-switching (mixing languages)

#### Performance Tuning
- [ ] Optimize for specific hardware (TPU, etc.)
- [ ] Implement advanced caching strategies
- [ ] Add dynamic batch sizing
- [ ] Create performance benchmarking tools
- [ ] Implement adaptive resource allocation
- [ ] Optimize model loading and unloading strategies

#### User Experience
- [ ] Create interactive documentation
- [ ] Add synthesis quality feedback mechanism
- [ ] Implement usage analytics
- [ ] Create administrative dashboard
- [ ] Add custom model fine-tuning options
- [ ] Implement voice preference learning

#### Integration Options
- [ ] Create SDK for common languages
- [ ] Implement webhook support
- [ ] Add event streaming integration
- [ ] Create plugin system
- [ ] Implement multi-cloud support
- [ ] Add support for custom voice model hosting

## Technical Implementation Details

### Model Configuration

```python
class TTSModelConfig(BaseModel):
    """Configuration for the TTS model."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(default="tts_models/en/vctk/vits", description="TTS model name")
    device: str = Field(default="cpu", description="Device to use")
    compute_type: str = Field(default="float32", description="Compute type")
    cpu_threads: int = Field(default=4, ge=1, description="Number of CPU threads")
    num_workers: int = Field(default=1, ge=1, description="Number of workers")
    download_root: str = Field(default="/tmp/tts_models", description="Root directory for model downloads")
    
    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        # This would be expanded with actual model validation
        if not v or len(v) < 5:
            raise ValueError(f"Invalid model name: {v}")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v
    
    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        valid_types = ["int8", "float16", "float32"]
        if v not in valid_types:
            raise ValueError(f"Invalid compute type: {v}. Must be one of {valid_types}")
        return v
```

### Synthesis Options

```python
class SynthesisOptions(BaseModel):
    """Options for speech synthesis."""
    language: str = Field(default="en", description="Language code")
    voice: str = Field(default="default", description="Voice identifier")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice pitch multiplier")
    energy: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice energy/volume")
    emotion: Optional[str] = Field(default=None, description="Emotion to convey")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        # This would be expanded with actual language code validation
        if len(v) < 2 or len(v) > 5:
            raise ValueError(f"Invalid language code: {v}")
        return v
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["wav", "mp3", "ogg"]
        if v not in valid_formats:
            raise ValueError(f"Invalid audio format: {v}. Must be one of {valid_formats}")
        return v
```

### API Endpoint Implementation

```python
@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize_speech(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks
):
    """
    Synthesize speech from text.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Update metrics
        SYNTHESIS_REQUESTS.inc()
        
        # Get model
        model = get_tts_model(model_config)
        
        # Perform synthesis
        audio_bytes = model.synthesize(
            text=request.text,
            language=request.options.language,
            voice=request.options.voice,
            speed=request.options.speed,
            pitch=request.options.pitch,
            energy=request.options.energy,
            emotion=request.options.emotion,
            format=request.options.format,
            sample_rate=request.options.sample_rate
        )
        
        # Record latency
        latency = time.time() - start_time
        SYNTHESIS_LATENCY.observe(latency)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_resources)
        
        # Determine content type based on format
        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg"
        }.get(request.options.format, "application/octet-stream")
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers={
                "X-Processing-Time": str(latency),
                "X-Language": request.options.language,
                "X-Voice": request.options.voice
            }
        )
    except Exception as e:
        # Update error metrics
        SYNTHESIS_ERRORS.labels(type=type(e).__name__).inc()
        
        # Log error
        logger.error("Synthesis error", error=str(e), exc_info=True)
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis error: {str(e)}"
        )
```

### Dockerfile

```dockerfile
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
```

### Kubernetes Deployment

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "tts-service"
  namespace: "default"
  annotations:
    sidecar.istio.io/inject: "true"
spec:
  predictor:
    containers:
      - name: tts-service
        image: ${REGISTRY_IMAGE}
        env:
        - name: MODEL_NAME
          value: "tts_models/en/vctk/vits"
        - name: MODEL_DEVICE
          value: "cpu"
        - name: MODEL_COMPUTE_TYPE
          value: "float32"
        - name: SERVER_LOG_LEVEL
          value: "info"
        ports:
        - name: http1
          containerPort: 8000
          protocol: TCP
        resources:
          limits:
            cpu: "2"
            memory: 4Gi
          requests:
            cpu: "500m"
            memory: 2Gi
        readinessProbe:
          httpGet:
            path: /ready
            port: http1
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: http1
          initialDelaySeconds: 120
          periodSeconds: 20
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        volumeMounts:
        - name: model-cache
          mountPath: /tmp/tts_models
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
    volumes:
    - name: model-cache
      emptyDir: {}
    imagePullSecrets:
    - name: acr-secret
```

### Makefile

```makefile
# Phony targets declaration
.PHONY: all build push deploy test run clean install install-test help \
        cluster-deploy cluster-test registry-start registry-stop dev-build \
        dev-push local-deploy cloud-deploy setup-local run-local test-local \
        debug-deps debug-container clean-local create-secret show-config venv \
        cache-clean acr-login acr-build acr-push acr-clean acr-rebuild check-env \
        clean-artifacts container-info kserve-url test-kserve

# Core variables
ACR_NAME ?= bnracr
REGISTRY_TYPE ?= acr
REGISTRY_NAME ?= ${ACR_NAME}
REGISTRY_URL ?= $(if $(filter acr,$(REGISTRY_TYPE)),$(REGISTRY_NAME).azurecr.io,\
                $(if $(filter ghcr,$(REGISTRY_TYPE)),ghcr.io/${GITHUB_USERNAME},\
                $(if $(filter dockerhub,$(REGISTRY_TYPE)),docker.io/${DOCKER_USERNAME},\
                $(REGISTRY_NAME))))

# Image configuration
IMAGE_NAME ?= tts-service
TAG ?= latest
REGISTRY_IMAGE = $(REGISTRY_URL)/$(IMAGE_NAME):$(TAG)
LOCAL_REGISTRY ?= localhost:5000
LOCAL_IMAGE_NAME = $(LOCAL_REGISTRY)/$(IMAGE_NAME):$(TAG)
REGISTRY_SECRET_NAME ?= acr-secret

# Container runtime configuration
CONTAINER_RUNTIME ?= $(shell which podman 2>/dev/null || which docker 2>/dev/null)
CONTAINER_TYPE := $(shell basename $(CONTAINER_RUNTIME))

# Build configuration
PLATFORMS ?= linux/amd64,linux/arm64
CACHE_DIR ?= $(HOME)/.cache/tts-build
BUILD_JOBS ?= 2

# Development configuration
PYTHON ?= python3
VENV ?= venv
PIP ?= $(VENV)/bin/pip
KUBECONFIG ?= ${KUBECONFIG}

# Create cache directories
$(shell mkdir -p $(CACHE_DIR)/amd64 $(CACHE_DIR)/arm64)

# Default target
all: build

# Virtual environment setup
venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Local development setup
setup-local:
	@echo "Setting up local development environment..."
	mkdir -p /tmp/tts_models
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Build and push commands
build:
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(TAG) .
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(TAG) $(REGISTRY_IMAGE)

push: check-env
	$(CONTAINER_RUNTIME) push $(REGISTRY_IMAGE)

# Deployment commands
deploy: check-env
	sed -e "s|\$${REGISTRY_IMAGE}|$(REGISTRY_IMAGE)|g" \
	    -e "s|\$${REGISTRY_SECRET_NAME}|$(REGISTRY_SECRET_NAME)|g" \
	    k8s/tts-service.yaml | kubectl apply -f -

# Testing commands
run:
	$(PYTHON) src/main.py

run-local: setup-local
	@echo "Starting TTS service locally..."
	. $(VENV)/bin/activate && $(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

test:
	. $(VENV)/bin/activate && pytest tests/

# Cleanup commands
clean: clean-artifacts clean-local
	@echo "Clean complete!"

clean-local:
	rm -rf $(VENV)
	rm -rf /tmp/tts_models
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache

# Help
help:
	@echo "Available commands:"
	@echo "  Local Development:"
	@echo "    make setup-local   - Set up local development environment"
	@echo "    make run-local     - Run service locally"
	@echo "    make test          - Run tests"
	@echo ""
	@echo "  Build and Deploy:"
	@echo "    make build         - Build container image"
	@echo "    make push          - Push image to registry"
	@echo "    make deploy        - Deploy to Kubernetes"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean         - Clean up all resources"
	@echo ""
	@echo "  Miscellaneous:"
	@echo "    make help          - Show this help message"
```

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/tts-service.git
   cd tts-service
   ```

2. **Set Up Development Environment**
   ```bash
   make setup-local
   ```

3. **Run Locally**
   ```bash
   make run-local
   ```

4. **Test the API**
   ```bash
   curl -X POST http://localhost:8000/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is a test of the text to speech system.",
       "options": {
         "language": "en",
         "voice": "default",
         "speed": 1.0,
         "format": "wav"
       }
     }' --output test.wav
   ```

5. **Run Tests**
   ```bash
   make test
   ```

6. **Build and Push Container**
   ```bash
   # Set environment variables
   export ACR_NAME=youracrname
   export REGISTRY_TYPE=acr
   
   # Build and push
   make build
   make push
   ```

7. **Deploy to Kubernetes**
   ```bash
   make deploy
   ```

## Next Steps

After completing the P0 tasks, focus on:

1. Enhancing the voice quality and language support
2. Improving performance through optimization
3. Adding WebSocket support for streaming synthesis
4. Integrating with the ASR and Translation services
5. Setting up comprehensive monitoring

## Conclusion

This implementation guide provides a roadmap for building a robust, scalable TTS service that can be deployed in various environments. By following the prioritized tasks and technical implementation details, you can create a powerful service that forms the final part of the speech-to-speech translation pipeline.

The modular approach allows for independent development and deployment while ensuring compatibility with the other pipeline components. The service is designed to be flexible, performant, and production-ready, with a focus on scalability and reliability. 