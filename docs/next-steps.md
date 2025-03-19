# Implementation Guide: Dockerizing vox-raga using Coqui TTS as a Base Image

This guide outlines the approach to Dockerize the vox-raga TTS service using the official Coqui TTS Docker image as a base, with priorities (P0, P1, P2) for each implementation task.

## P0: Basic Implementation - Using Coqui Base Image with vox-raga

### Base Image Selection

```dockerfile
# For CPU-only deployments
FROM ghcr.io/coqui-ai/tts-cpu:v0.22.0

# OR for GPU-enabled deployments
FROM ghcr.io/coqui-ai/tts:v0.22.0
```

**Note on GPU vs CPU**: The GPU image will not automatically fall back to CPU if a GPU is not available. You must select the appropriate image for your environment. If you're using Kubernetes, you can have separate deployments for GPU and CPU nodes with the appropriate image for each.

### Basic Dockerfile

```dockerfile
# Select the appropriate base image
FROM ghcr.io/coqui-ai/tts-cpu:v0.22.0

# Set up working directory
WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Create model directory with proper permissions
RUN mkdir -p /app/models

# Environment variables
ENV PYTHONPATH=/app \
    SERVER_PORT=8888 \
    # Point to where models might be found in the Coqui image
    MODEL_DOWNLOAD_ROOT=/app/models \
    MODEL_NAME=tts_models/en/vctk/vits \
    SERVER_LOG_LEVEL=info

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run your FastAPI application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8888"]
```

### Model Path Modification

To avoid permission issues and the need for copying from root directories, modify your `TTSModelManager` class to be aware of where Coqui stores models. 

Add this code to `src/models/tts_model.py`:

```python
def get_model_paths(self, model_name: str) -> list[str]:
    """Get all possible paths where a model might be located."""
    # Standard paths in our configuration
    paths = [os.path.join(self.config.download_root, *model_name.split("/"))]
    
    # Add Coqui's model path format (transforming tts_models/en/vctk/vits to tts_models--en--vctk--vits)
    coqui_model_id = "--".join(model_name.split("/"))
    
    # Common locations where Coqui might store models
    coqui_paths = [
        f"/usr/local/lib/python3.8/site-packages/TTS/.models/{coqui_model_id}",
        f"/usr/local/lib/python3.9/site-packages/TTS/.models/{coqui_model_id}",
        f"/usr/local/lib/python3.10/site-packages/TTS/.models/{coqui_model_id}",
    ]
    
    paths.extend(coqui_paths)
    return paths
```

Then update your model loading code to check all these paths:

```python
def load_model(self, model_name: str) -> Synthesizer:
    """Load model from any available location."""
    model_paths = self.get_model_paths(model_name)
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            # Your existing model loading code...
            return synthesizer
    
    # If no path worked, try using TTS's built-in model loading
    try:
        logger.info(f"Trying to load {model_name} using TTS built-in model loading")
        synthesizer = Synthesizer(
            model_name=model_name,
            tts_checkpoint=None,
            tts_config_path=None,
            vocoder_checkpoint=None,
            vocoder_config=None,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=self.config.device == "cuda",
        )
        return synthesizer
    except Exception as e:
        logger.error(f"Failed to load model using TTS built-in loader: {str(e)}")
        raise ValueError(f"Could not find or load model: {model_name}")
```

### Kubernetes Deployment (basic)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vox-raga
  labels:
    app: vox-raga
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vox-raga
  template:
    metadata:
      labels:
        app: vox-raga
    spec:
      containers:
      - name: vox-raga
        image: your-registry/vox-raga:latest
        ports:
        - containerPort: 8888
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: SERVER_PORT
          value: "8888"
        - name: MODEL_NAME
          value: "tts_models/en/vctk/vits"
        - name: SERVER_LOG_LEVEL
          value: "info"
        livenessProbe:
          httpGet:
            path: /live
            port: 8888
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8888
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: vox-raga
spec:
  selector:
    app: vox-raga
  ports:
  - port: 8888
    targetPort: 8888
  type: ClusterIP
```

## P1: Enhanced Features - Custom Languages and Models

### 1. Custom Model Support

#### Update the model manager to allow loading custom models:

Add this to `src/models/tts_model.py`:

```python
def register_custom_model(self, model_path: str, config_path: str, model_name: str) -> None:
    """Register a custom TTS model."""
    # Create directory for the model
    model_dir = os.path.join(self.config.download_root, *model_name.split("/"))
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    
    # Copy model and config files
    import shutil
    model_target = os.path.join(model_dir, "model_file.pth")
    config_target = os.path.join(model_dir, "config.json")
    
    shutil.copy(model_path, model_target)
    shutil.copy(config_path, config_target)
    
    # If this is a multi-speaker model, generate speaker_ids.json if needed
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if config.get("num_speakers", 0) > 1 and not os.path.exists(os.path.join(model_dir, "speaker_ids.json")):
            speaker_ids = {f"speaker_{i}": i for i in range(config["num_speakers"])}
            with open(os.path.join(model_dir, "speaker_ids.json"), "w") as f:
                json.dump(speaker_ids, f)
    except Exception as e:
        logger.warning(f"Failed to check or create speaker_ids.json: {str(e)}")
    
    # Clear model from cache if it exists
    if model_name in self.models:
        del self.models[model_name]
    
    logger.info(f"Custom model {model_name} registered at {model_dir}")
```

#### Add an API endpoint for uploading custom models:

Add to `src/api/tts.py`:

```python
@router.post("/models/upload")
async def upload_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    config_file: UploadFile = File(...),
    model_name: str = Form(...),
) -> Dict[str, Any]:
    """Upload a custom TTS model."""
    # Create temporary directory
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files
        model_path = os.path.join(temp_dir, "model.pth")
        config_path = os.path.join(temp_dir, "config.json")
        
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        with open(config_path, "wb") as f:
            content = await config_file.read()
            f.write(content)
        
        # Register model
        model_manager.register_custom_model(model_path, config_path, model_name)
        
        # Schedule cleanup
        background_tasks.add_task(lambda: os.unlink(model_path))
        background_tasks.add_task(lambda: os.unlink(config_path))
        background_tasks.add_task(lambda: os.rmdir(temp_dir))
        
        return {
            "status": "success", 
            "model_name": model_name,
            "message": f"Model {model_name} uploaded successfully"
        }
    except Exception as e:
        # Clean up on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.error(f"Failed to upload model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload model: {str(e)}"
        )
```

### 2. Enhanced Kubernetes Deployment with Persistent Storage

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vox-raga-models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vox-raga
spec:
  # ... other deployment fields ...
  template:
    spec:
      containers:
      - name: vox-raga
        # ... other container fields ...
        volumeMounts:
        - name: models-storage
          mountPath: /app/models
      volumes:
      - name: models-storage
        persistentVolumeClaim:
          claimName: vox-raga-models
```

### 3. Model Discovery Endpoint

Add to `src/api/tts.py`:

```python
@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available TTS models."""
    # Try to fetch models from TTS library
    from TTS.utils.manage import ModelManager
    try:
        mm = ModelManager()
        tts_models = mm.list_tts_models()
    except Exception as e:
        logger.warning(f"Failed to list TTS models: {str(e)}")
        tts_models = []
    
    # Add custom models from our storage
    custom_models = []
    try:
        model_root = config.model.download_root
        if os.path.exists(model_root):
            for root, dirs, files in os.walk(model_root):
                if "config.json" in files and any(f.endswith(".pth") for f in files):
                    rel_path = os.path.relpath(root, model_root)
                    if rel_path != ".":
                        custom_models.append(rel_path)
    except Exception as e:
        logger.warning(f"Failed to list custom models: {str(e)}")
    
    # Return combined list with metadata
    models_info = {}
    
    # Add library models
    for model in tts_models:
        model_parts = model.split("/")
        models_info[model] = {
            "type": "library",
            "language": model_parts[1] if len(model_parts) > 1 else "unknown",
            "dataset": model_parts[2] if len(model_parts) > 2 else "unknown",
            "architecture": model_parts[3] if len(model_parts) > 3 else "unknown"
        }
    
    # Add custom models
    for model in custom_models:
        model_parts = model.split("/")
        models_info[model] = {
            "type": "custom",
            "language": model_parts[1] if len(model_parts) > 1 else "unknown",
            "dataset": model_parts[2] if len(model_parts) > 2 else "unknown",
            "architecture": model_parts[3] if len(model_parts) > 3 else "unknown"
        }
    
    return {"models": models_info}
```

## P2: Advanced Optimizations and Features

### 1. Request Queueing for Performance Management

Add a request queue to prevent overloading the system:

```python
class RequestQueue:
    """Queue for managing concurrent synthesis requests."""
    
    def __init__(self, max_concurrent=4):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_gauge = Gauge(
            "tts_request_queue_size", 
            "Number of requests in queue"
        )
        self._queue_gauge.set(0)
    
    async def process(self, func, *args, **kwargs):
        """Process a request through the queue."""
        # Increase queue counter
        self._queue_gauge.inc()
        
        try:
            # Wait for a slot
            async with self._semaphore:
                # Process the request
                return await func(*args, **kwargs)
        finally:
            # Decrease queue counter
            self._queue_gauge.dec()

# Initialize queue
request_queue = RequestQueue(max_concurrent=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")))

# Modify synthesis endpoint
@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """Synthesize speech from text."""
    return await request_queue.process(
        _synthesize_speech_impl, request, background_tasks
    )

async def _synthesize_speech_impl(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """Implementation of speech synthesis."""
    # Your existing implementation...
```

### 2. Model Caching Strategy

Enhance the model manager with better caching:

```python
def get_model(self, model_name: Optional[str] = None) -> Synthesizer:
    """Get a TTS model by name with improved caching."""
    # Use default model if not specified
    if model_name is None:
        model_name = self.config.model_name
    
    # Return cached model if available
    if model_name in self.models:
        # Update last used timestamp
        self.model_usage[model_name] = time.time()
        return self.models[model_name]
    
    # Check cache limit before loading new model
    if len(self.models) >= self.config.max_cached_models:
        # Evict least recently used model
        self._evict_lru_model()
    
    # Load model
    logger.info("Loading TTS model", model_name=model_name)
    model = self.load_model(model_name)
    
    # Cache model
    self.models[model_name] = model
    self.model_usage[model_name] = time.time()
    
    # Update metrics
    from src.utils.metrics import MODEL_CACHE_SIZE
    MODEL_CACHE_SIZE.set(len(self.models))
    
    return model

def _evict_lru_model(self):
    """Evict least recently used model from cache."""
    if not self.models:
        return
    
    # Find least recently used model
    lru_model = min(self.model_usage.items(), key=lambda x: x[1])[0]
    
    # Remove from cache
    if lru_model in self.models:
        logger.info(f"Evicting model {lru_model} from cache")
        del self.models[lru_model]
        del self.model_usage[lru_model]
```

### 3. Advanced Monitoring with Prometheus

Add more detailed metrics:

```python
# In src/utils/metrics.py
# Add model-specific metrics
MODEL_SYNTHESIS_COUNT = Counter(
    "tts_model_synthesis_count",
    "Number of synthesis operations per model",
    ["model_name"]
)

MODEL_SYNTHESIS_LATENCY = Histogram(
    "tts_model_synthesis_latency_seconds",
    "Latency of synthesis operations per model",
    ["model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# In TTSSynthesizer.synthesize
def synthesize(self, text: str, options: SynthesisOptions) -> bytes:
    """Synthesize speech from text."""
    start_time = time.time()
    model_name = options.model or self.config.model_name
    
    try:
        # Synthesis code...
        
        # Record metrics
        latency = time.time() - start_time
        MODEL_SYNTHESIS_COUNT.labels(model_name=model_name).inc()
        MODEL_SYNTHESIS_LATENCY.labels(model_name=model_name).observe(latency)
        
        return audio_bytes
    except Exception as e:
        # Error handling...
        raise
```

### 4. A/B Testing Support

Add support for A/B testing different models:

```python
@router.post("/synthesize/compare")
async def compare_models(
    text: str = Body(..., description="Text to synthesize"),
    models: List[str] = Body(..., description="Models to compare"),
    options: SynthesisOptions = Body(SynthesisOptions(), description="Synthesis options"),
) -> Dict[str, Any]:
    """Synthesize the same text with multiple models for comparison."""
    results = {}
    errors = {}
    
    for model_name in models:
        try:
            # Create options with the specific model
            model_options = SynthesisOptions(**options.dict())
            
            # Synthesize
            start_time = time.time()
            audio_bytes = synthesizer.synthesize(text=text, options=model_options, model_name=model_name)
            latency = time.time() - start_time
            
            # Encode audio as base64
            import base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Add to results
            results[model_name] = {
                "audio_base64": audio_b64,
                "latency_seconds": latency,
                "format": options.format
            }
        except Exception as e:
            errors[model_name] = str(e)
    
    return {
        "results": results,
        "errors": errors,
        "text": text
    }
```

### 5. Enhanced Dockerfile with Resource Optimization

```dockerfile
FROM ghcr.io/coqui-ai/tts-cpu:v0.22.0

# Set up working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add non-root user
RUN adduser --disabled-password --gecos '' appuser

# Create and configure directories
RUN mkdir -p /app/models /app/cache \
    && chown -R appuser:appuser /app/models /app/cache

# Set up proper Python compilation
ENV PYTHONPYCACHEPREFIX=/app/cache \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/

# Switch to non-root user
USER appuser

# Environment variables for performance tuning
ENV PYTHONPATH=/app \
    SERVER_PORT=8888 \
    MODEL_DOWNLOAD_ROOT=/app/models \
    MODEL_NAME=tts_models/en/vctk/vits \
    SERVER_LOG_LEVEL=info \
    MAX_CONCURRENT_REQUESTS=4 \
    MAX_CACHED_MODELS=3 \
    MODEL_CPU_THREADS=4

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run with optimized settings
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8888", "--workers", "1", "--limit-concurrency", "20"]
```

## Implementation Considerations

### No Root Access Needed

This implementation avoids the need for root access or scripts that copy from root directories by:

1. Setting up the application to look for models in multiple locations
2. Using the Coqui TTS library's built-in model loading capabilities
3. Creating a separate model directory within the app's space

### Image Selection (CPU vs GPU)

- **CPU Image**: Use `ghcr.io/coqui-ai/tts-cpu:v0.22.0` for deployments without GPU requirements
- **GPU Image**: Use `ghcr.io/coqui-ai/tts:v0.22.0` for deployments with NVIDIA GPU support

The GPU version will **not** automatically fall back to CPU mode if a GPU is not available. Instead, it will typically fail to initialize. For deployment environments with mixed GPU availability, consider:

1. Separate CPU and GPU deployments with appropriate node affinities
2. Custom entrypoint script that checks for GPU and modifies configuration accordingly

### CI/CD Pipeline Recommendations

For a complete deployment pipeline:

1. **Build**: Create the Docker image with your application code
2. **Test**: Run unit and integration tests against the built image
3. **Scan**: Use security scanning tools like Trivy to check for vulnerabilities
4. **Push**: Push the image to your container registry
5. **Deploy**: Update Kubernetes manifests and deploy

### Next Steps After Implementation

After implementing the P0, P1, and P2 features:

1. **Performance testing** under load to adjust resource settings
2. **Security audit** of the application and container
3. **Documentation** for API users and operations teams
4. **Monitoring dashboards** using Prometheus metrics
5. **Backup strategy** for custom models and data

This implementation approach balances leveraging the Coqui TTS image's capabilities while preserving and extending your custom application's features without requiring root access or custom scripts.
