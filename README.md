# VoxRaga

A high-quality, scalable Text-to-Speech (TTS) service built with FastAPI and Coqui TTS.

## Project Overview

VoxRaga is a modular, scalable, and high-performance Text-to-Speech service that can operate in both cloud and edge environments. It is part of a larger speech-to-speech translation pipeline that includes:

1. **Speech-to-Text (ASR)** - Kube-Whisperer
2. **Text Translation** - Translation Service
3. **Text-to-Speech (TTS)** - VoxRaga (this service)

## Key Features

- High-quality, natural-sounding speech synthesis
- Support for 40+ languages
- Multiple voice options per language
- Adjustable speech parameters (speed, pitch, etc.)
- Emotion and style control
- REST and WebSocket APIs
- Kubernetes-native deployment
- Scalable architecture
- Comprehensive monitoring and observability

## Technical Architecture

- FastAPI for the API layer
- Coqui TTS for speech synthesis
- Docker containers for packaging
- Kubernetes-native deployment
- Prometheus for metrics
- Structured logging
- GPU acceleration with CPU fallback
- Multi-architecture support (AMD64/ARM64)

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Kubernetes (for cloud deployment)
- espeak or espeak-ng (required for phonemization)

### Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/vox-raga.git
   cd vox-raga
   ```

2. **Install espeak (required for TTS)**
   ```bash
   # On macOS
   brew install espeak
   
   # On Ubuntu/Debian
   sudo apt-get install espeak
   
   # On CentOS/RHEL
   sudo yum install espeak
   ```

3. **Set Up Development Environment**
   ```bash
   make setup-local
   ```

4. **Run Locally**
   ```bash
   make run-local
   ```

## Configuration Options

VoxRaga provides extensive configuration options to customize the TTS service behavior. Configuration can be set through environment variables or directly in the code.

### Environment Variables

#### Server Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SERVER_HOST` | Host to bind to | `0.0.0.0` | `127.0.0.1` |
| `SERVER_PORT` | Port to bind to | `8000` | `9000` |
| `SERVER_LOG_LEVEL` | Logging level | `info` | `debug` |
| `SERVER_CORS_ORIGINS` | CORS origins (comma-separated) | `*` | `http://localhost:3000,https://example.com` |
| `SERVER_METRICS_ENABLED` | Enable Prometheus metrics | `true` | `false` |
| `SERVER_CACHE_DIR` | Cache directory | `/tmp/tts_cache` | `/data/cache` |
| `SERVER_MAX_CACHE_SIZE_MB` | Maximum cache size in MB | `1024` | `2048` |

#### Model Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MODEL_NAME` | TTS model name | `tts_models/en/vctk/vits` | `tts_models/en/ljspeech/tacotron2-DDC` |
| `MODEL_DEVICE` | Device to use | `cpu` | `cuda` |
| `MODEL_COMPUTE_TYPE` | Compute type | `float32` | `float16` |
| `MODEL_CPU_THREADS` | Number of CPU threads | `4` | `8` |
| `MODEL_NUM_WORKERS` | Number of workers | `1` | `2` |
| `MODEL_DOWNLOAD_ROOT` | Root directory for model downloads | `/tmp/tts_models` | `/data/models` |

### Available Models

VoxRaga uses Coqui TTS models. The default model is `tts_models/en/vctk/vits`, which provides high-quality multi-speaker English synthesis. Other available models include:

- `tts_models/en/ljspeech/tacotron2-DDC` - Single-speaker English
- `tts_models/en/ljspeech/glow-tts` - Single-speaker English with faster inference
- `tts_models/en/ljspeech/fast_pitch` - Single-speaker English with controllable prosody
- `tts_models/multilingual/multi-dataset/your_tts` - Multilingual with voice cloning
- `tts_models/de/thorsten/tacotron2-DDC` - German
- `tts_models/fr/mai/tacotron2-DDC` - French
- `tts_models/es/mai/tacotron2-DDC` - Spanish

For a complete list of available models, refer to the [Coqui TTS documentation](https://github.com/coqui-ai/TTS/wiki/Models).

### Hardware Acceleration

VoxRaga supports hardware acceleration for faster inference:

- **CUDA**: Set `MODEL_DEVICE=cuda` to use NVIDIA GPUs
- **MPS**: Set `MODEL_DEVICE=mps` to use Apple Silicon GPUs (M1/M2)
- **CPU**: Set `MODEL_DEVICE=cpu` for CPU-only inference

For optimal performance with GPU acceleration, set `MODEL_COMPUTE_TYPE=float16`.

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/ready` | GET | Readiness check endpoint |
| `/live` | GET | Liveness check endpoint |
| `/synthesize` | POST | TTS endpoint |
| `/config` | GET | Configuration endpoint |
| `/voices` | GET | List available voices |
| `/languages` | GET | List available languages |

### Synthesize Endpoint

The `/synthesize` endpoint accepts a POST request with the following JSON body:

```json
{
  "text": "Text to synthesize",
  "options": {
    "language": "en",
    "voice": "p225",
    "speed": 1.0,
    "pitch": 1.0,
    "energy": 1.0,
    "format": "wav",
    "sample_rate": 22050
  }
}
```

#### Request Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `text` | Text to synthesize | (required) | Any text string |
| `options.language` | Language code | `"en"` | Depends on available languages |
| `options.voice` | Voice identifier | `"p225"` | Depends on available voices |
| `options.speed` | Speech speed multiplier | `1.0` | `0.5` to `2.0` |
| `options.pitch` | Voice pitch multiplier | `1.0` | `0.5` to `2.0` |
| `options.energy` | Voice energy/volume | `1.0` | `0.5` to `2.0` |
| `options.format` | Audio format | `"wav"` | `"wav"`, `"mp3"`, `"ogg"` |
| `options.sample_rate` | Audio sample rate | `22050` | Common values: `8000`, `16000`, `22050`, `44100`, `48000` |

#### Response

The response is a binary audio file with the appropriate content type:
- `audio/wav` for WAV format
- `audio/mpeg` for MP3 format
- `audio/ogg` for OGG format

The response includes the following headers:
- `X-Processing-Time`: Processing time in seconds
- `X-Language`: Language used for synthesis
- `X-Voice`: Voice used for synthesis

### Voice Selection

To list all available voices:

```bash
curl -X GET http://localhost:8000/voices
```

Response:
```json
{
  "voices": ["p225", "p226", "p227", "p228", "..."]
}
```

### Language Selection

To list all available languages:

```bash
curl -X GET http://localhost:8000/languages
```

Response:
```json
{
  "languages": ["en", "fr", "de", "es", "..."]
}
```

## Testing the TTS Service

VoxRaga provides multiple ways to test the text-to-speech functionality:

### Using the API Directly

You can test the API endpoints directly using curl:

1. **Health Check**
   ```bash
   curl -X GET http://localhost:8000/health
   ```

2. **List Available Voices**
   ```bash
   curl -X GET http://localhost:8000/voices
   ```

3. **List Available Languages**
   ```bash
   curl -X GET http://localhost:8000/languages
   ```

4. **Synthesize Speech**
   ```bash
   curl -X POST http://localhost:8000/synthesize \
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

### Voice Selection and Configuration Options

The TTS service supports various configuration options to customize the generated speech:

#### Voice Selection

The default model (VITS trained on VCTK dataset) includes over 100 different voices. Each voice has a unique identifier (e.g., "p225", "p226", etc.). To list all available voices:

```bash
curl -X GET http://localhost:8000/voices
```

Or using the test script:

```bash
cd samples
python test_tts.py --list-voices
```

#### Language Selection

To list all available languages:

```bash
curl -X GET http://localhost:8000/languages
```

Or using the test script:

```bash
cd samples
python test_tts.py --list-languages
```

#### Speech Parameters

The following parameters can be adjusted when synthesizing speech:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `language` | Language code | "en" | Depends on available languages |
| `voice` | Voice identifier | "p225" | Depends on available voices |
| `speed` | Speech speed multiplier | 1.0 | 0.5 to 2.0 |
| `pitch` | Voice pitch multiplier | 1.0 | 0.5 to 2.0 |
| `energy` | Voice energy/volume | 1.0 | 0.5 to 2.0 |
| `format` | Audio format | "wav" | "wav", "mp3", "ogg" |
| `sample_rate` | Audio sample rate | 22050 | Common values: 8000, 16000, 22050, 44100, 48000 |

Example with all parameters:

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test of the text to speech system.",
    "options": {
      "language": "en",
      "voice": "p226",
      "speed": 1.2,
      "pitch": 0.9,
      "energy": 1.1,
      "format": "mp3",
      "sample_rate": 44100
    }
  }' --output test.mp3
```

Using the test script:

```bash
python test_tts.py --voice p226 --speed 1.2 --format mp3 --language en
```

Note: Not all parameters are available through the test script. For full control, use the API directly.

### Using the Sample Test Script

We provide a convenient test script that can process multiple text files and generate audio outputs:

1. **Navigate to the samples directory**
   ```bash
   cd samples
   ```

2. **List available voices**
   ```bash
   python test_tts.py --list-voices
   ```

3. **List available languages**
   ```bash
   python test_tts.py --list-languages
   ```

4. **Process sample text files**
   ```bash
   python test_tts.py --voice p225 --format wav
   ```

5. **Customize the output**
   ```bash
   python test_tts.py --voice p226 --speed 1.2 --format mp3
   ```

6. **Run all tests with different voices and formats**
   ```bash
   ./run_all_tests.sh
   ```

The script will:
- Read all .txt files from the `samples/input` directory
- Send each file's content to the TTS service
- Save the resulting audio files to the `samples/output` directory

### Using the HTML Audio Player

We also provide a simple HTML player to listen to the generated audio files:

1. **Generate audio files first**
   ```bash
   cd samples
   python test_tts.py --voice p225 --format wav
   python test_tts.py --voice p225 --format mp3
   python test_tts.py --voice p225 --format ogg
   ```

2. **Open the HTML player**
   ```bash
   open player.html  # On macOS
   # Or open it in your browser manually
   ```

The HTML player allows you to:
- See the original text for each sample
- Play the audio in different formats (WAV, MP3, OGG)
- Compare the quality of different formats

### Sample Files

The repository includes several sample text files in the `samples/input` directory:

- `sample1.txt`: A simple test sentence
- `sample2.txt`: A longer paragraph about AI
- `sample3.txt`: Text with questions and exclamations to test intonation

### Output Formats

The TTS service supports the following output formats:

- **WAV** (.wav): Uncompressed audio, highest quality but larger file size
- **MP3** (.mp3): Compressed audio, good balance of quality and file size
- **OGG** (.ogg): Open format compressed audio, similar to MP3

### Playing the Audio Files

You can play the generated audio files using any standard media player:

- **macOS**: Double-click the file or use `afplay test.wav` in the terminal
- **Linux**: Use `aplay test.wav` or `mpg123 test.mp3` in the terminal
- **Windows**: Double-click the file or use Windows Media Player

## Advanced Usage

### Batch Processing

For batch processing of multiple text files, you can use the sample test script or create your own script using the API:

```python
import requests
import os

def synthesize_batch(texts, output_dir, options=None):
    if options is None:
        options = {"language": "en", "voice": "p225", "format": "wav"}
    
    for i, text in enumerate(texts):
        response = requests.post(
            "http://localhost:8000/synthesize",
            json={"text": text, "options": options}
        )
        
        if response.status_code == 200:
            output_file = os.path.join(output_dir, f"output_{i}.{options['format']}")
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"Generated {output_file}")
        else:
            print(f"Error processing text {i}: {response.text}")

# Example usage
texts = [
    "This is the first sample text.",
    "This is the second sample text.",
    "This is the third sample text."
]
synthesize_batch(texts, "output", {"voice": "p226", "format": "mp3"})
```

### Streaming Audio

For real-time applications, you can stream the audio directly to the client:

```python
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import requests
import io

app = FastAPI()

@app.get("/stream-tts")
async def stream_tts(text: str, voice: str = "p225"):
    response = requests.post(
        "http://localhost:8000/synthesize",
        json={
            "text": text,
            "options": {"voice": voice, "format": "mp3"}
        }
    )
    
    if response.status_code == 200:
        return StreamingResponse(
            io.BytesIO(response.content),
            media_type="audio/mpeg"
        )
    else:
        return Response(
            content=f"Error: {response.text}",
            status_code=response.status_code
        )
```

### Voice Cloning

Some advanced models support voice cloning. To use this feature, you need to:

1. Select a model that supports voice cloning (e.g., `tts_models/multilingual/multi-dataset/your_tts`)
2. Provide a reference audio file of the target voice
3. Use the API to clone the voice and synthesize speech

This feature is not available in the current version but will be added in a future release.

## Docker Deployment

1. **Build the Docker Image**
   ```bash
   make build
   ```

2. **Push to Container Registry**
   ```bash
   export ACR_NAME=youracrname
   export REGISTRY_TYPE=acr
   make push
   ```

## Kubernetes Deployment

1. **Deploy to Kubernetes**
   ```bash
   make deploy
   ```

The Kubernetes deployment uses the following resources:
- Deployment with configurable replicas
- Service for internal access
- Ingress for external access
- ConfigMap for configuration
- Secret for sensitive data
- PersistentVolumeClaim for model storage

## Performance Tuning

To optimize performance:

1. **Use GPU acceleration** by setting `MODEL_DEVICE=cuda` or `MODEL_DEVICE=mps`
2. **Reduce precision** by setting `MODEL_COMPUTE_TYPE=float16`
3. **Increase cache size** by setting `SERVER_MAX_CACHE_SIZE_MB=2048`
4. **Use a faster model** like `tts_models/en/ljspeech/glow-tts` for lower latency
5. **Adjust CPU threads** based on your hardware by setting `MODEL_CPU_THREADS`

## Running Tests

VoxRaga includes a comprehensive test suite to ensure the service functions correctly. The tests are organized into different categories:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test the interaction between components
- **API tests**: Test the API endpoints
- **Model tests**: Test the TTS model functionality

### Running All Tests

To run all tests:

```bash
python -m pytest
```

### Running Specific Test Categories

To run only integration tests:

```bash
python -m pytest tests/test_integration.py
```

To run tests with a specific mark:

```bash
python -m pytest -m integration
python -m pytest -m unit
python -m pytest -m api
python -m pytest -m model
```

### Test Coverage

To generate a test coverage report:

```bash
python -m pytest --cov=src
```

For a detailed HTML coverage report:

```bash
python -m pytest --cov=src --cov-report=html
```

The HTML report will be available in the `htmlcov` directory.

## Monitoring and Observability

VoxRaga includes comprehensive monitoring and observability features:

- **Prometheus metrics** at `/metrics` endpoint
- **Structured logging** with configurable log levels
- **Health checks** at `/health`, `/ready`, and `/live` endpoints
- **Request tracing** with unique request IDs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.