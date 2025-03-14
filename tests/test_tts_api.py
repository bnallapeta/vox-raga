"""
Tests for TTS API endpoints.
"""
import io
import json
import zipfile
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.config import SynthesisOptions


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_synthesizer():
    """Mock synthesizer fixture."""
    mock = MagicMock()
    mock.synthesize.return_value = b"mock audio data"
    return mock


@pytest.mark.unit
@pytest.mark.api
def test_list_voices_endpoint(client):
    """Test list_voices endpoint."""
    with patch("src.api.tts.model_manager.list_available_voices", return_value=["p225", "p226"]):
        response = client.get("/voices")
        assert response.status_code == 200
        assert response.json() == {"voices": ["p225", "p226"]}


@pytest.mark.unit
@pytest.mark.api
def test_list_languages_endpoint(client):
    """Test list_languages endpoint."""
    with patch("src.api.tts.model_manager.list_available_languages", return_value=["en", "fr"]):
        response = client.get("/languages")
        assert response.status_code == 200
        assert response.json() == {"languages": ["en", "fr"]}


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_success(client, mock_synthesizer):
    """Test synthesize endpoint success."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        response = client.post(
            "/synthesize",
            json={
                "text": "Hello, world!",
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.0,
                    "format": "wav"
                }
            }
        )
        assert response.status_code == 200
        assert response.content == b"mock audio data"
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Processing-Time" in response.headers
        assert response.headers["X-Language"] == "en"
        assert response.headers["X-Voice"] == "p225"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_mp3(client, mock_synthesizer):
    """Test synthesize endpoint with MP3 format."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        response = client.post(
            "/synthesize",
            json={
                "text": "Hello, world!",
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.0,
                    "format": "mp3"
                }
            }
        )
        assert response.status_code == 200
        assert response.content == b"mock audio data"
        assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_ogg(client, mock_synthesizer):
    """Test synthesize endpoint with OGG format."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        response = client.post(
            "/synthesize",
            json={
                "text": "Hello, world!",
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.0,
                    "format": "ogg"
                }
            }
        )
        assert response.status_code == 200
        assert response.content == b"mock audio data"
        assert response.headers["content-type"] == "audio/ogg"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_custom_parameters(client, mock_synthesizer):
    """Test synthesize endpoint with custom parameters."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        response = client.post(
            "/synthesize",
            json={
                "text": "Hello, world!",
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.2,
                    "pitch": 0.9,
                    "energy": 1.1,
                    "emotion": "happy",
                    "style": "formal",
                    "format": "wav",
                    "sample_rate": 44100
                }
            }
        )
        assert response.status_code == 200
        assert response.content == b"mock audio data"
        
        # Verify the synthesizer was called with the right parameters
        options = mock_synthesizer.synthesize.call_args[1]["options"]
        assert options.language == "en"
        assert options.voice == "p225"
        assert options.speed == 1.2
        assert options.pitch == 0.9
        assert options.energy == 1.1
        assert options.emotion == "happy"
        assert options.style == "formal"
        assert options.format == "wav"
        assert options.sample_rate == 44100


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_error(client):
    """Test synthesize endpoint error."""
    with patch("src.api.tts.synthesizer.synthesize", side_effect=Exception("Test error")):
        response = client.post(
            "/synthesize",
            json={
                "text": "Hello, world!",
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.0,
                    "format": "wav"
                }
            }
        )
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
def test_batch_synthesize_endpoint(client, mock_synthesizer):
    """Test batch_synthesize endpoint."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        response = client.post(
            "/batch_synthesize",
            json={
                "texts": ["Hello, world!", "This is a test."],
                "options": {
                    "language": "en",
                    "voice": "p225",
                    "speed": 1.0,
                    "format": "wav"
                }
            }
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "X-Processing-Time" in response.headers
        assert response.headers["X-Language"] == "en"
        assert response.headers["X-Voice"] == "p225"
        
        # Verify the ZIP file contains the expected files
        zip_buffer = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            assert len(zip_file.namelist()) == 2
            assert "audio_1.wav" in zip_file.namelist()
            assert "audio_2.wav" in zip_file.namelist()
            
            # Verify the file contents
            for filename in zip_file.namelist():
                with zip_file.open(filename) as file:
                    assert file.read() == b"mock audio data"


@pytest.mark.unit
@pytest.mark.api
def test_async_synthesis_endpoint(client, mock_synthesizer):
    """Test async synthesis endpoint."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        # Clear the jobs dictionary
        with patch("src.api.tts.synthesis_jobs", {}):
            # Submit the job
            response = client.post(
                "/synthesize/async",
                json={
                    "text": "Hello, world!",
                    "options": {
                        "language": "en",
                        "voice": "p225",
                        "speed": 1.0,
                        "format": "wav"
                    }
                }
            )
            assert response.status_code == 202
            assert "job_id" in response.json()
            assert response.json()["status"] == "pending"
            
            # Get the job ID
            job_id = response.json()["job_id"]
            
            # Manually set the job result instead of calling the async function
            from src.api.tts import synthesis_jobs
            synthesis_jobs[job_id]["status"] = "completed"
            synthesis_jobs[job_id]["result"] = b"mock audio data"
            
            # Check the status
            response = client.get(f"/synthesize/status/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "completed"
            
            # Get the result
            response = client.get(f"/synthesize/result/{job_id}")
            assert response.status_code == 200
            assert response.content == b"mock audio data"
            assert response.headers["content-type"] == "audio/wav"


@pytest.mark.unit
@pytest.mark.api
def test_async_batch_synthesis_endpoint(client, mock_synthesizer):
    """Test async batch synthesis endpoint."""
    with patch("src.api.tts.synthesizer", mock_synthesizer):
        # Clear the jobs dictionary
        with patch("src.api.tts.synthesis_jobs", {}):
            # Submit the job
            response = client.post(
                "/batch_synthesize/async",
                json={
                    "texts": ["Hello, world!", "This is a test."],
                    "options": {
                        "language": "en",
                        "voice": "p225",
                        "speed": 1.0,
                        "format": "wav"
                    }
                }
            )
            assert response.status_code == 202
            assert "job_id" in response.json()
            assert response.json()["status"] == "pending"
            
            # Get the job ID
            job_id = response.json()["job_id"]
            
            # Create a ZIP file in memory for testing
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr("audio_1.wav", b"mock audio data")
                zip_file.writestr("audio_2.wav", b"mock audio data")
            
            # Manually set the job result instead of calling the async function
            from src.api.tts import synthesis_jobs
            synthesis_jobs[job_id]["status"] = "completed"
            synthesis_jobs[job_id]["result"] = zip_buffer.getvalue()
            synthesis_jobs[job_id]["is_batch"] = True
            
            # Check the status
            response = client.get(f"/synthesize/status/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] == "completed"
            
            # Get the result
            response = client.get(f"/batch_synthesize/result/{job_id}")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/zip"
            
            # Verify the ZIP file contains the expected files
            zip_buffer = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
                assert len(zip_file.namelist()) == 2
                assert "audio_1.wav" in zip_file.namelist()
                assert "audio_2.wav" in zip_file.namelist()


@pytest.mark.unit
@pytest.mark.api
def test_voices_by_language_endpoint(client):
    """Test voices_by_language endpoint."""
    with patch("src.api.tts.model_manager.list_available_voices", return_value=["p225", "p226", "p227"]):
        # Test English voices
        response = client.get("/voices/en")
        assert response.status_code == 200
        assert "voices" in response.json()
        
        # Test French voices (should return a subset or empty list)
        response = client.get("/voices/fr")
        assert response.status_code == 200
        assert "voices" in response.json()


@pytest.mark.unit
@pytest.mark.api
def test_get_config_endpoint(client):
    """Test get_config endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    assert "model" in response.json()
    assert "server" in response.json() 