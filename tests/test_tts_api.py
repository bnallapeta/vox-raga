"""
Tests for TTS API endpoints.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app
from src.config import SynthesisOptions


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_synthesizer():
    """Mock TTS synthesizer."""
    with patch("src.api.tts.synthesizer") as mock:
        # Mock the synthesize method to return a simple WAV file
        mock.synthesize.return_value = b"mock audio data"
        yield mock


@pytest.fixture
def mock_model_manager():
    """Mock TTS model manager."""
    with patch("src.api.tts.model_manager") as mock:
        # Mock the list_available_voices method
        mock.list_available_voices.return_value = ["p225", "p226", "p227"]
        # Mock the list_available_languages method
        mock.list_available_languages.return_value = ["en", "fr", "de"]
        yield mock


@pytest.mark.unit
@pytest.mark.api
def test_list_voices_endpoint(client, mock_model_manager):
    """Test list voices endpoint."""
    response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert data["voices"] == ["p225", "p226", "p227"]


@pytest.mark.unit
@pytest.mark.api
def test_list_languages_endpoint(client, mock_model_manager):
    """Test list languages endpoint."""
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert data["languages"] == ["en", "fr", "de"]


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_success(client, mock_synthesizer):
    """Test synthesize endpoint with successful synthesis."""
    request_data = {
        "text": "Hello, this is a test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock audio data"
    assert response.headers["content-type"] == "audio/wav"
    
    # Verify the synthesizer was called with correct parameters
    mock_synthesizer.synthesize.assert_called_once()
    call_args = mock_synthesizer.synthesize.call_args[1]
    assert call_args["text"] == "Hello, this is a test."
    assert isinstance(call_args["options"], SynthesisOptions)
    assert call_args["options"].language == "en"
    assert call_args["options"].voice == "p225"
    assert call_args["options"].speed == 1.0
    assert call_args["options"].format == "wav"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_mp3(client, mock_synthesizer):
    """Test synthesize endpoint with MP3 format."""
    request_data = {
        "text": "Hello, this is a test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "mp3"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock audio data"
    assert response.headers["content-type"] == "audio/mpeg"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_ogg(client, mock_synthesizer):
    """Test synthesize endpoint with OGG format."""
    request_data = {
        "text": "Hello, this is a test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "ogg"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock audio data"
    assert response.headers["content-type"] == "audio/ogg"


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_with_custom_parameters(client, mock_synthesizer):
    """Test synthesize endpoint with custom parameters."""
    request_data = {
        "text": "Hello, this is a test.",
        "options": {
            "language": "fr",
            "voice": "p226",
            "speed": 1.2,
            "pitch": 0.9,
            "energy": 1.1,
            "format": "wav",
            "sample_rate": 44100
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    
    # Verify the synthesizer was called with correct parameters
    call_args = mock_synthesizer.synthesize.call_args[1]
    assert call_args["options"].language == "fr"
    assert call_args["options"].voice == "p226"
    assert call_args["options"].speed == 1.2
    assert call_args["options"].pitch == 0.9
    assert call_args["options"].energy == 1.1
    assert call_args["options"].format == "wav"
    assert call_args["options"].sample_rate == 44100


@pytest.mark.unit
@pytest.mark.api
def test_synthesize_endpoint_error(client, mock_synthesizer):
    """Test synthesize endpoint with an error during synthesis."""
    # Make the synthesizer raise an exception
    mock_synthesizer.synthesize.side_effect = Exception("Synthesis error")
    
    request_data = {
        "text": "Hello, this is a test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Synthesis error" in data["detail"]


@pytest.mark.unit
@pytest.mark.api
def test_get_config_endpoint(client):
    """Test get config endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "server" in data 