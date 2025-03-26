"""
Integration tests for the TTS service.
"""
import io
import pytest
from unittest.mock import patch, MagicMock, ANY
import numpy as np
from fastapi.testclient import TestClient

from src.main import app
from src.models.tts_model import TTSModelManager, TTSSynthesizer
from src.config import SynthesisOptions


class OptionsMatcher:
    """Custom matcher for SynthesisOptions."""
    
    def __init__(self, expected_options):
        self.expected_options = expected_options
    
    def __eq__(self, other):
        """Check if the provided options match the expected ones."""
        if not isinstance(other, SynthesisOptions):
            return False
        
        # Check the fields we care about
        for key, value in self.expected_options.items():
            if getattr(other, key) != value:
                return False
        
        return True
    
    def __repr__(self):
        """String representation for error messages."""
        return f"OptionsMatcher({self.expected_options})"


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_tts_api():
    """Mock the TTS API components."""
    # Create mock synthesizer
    synthesizer_mock = MagicMock()
    
    # Mock the synthesize method
    def mock_synthesize(text, speaker, language):
        # Return a numpy array as mock audio data
        return np.zeros(1000, dtype=np.float32)
    
    synthesizer_mock.synthesize.side_effect = mock_synthesize
    
    # Mock the _convert_audio method
    def mock_convert_audio(wav, format, sample_rate):
        # Return different mock data based on format
        return f"mock {format} data".encode()
    
    synthesizer_mock._convert_audio.side_effect = mock_convert_audio
    
    # Create mock model manager
    model_manager_mock = MagicMock()
    model_manager_mock.list_available_voices.return_value = ["p225", "p226", "p227"]
    model_manager_mock.list_available_languages.return_value = ["en", "fr", "de"]
    
    # Patch the synthesizer and model_manager in the TTS API module
    with patch("src.api.tts.synthesizer", synthesizer_mock):
        with patch("src.api.tts.model_manager", model_manager_mock):
            yield {
                "synthesizer": synthesizer_mock,
                "model_manager": model_manager_mock
            }


@pytest.mark.integration
def test_health_endpoint_integration(client):
    """Integration test for health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.integration
def test_voices_endpoint_integration(client, mock_tts_api):
    """Integration test for voices endpoint."""
    response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert data["voices"] == ["p225", "p226", "p227"]
    
    # Verify the model manager was called
    mock_tts_api["model_manager"].list_available_voices.assert_called_once()


@pytest.mark.integration
def test_languages_endpoint_integration(client, mock_tts_api):
    """Integration test for languages endpoint."""
    response = client.get("/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert data["languages"] == ["en", "fr", "de"]
    
    # Verify the model manager was called
    mock_tts_api["model_manager"].list_available_languages.assert_called_once()


@pytest.mark.integration
def test_synthesize_endpoint_integration(client, mock_tts_api):
    """Integration test for synthesize endpoint."""
    request_data = {
        "text": "Hello, this is an integration test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock wav data"
    assert response.headers["content-type"] == "audio/wav"
    assert "X-Processing-Time" in response.headers
    assert response.headers["X-Language"] == "en"
    assert response.headers["X-Voice"] == "p225"
    
    # Verify the synthesizer was called with correct parameters
    mock_tts_api["synthesizer"].synthesize.assert_called_with(
        text="Hello, this is an integration test.",
        speaker=request_data["options"]["voice"],
        language=request_data["options"]["language"]
    )


@pytest.mark.integration
def test_synthesize_endpoint_with_different_formats(client, mock_tts_api):
    """Integration test for synthesize endpoint with different formats."""
    # Test with MP3 format
    request_data = {
        "text": "Hello, this is an integration test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.0,
            "format": "mp3"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock mp3 data"
    assert response.headers["content-type"] == "audio/mpeg"
    
    # Test with OGG format
    request_data["options"]["format"] = "ogg"
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.content == b"mock ogg data"
    assert response.headers["content-type"] == "audio/ogg"


@pytest.mark.integration
def test_synthesize_endpoint_with_different_voices(client, mock_tts_api):
    """Integration test for synthesize endpoint with different voices."""
    # Test with p226 voice
    request_data = {
        "text": "Hello, this is an integration test.",
        "options": {
            "language": "en",
            "voice": "p226",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.headers["X-Voice"] == "p226"
    
    # Verify the synthesizer was called with correct parameters
    mock_tts_api["synthesizer"].synthesize.assert_called_with(
        text="Hello, this is an integration test.",
        speaker=request_data["options"]["voice"],
        language=request_data["options"]["language"]
    )


@pytest.mark.integration
def test_synthesize_endpoint_with_different_languages(client, mock_tts_api):
    """Integration test for synthesize endpoint with different languages."""
    # Test with French language
    request_data = {
        "text": "Bonjour, ceci est un test d'intégration.",
        "options": {
            "language": "fr",
            "voice": "p225",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    assert response.headers["X-Language"] == "fr"
    
    # Verify the synthesizer was called with correct parameters
    mock_tts_api["synthesizer"].synthesize.assert_called_with(
        text="Bonjour, ceci est un test d'intégration.",
        speaker=request_data["options"]["voice"],
        language=request_data["options"]["language"]
    )


@pytest.mark.integration
def test_synthesize_endpoint_with_different_speeds(client, mock_tts_api):
    """Integration test for synthesize endpoint with different speeds."""
    # Test with speed 1.5
    request_data = {
        "text": "Hello, this is an integration test.",
        "options": {
            "language": "en",
            "voice": "p225",
            "speed": 1.5,
            "format": "wav"
        }
    }
    
    response = client.post("/synthesize", json=request_data)
    
    assert response.status_code == 200
    
    # Verify the synthesizer was called with correct parameters
    mock_tts_api["synthesizer"].synthesize.assert_called_with(
        text="Hello, this is an integration test.",
        speaker=request_data["options"]["voice"],
        language=request_data["options"]["language"]
    )


@pytest.mark.integration
def test_config_endpoint_integration(client):
    """Integration test for config endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "server" in data
    assert "model_name" in data["model"]
    assert "host" in data["server"] 