"""
Tests for TTS model module.
"""
import io
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY, call

from src.config import TTSModelConfig, SynthesisOptions
from src.models.tts_model import TTSModelManager, TTSSynthesizer


@pytest.fixture
def mock_model_manager():
    """Mock model manager."""
    mock = MagicMock()
    # Return a tuple of (model_path, config_path, model_item)
    mock.download_model.return_value = ("/path/to/model", "/path/to/config", {"default_vocoder": "vocoder_model"})
    return mock


@pytest.fixture
def mock_synthesizer():
    """Mock synthesizer."""
    mock = MagicMock()
    mock.tts.return_value = (np.zeros(22050, dtype=np.float32), 22050)
    # Add speaker and language attributes
    mock.speakers = ["p225", "p226", "p227"]
    mock.languages = ["en", "fr", "de"]
    return mock


@pytest.mark.unit
@pytest.mark.model
def test_model_manager_singleton():
    """Test TTSModelManager singleton pattern."""
    config1 = TTSModelConfig()
    config2 = TTSModelConfig(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create two instances
    manager1 = TTSModelManager(config1)
    manager2 = TTSModelManager(config2)
    
    # Verify they are the same instance
    assert manager1 is manager2
    
    # Verify the config is from the first initialization
    assert manager1.config.model_name == "tts_models/en/vctk/vits"


@pytest.mark.unit
@pytest.mark.model
def test_model_manager_init():
    """Test TTSModelManager initialization."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.ModelManager") as mock_model_manager_class:
        with patch("os.path.isfile", return_value=False):
            with patch("os.makedirs") as mock_makedirs:
                with patch("builtins.open", MagicMock()):
                    manager = TTSModelManager(config)
                    
                    # Verify the config was set
                    assert manager.config == config
                    
                    # Verify the model manager was created
                    mock_model_manager_class.assert_called_once()
                    
                    # Verify the directory was created
                    mock_makedirs.assert_called_once()


@pytest.mark.unit
@pytest.mark.model
def test_get_model():
    """Test get_model method."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create a mock for ModelManager that returns the expected tuple
    mock_model_manager = MagicMock()
    mock_model_manager.download_model.return_value = (
        "/path/to/model", 
        "/path/to/config", 
        {"default_vocoder": "vocoder_model"}
    )
    
    # Create a mock for Synthesizer
    mock_synthesizer = MagicMock()
    
    with patch("src.models.tts_model.ModelManager", return_value=mock_model_manager):
        with patch("src.models.tts_model.Synthesizer", return_value=mock_synthesizer):
            # Create the manager
            manager = TTSModelManager(config)
            
            # Clear the models cache to ensure we test the get_model method properly
            manager.models = {}
            manager.default_model = None
            
            # First call should download and cache the model
            model = manager.get_model()
            
            # Verify the model was downloaded and cached
            assert model == mock_synthesizer
            assert manager.models["tts_models/en/vctk/vits"] == mock_synthesizer
            assert manager.default_model == mock_synthesizer
            
            # Verify download_model was called with both the TTS model and vocoder model
            # The first call should be for the TTS model
            assert mock_model_manager.download_model.call_count >= 2
            assert call("tts_models/en/vctk/vits") in mock_model_manager.download_model.call_args_list
            assert call("vocoder_model") in mock_model_manager.download_model.call_args_list
            
            # Store the current call count
            call_count_before_second_call = mock_model_manager.download_model.call_count
            
            # Second call should return the cached model without downloading again
            model2 = manager.get_model()
            
            # Verify the same model was returned
            assert model2 == mock_synthesizer
            
            # Verify download_model was not called again
            assert mock_model_manager.download_model.call_count == call_count_before_second_call


@pytest.mark.unit
@pytest.mark.model
def test_get_model_with_name():
    """Test get_model method with specific model name."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create a mock for ModelManager that returns the expected tuple
    mock_model_manager = MagicMock()
    mock_model_manager.download_model.return_value = (
        "/path/to/model", 
        "/path/to/config", 
        {"default_vocoder": "vocoder_model"}
    )
    
    # Create mocks for Synthesizer
    mock_synthesizer1 = MagicMock()
    mock_synthesizer2 = MagicMock()
    
    with patch("src.models.tts_model.ModelManager", return_value=mock_model_manager):
        with patch("src.models.tts_model.Synthesizer", side_effect=[mock_synthesizer1, mock_synthesizer2]):
            # Create the manager
            manager = TTSModelManager(config)
            
            # Clear the models cache to ensure we test the get_model method properly
            manager.models = {}
            manager.default_model = None
            
            # First call with default model
            model1 = manager.get_model()
            
            # Verify the first model was downloaded and cached
            assert model1 == mock_synthesizer1
            assert manager.models["tts_models/en/vctk/vits"] == mock_synthesizer1
            
            # Verify download_model was called with the default model name
            mock_model_manager.download_model.assert_any_call("tts_models/en/vctk/vits")
            
            # Call with different model
            model2 = manager.get_model("tts_models/en/ljspeech/tacotron2-DDC")
            
            # Verify the second model was downloaded and cached
            assert model2 == mock_synthesizer2
            assert manager.models["tts_models/en/ljspeech/tacotron2-DDC"] == mock_synthesizer2
            
            # Verify download_model was called with the second model name
            mock_model_manager.download_model.assert_any_call("tts_models/en/ljspeech/tacotron2-DDC")
            
            # Verify both models are cached
            assert len(manager.models) == 2
            assert "tts_models/en/vctk/vits" in manager.models
            assert "tts_models/en/ljspeech/tacotron2-DDC" in manager.models


@pytest.mark.unit
@pytest.mark.model
def test_list_available_models():
    """Test list_available_models method."""
    config = TTSModelConfig()
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create a mock for ModelManager
    mock_model_manager = MagicMock()
    # Use list_tts_models instead of list_models
    mock_model_manager.list_tts_models.return_value = [
        {"model_name": "tts_models/en/vctk/vits"},
        {"model_name": "tts_models/en/ljspeech/tacotron2-DDC"}
    ]
    
    with patch("src.models.tts_model.ModelManager", return_value=mock_model_manager):
        manager = TTSModelManager(config)
        
        models = manager.list_available_models()
        
        assert len(models) == 2
        assert models[0]["model_name"] == "tts_models/en/vctk/vits"
        assert models[1]["model_name"] == "tts_models/en/ljspeech/tacotron2-DDC"


@pytest.mark.unit
@pytest.mark.model
def test_list_available_voices():
    """Test list_available_voices method."""
    config = TTSModelConfig()
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create a mock for Synthesizer with the correct structure
    mock_synthesizer = MagicMock()
    # The actual implementation uses tts_model.speaker_manager.speaker_names
    mock_synthesizer.tts_model = MagicMock()
    mock_synthesizer.tts_model.speaker_manager = MagicMock()
    mock_synthesizer.tts_model.speaker_manager.speaker_names = ["p225", "p226", "p227"]
    
    with patch("src.models.tts_model.ModelManager"):
        with patch.object(TTSModelManager, "get_model", return_value=mock_synthesizer):
            manager = TTSModelManager(config)
            
            voices = manager.list_available_voices()
            
            assert voices == ["p225", "p226", "p227"]


@pytest.mark.unit
@pytest.mark.model
def test_list_available_languages():
    """Test list_available_languages method."""
    config = TTSModelConfig()
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create a mock for Synthesizer with the correct structure
    mock_synthesizer = MagicMock()
    # The actual implementation uses tts_model.language_manager.language_names
    mock_synthesizer.tts_model = MagicMock()
    mock_synthesizer.tts_model.language_manager = MagicMock()
    mock_synthesizer.tts_model.language_manager.language_names = ["en", "fr", "de"]
    
    with patch("src.models.tts_model.ModelManager"):
        with patch.object(TTSModelManager, "get_model", return_value=mock_synthesizer):
            manager = TTSModelManager(config)
            
            languages = manager.list_available_languages()
            
            assert languages == ["en", "fr", "de"]


@pytest.mark.unit
@pytest.mark.model
def test_synthesizer_init():
    """Test TTSSynthesizer initialization."""
    config = TTSModelConfig()
    
    with patch("src.models.tts_model.TTSModelManager") as mock_manager_class:
        synthesizer = TTSSynthesizer(config)
        
        assert synthesizer.model_config == config
        mock_manager_class.assert_called_once_with(config)


@pytest.mark.unit
@pytest.mark.model
def test_synthesize():
    """Test synthesize method."""
    config = TTSModelConfig()
    options = SynthesisOptions(
        language="en",
        voice="p225",
        speed=1.0,
        format="wav"
    )
    text = "Hello, world!"
    
    # Create a mock for TTSModelManager
    mock_manager = MagicMock()
    mock_model = MagicMock()
    mock_manager.get_model.return_value = mock_model
    
    # Set up the mock model to return audio data
    mock_model.tts.return_value = (np.zeros(22050, dtype=np.float32), 22050)
    
    with patch("src.models.tts_model.TTSModelManager", return_value=mock_manager):
        with patch.object(TTSSynthesizer, "_convert_audio", return_value=b"audio data"):
            with patch("time.time", side_effect=[0, 1]):  # Mock time.time to return 0 then 1
                synthesizer = TTSSynthesizer(config)
                
                # Call the method
                audio_data = synthesizer.synthesize(text, options)
                
                # Verify the result
                assert audio_data == b"audio data"
                
                # Verify the model was called correctly with the right parameter names
                mock_model.tts.assert_called_once_with(
                    text=text,
                    speaker_name=options.voice,
                    language_name=options.language,
                    speed=options.speed
                )
                
                # Verify _convert_audio was called correctly
                synthesizer._convert_audio.assert_called_once()


@pytest.mark.unit
@pytest.mark.model
def test_convert_audio():
    """Test _convert_audio method."""
    config = TTSModelConfig()
    
    # Create a simple audio array (1 second of silence at 22050Hz)
    wav = np.zeros(22050, dtype=np.float32)
    
    with patch("src.models.tts_model.TTSModelManager"):
        # Mock the soundfile module
        with patch.dict("sys.modules", {"soundfile": MagicMock()}):
            # Import soundfile inside the patch context
            import sys
            mock_sf = sys.modules["soundfile"]
            
            # Configure the mock
            mock_sf.write = MagicMock()
            
            # Create a mock for BytesIO
            mock_buffer = MagicMock(spec=io.BytesIO)
            mock_buffer.read.return_value = b"mock audio data"
            
            with patch("io.BytesIO", return_value=mock_buffer):
                synthesizer = TTSSynthesizer(config)
                
                # Test WAV format
                audio_data = synthesizer._convert_audio(wav, "wav", 22050)
                
                # Verify soundfile.write was called correctly
                mock_sf.write.assert_called_once_with(mock_buffer, wav, 22050, format="wav")
                
                # Verify the buffer was read
                mock_buffer.seek.assert_called_once_with(0)
                mock_buffer.read.assert_called_once()
                
                # Verify the result
                assert audio_data == b"mock audio data"


# Import time for the test_synthesize function
import time 