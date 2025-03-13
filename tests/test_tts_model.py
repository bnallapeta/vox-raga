"""
Tests for TTS model.
"""
import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, call

from src.models.tts_model import TTSModelManager, TTSSynthesizer
from src.config import TTSModelConfig, SynthesisOptions


@pytest.fixture
def mock_model_manager():
    """Mock the ModelManager class."""
    with patch("src.models.tts_model.ModelManager") as mock:
        instance = mock.return_value
        
        # Mock download_model method
        instance.download_model.return_value = (
            "/path/to/model.pth",
            "/path/to/config.json",
            {"default_vocoder": "vocoder_model"}
        )
        
        # Mock list_tts_models method
        instance.list_tts_models.return_value = [
            {"name": "model1", "language": "en"},
            {"name": "model2", "language": "fr"}
        ]
        
        yield instance


@pytest.fixture
def mock_synthesizer():
    """Mock the Synthesizer class."""
    with patch("src.models.tts_model.Synthesizer") as mock:
        instance = mock.return_value
        
        # Mock tts method
        instance.tts.return_value = np.zeros(22050)  # 1 second of silence at 22050Hz
        
        # Mock speaker_manager
        instance.tts_model = MagicMock()
        instance.tts_model.speaker_manager = MagicMock()
        instance.tts_model.speaker_manager.speaker_names = ["p225", "p226", "p227"]
        
        # Mock language_manager
        instance.tts_model.language_manager = MagicMock()
        instance.tts_model.language_manager.language_names = ["en", "fr", "de"]
        
        yield instance


@pytest.mark.unit
@pytest.mark.model
def test_tts_model_manager_initialization(mock_model_manager):
    """Test TTSModelManager initialization."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    manager = TTSModelManager(config)
    
    assert manager.config == config
    assert manager.model_manager == mock_model_manager
    assert manager.models == {}
    assert manager.default_model is None
    assert manager._initialized is True


@pytest.mark.unit
@pytest.mark.model
def test_tts_model_manager_singleton(mock_model_manager):
    """Test TTSModelManager singleton pattern."""
    config1 = TTSModelConfig(model_name="model1")
    config2 = TTSModelConfig(model_name="model2")
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    manager1 = TTSModelManager(config1)
    manager2 = TTSModelManager(config2)
    
    # Both instances should be the same object
    assert manager1 is manager2
    
    # The config should be from the first initialization
    assert manager1.config == config1
    assert manager2.config == config1  # Not config2


@pytest.mark.unit
@pytest.mark.model
def test_get_model(mock_model_manager, mock_synthesizer):
    """Test get_model method."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.Synthesizer", return_value=mock_synthesizer):
        manager = TTSModelManager(config)
        
        # First call should download and cache the model
        model = manager.get_model()
        
        assert model == mock_synthesizer
        assert manager.models["tts_models/en/vctk/vits"] == mock_synthesizer
        assert manager.default_model == mock_synthesizer
        
        # Second call should return the cached model
        model2 = manager.get_model()
        
        assert model2 == mock_synthesizer
        assert mock_model_manager.download_model.call_count == 1


@pytest.mark.unit
@pytest.mark.model
def test_list_available_voices(mock_model_manager, mock_synthesizer):
    """Test list_available_voices method."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.Synthesizer", return_value=mock_synthesizer):
        manager = TTSModelManager(config)
        
        voices = manager.list_available_voices()
        
        assert voices == ["p225", "p226", "p227"]


@pytest.mark.unit
@pytest.mark.model
def test_list_available_languages(mock_model_manager, mock_synthesizer):
    """Test list_available_languages method."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.Synthesizer", return_value=mock_synthesizer):
        manager = TTSModelManager(config)
        
        languages = manager.list_available_languages()
        
        assert languages == ["en", "fr", "de"]


@pytest.mark.unit
@pytest.mark.model
def test_tts_synthesizer_initialization(mock_model_manager, mock_synthesizer):
    """Test TTSSynthesizer initialization."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.TTSModelManager", return_value=MagicMock()) as mock_manager_class:
        synthesizer = TTSSynthesizer(config)
        
        assert synthesizer.model_config == config
        mock_manager_class.assert_called_once_with(config)


@pytest.mark.unit
@pytest.mark.model
def test_synthesize(mock_model_manager, mock_synthesizer):
    """Test synthesize method."""
    config = TTSModelConfig(
        model_name="tts_models/en/vctk/vits",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    options = SynthesisOptions(
        language="en",
        voice="p225",
        speed=1.0,
        format="wav"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    with patch("src.models.tts_model.TTSModelManager") as mock_manager_class:
        # Set up the mock manager
        mock_manager = mock_manager_class.return_value
        mock_manager.get_model.return_value = mock_synthesizer
        
        # Create the synthesizer
        synthesizer = TTSSynthesizer(config)
        
        # Mock the _convert_audio method
        with patch.object(synthesizer, "_convert_audio", return_value=b"mock audio data"):
            # Call the synthesize method
            result = synthesizer.synthesize("Hello, this is a test.", options)
            
            # Verify the result
            assert result == b"mock audio data"
            
            # Verify the TTS model was called with correct parameters
            mock_synthesizer.tts.assert_called_once_with(
                text="Hello, this is a test.",
                speaker_name="p225",
                language_name="en",
                speed=1.0
            )


@pytest.mark.unit
@pytest.mark.model
def test_convert_audio():
    """Test _convert_audio method."""
    config = TTSModelConfig()
    
    # Create a simple audio array (1 second of silence at 22050Hz)
    wav = np.zeros(22050, dtype=np.float32)
    
    with patch("src.models.tts_model.TTSModelManager"):
        synthesizer = TTSSynthesizer(config)
        
        # Test WAV format
        with patch("src.models.tts_model.sf") as mock_sf:
            # Mock the write method to capture the buffer
            def mock_write(buffer, data, sample_rate, format):
                buffer.write(b"mock wav data")
            
            mock_sf.write.side_effect = mock_write
            
            result = synthesizer._convert_audio(wav, "wav", 22050)
            
            assert result == b"mock wav data"
            mock_sf.write.assert_called_once()
            
        # Test MP3 format
        with patch("src.models.tts_model.sf") as mock_sf:
            # Mock the write method to capture the buffer
            def mock_write(buffer, data, sample_rate, format):
                buffer.write(b"mock mp3 data")
            
            mock_sf.write.side_effect = mock_write
            
            result = synthesizer._convert_audio(wav, "mp3", 22050)
            
            assert result == b"mock mp3 data"
            mock_sf.write.assert_called_once()
            
        # Test OGG format
        with patch("src.models.tts_model.sf") as mock_sf:
            # Mock the write method to capture the buffer
            def mock_write(buffer, data, sample_rate, format):
                buffer.write(b"mock ogg data")
            
            mock_sf.write.side_effect = mock_write
            
            result = synthesizer._convert_audio(wav, "ogg", 22050)
            
            assert result == b"mock ogg data"
            mock_sf.write.assert_called_once() 