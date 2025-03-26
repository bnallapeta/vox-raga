"""
Tests for TTS model module.
"""
import os
import pytest
import numpy as np
import io
from unittest.mock import patch, MagicMock, ANY, call
from contextlib import ExitStack
import torch

from src.config import TTSModelConfig, SynthesisOptions
from src.models.tts_model import TTSModelManager
from src.tts import TTSSynthesizer


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
    assert manager1.config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"


@pytest.mark.unit
@pytest.mark.model
def test_model_manager_init():
    """Test model manager initialization."""
    config = TTSModelConfig()

    # Reset the singleton instance
    TTSModelManager._instance = None

    # Create mocks
    with patch("os.path.exists") as mock_exists, \
         patch("os.listdir") as mock_listdir, \
         patch("os.makedirs") as mock_makedirs:
        
        # Configure mocks
        mock_exists.return_value = False
        mock_listdir.return_value = []

        # Initialize the manager
        manager = TTSModelManager(config)

        # Verify the manager was initialized correctly
        assert manager.config == config
        assert isinstance(manager.models, dict)
        assert manager.default_model is None
        assert manager._initialized is True

        # Verify makedirs is called when creating reference audio directory
        mock_makedirs.assert_called_once_with(
            os.path.join(config.download_root, "tts_models", "multilingual", "multi-dataset", "xtts_v2", "reference_audio"),
            exist_ok=True
        )


@pytest.mark.unit
@pytest.mark.model
def test_get_model():
    """Test get_model method."""
    config = TTSModelConfig()

    # Reset the singleton instance
    TTSModelManager._instance = None

    # Create a mock for Synthesizer
    mock_synthesizer = MagicMock()
    mock_synthesizer.tts_model = MagicMock()
    mock_synthesizer.tts_model.speaker_manager = MagicMock()
    mock_synthesizer.tts_model.speaker_manager.name_to_id = {"p225": 0, "p226": 1}

    # Create patches for file operations and Synthesizer
    patches = [
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("os.makedirs"),
        patch("TTS.utils.synthesizer.Synthesizer", return_value=mock_synthesizer),
        patch("torch.load", return_value={}),
        patch("src.models.tts_model.TTSModelManager.load_model", return_value=mock_synthesizer)
    ]

    # Only add the add_safe_globals patch if it exists in torch.serialization
    if hasattr(torch.serialization, 'add_safe_globals'):
        patches.append(patch("torch.serialization.add_safe_globals", return_value=None))

    with ExitStack() as stack:
        # Enter all patches
        for p in patches:
            stack.enter_context(p)

        manager = TTSModelManager(config)
        model = manager.get_model()

        # Verify we got the mock synthesizer
        assert model == mock_synthesizer

        # Verify the model was cached
        assert config.model_name in manager.models
        assert manager.models[config.model_name] == mock_synthesizer


@pytest.mark.unit
@pytest.mark.model
def test_get_model_with_name():
    """Test get_model method with specific model name."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    
    # Reset the singleton instance
    TTSModelManager._instance = None
    
    # Create mocks for Synthesizer
    mock_synthesizer1 = MagicMock()
    mock_synthesizer2 = MagicMock()
    
    # Create patches for file operations and Synthesizer
    patches = [
        patch("os.path.exists", return_value=True),
        patch("os.path.isdir", return_value=True),
        patch("os.path.isfile", return_value=True),
        patch("os.makedirs"),
        patch("TTS.utils.synthesizer.Synthesizer", side_effect=[mock_synthesizer1, mock_synthesizer2]),
        patch("torch.load", return_value={}),
        patch("src.models.tts_model.TTSModelManager.load_model", side_effect=[mock_synthesizer1, mock_synthesizer2])
    ]

    # Only add the add_safe_globals patch if it exists in torch.serialization
    if hasattr(torch.serialization, 'add_safe_globals'):
        patches.append(patch("torch.serialization.add_safe_globals", return_value=None))

    with ExitStack() as stack:
        # Enter all patches
        for p in patches:
            stack.enter_context(p)
        
        # Create the manager
        manager = TTSModelManager(config)
        
        # Clear the models cache to ensure we test the get_model method properly
        manager.models = {}
        manager.default_model = None
        
        # First call with default model
        model1 = manager.get_model()
        
        # Verify the first model was downloaded and cached
        assert model1 == mock_synthesizer1
        assert manager.models["tts_models/multilingual/multi-dataset/xtts_v2"] == mock_synthesizer1
        
        # Call with different model
        model2 = manager.get_model("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Verify the second model was downloaded and cached
        assert model2 == mock_synthesizer2
        assert manager.models["tts_models/en/ljspeech/tacotron2-DDC"] == mock_synthesizer2
        
        # Verify both models are cached
        assert len(manager.models) == 2
        assert "tts_models/multilingual/multi-dataset/xtts_v2" in manager.models


@pytest.mark.unit
@pytest.mark.model
def test_list_available_models():
    """Test list_available_models method."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )

    # Reset the singleton instance
    TTSModelManager._instance = None

    # Create a mock directory structure
    mock_structure = {
        "/tmp/tts_models/tts_models": True,
        "/tmp/tts_models/tts_models/multilingual": True,
        "/tmp/tts_models/tts_models/multilingual/multi-dataset": True,
        "/tmp/tts_models/tts_models/multilingual/multi-dataset/xtts_v2": True,
        "/tmp/tts_models/tts_models/multilingual/multi-dataset/xtts_v2/model.pth": True,
        "/tmp/tts_models/tts_models/multilingual/multi-dataset/xtts_v2/config.json": True,
    }

    def mock_exists(path):
        return mock_structure.get(path, False)

    def mock_isdir(path):
        return mock_structure.get(path, False) and not path.endswith((".pth", ".json"))

    def mock_listdir(path):
        if path == "/tmp/tts_models/tts_models":
            return ["multilingual"]
        elif path == "/tmp/tts_models/tts_models/multilingual":
            return ["multi-dataset"]
        elif path == "/tmp/tts_models/tts_models/multilingual/multi-dataset":
            return ["xtts_v2"]
        elif path == "/tmp/tts_models/tts_models/multilingual/multi-dataset/xtts_v2":
            return ["model.pth", "config.json"]
        return []

    with patch("os.path.exists", side_effect=mock_exists), \
         patch("os.path.isdir", side_effect=mock_isdir), \
         patch("os.listdir", side_effect=mock_listdir), \
         patch("os.makedirs"):

        manager = TTSModelManager(config)
        models = manager.list_available_models()

        # Verify we got the expected model
        assert len(models) == 1
        assert models[0]["model_name"] == "tts_models/multilingual/multi-dataset/xtts_v2"
        assert models[0]["language"] == "multilingual"
        assert models[0]["dataset"] == "multi-dataset"
        assert models[0]["model_type"] == "xtts_v2"


class SimpleSpeakerManager:
    """A simple speaker manager for testing."""
    def __init__(self):
        self.name_to_id = {}


@pytest.mark.unit
@pytest.mark.model
def test_list_available_voices():
    """Test list_available_voices method."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )

    # Reset the singleton instance
    TTSModelManager._instance = None

    # Create a mock for Synthesizer with the correct structure
    mock_synthesizer = MagicMock()
    mock_synthesizer.tts_model = MagicMock()
    mock_synthesizer.tts_model.speaker_manager = SimpleSpeakerManager()
    mock_synthesizer.tts_model.speaker_manager.name_to_id = {
        "Claribel Dervla": 0,
        "Daisy Studious": 1,
        "Gracie Wise": 2,
        "Tammie Ema": 3,
        "Alison Dietlinde": 4,
        "Andrew Chipper": 5
    }

    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("os.makedirs"), \
         patch("TTS.utils.synthesizer.Synthesizer", return_value=mock_synthesizer):

        manager = TTSModelManager(config)
        voices = manager.list_available_voices()

        # Verify we got the expected voices
        assert set(voices) == {
            "Claribel Dervla", "Daisy Studious", "Gracie Wise",
            "Tammie Ema", "Alison Dietlinde", "Andrew Chipper"
        }


@pytest.mark.unit
@pytest.mark.model
def test_list_available_languages():
    """Test list_available_languages method."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )

    # Reset the singleton instance
    TTSModelManager._instance = None

    # Create a mock for Synthesizer with the correct structure
    mock_synthesizer = MagicMock()
    mock_synthesizer.tts_model = MagicMock()
    mock_synthesizer.tts_model.language_manager = MagicMock()
    mock_synthesizer.tts_model.language_manager.language_names = ["en"]

    with patch("os.path.exists", return_value=True), \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("os.makedirs"), \
         patch("TTS.utils.synthesizer.Synthesizer", return_value=mock_synthesizer):

        manager = TTSModelManager(config)
        languages = manager.list_available_languages()

        # Verify we got the expected languages
        assert languages == ["en"]


@pytest.mark.unit
@pytest.mark.model
def test_synthesizer_init():
    """Test TTSSynthesizer initialization."""
    config = TTSModelConfig()

    # Create a mock for TTSModelManager
    mock_manager = MagicMock()

    # Create the synthesizer with the correct arguments
    synthesizer = TTSSynthesizer(config, mock_manager)

    assert synthesizer.model_config == config
    assert synthesizer.model_manager == mock_manager


@pytest.mark.unit
@pytest.mark.model
def test_synthesize():
    """Test synthesize method."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    options = SynthesisOptions(
        language="en",
        voice="Claribel Dervla",
        speed=1.0,
        format="wav",
        emotion="neutral"
    )
    text = "Hello, world!"

    # Create a mock for TTSModelManager
    mock_manager = MagicMock()

    # Set up the mock manager to return audio data
    mock_audio = np.zeros(22050, dtype=np.float32)
    mock_manager.synthesize.return_value = mock_audio

    # Create the synthesizer with the correct arguments
    synthesizer = TTSSynthesizer(config, mock_manager)

    # Mock _convert_audio to return mock audio data
    synthesizer._convert_audio = MagicMock(return_value=b"mock audio data")

    # Call the method
    audio_data = synthesizer.synthesize(text, options)

    # Verify the result is bytes
    assert isinstance(audio_data, bytes)
    assert audio_data == b"mock audio data"

    # Verify the mock manager was called correctly
    mock_manager.synthesize.assert_called_once()
    call_args = mock_manager.synthesize.call_args[1]
    assert call_args["speaker"] == "Claribel Dervla"
    assert call_args["language"] == "en"
    assert "Say this in a neutral tone: Hello, world!" in call_args["text"]

    # Verify _convert_audio was called correctly
    synthesizer._convert_audio.assert_called_once_with(mock_audio, "wav", 22050)


@pytest.mark.unit
@pytest.mark.model
def test_convert_audio():
    """Test _convert_audio method."""
    config = TTSModelConfig()

    # Create a simple audio array (1 second of silence at 22050Hz)
    wav = np.zeros(22050, dtype=np.float32)

    # Create a mock for TTSModelManager
    mock_manager = MagicMock()

    # Mock the soundfile module
    with patch.dict("sys.modules", {"soundfile": MagicMock()}):
        # Import soundfile inside the patch context
        import sys
        mock_sf = sys.modules["soundfile"]

        # Configure the mock
        mock_sf.write = MagicMock()

        # Create a mock for BytesIO that behaves like a real file object
        mock_buffer = io.BytesIO()
        mock_buffer.write(b"mock audio data")

        with patch("io.BytesIO", return_value=mock_buffer):
            # Create the synthesizer with the correct arguments
            synthesizer = TTSSynthesizer(config, mock_manager)

            # Call the method
            audio_data = synthesizer._convert_audio(wav, "wav", 22050)
            
            # Verify the result
            assert audio_data == b"mock audio data"
            
            # Verify soundfile.write was called correctly
            mock_sf.write.assert_called_once_with(mock_buffer, wav, 22050, format="wav")


@pytest.mark.unit
@pytest.mark.model
def test_synthesize_with_emotion_and_style():
    """Test synthesize method with emotion and style parameters."""
    config = TTSModelConfig(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        device="cpu",
        download_root="/tmp/tts_models"
    )
    options = SynthesisOptions(
        language="en",
        voice="Claribel Dervla",
        speed=1.0,
        format="wav",
        emotion="happy",
        style="conversational"
    )
    text = "Hello, world!"

    # Create a mock for TTSModelManager
    mock_manager = MagicMock()

    # Set up the mock manager to return audio data
    mock_audio = np.zeros(22050, dtype=np.float32)
    mock_manager.synthesize.return_value = mock_audio

    # Create the synthesizer with the correct arguments
    synthesizer = TTSSynthesizer(config, mock_manager)

    # Mock _convert_audio to return mock audio data
    synthesizer._convert_audio = MagicMock(return_value=b"mock audio data")

    # Call the method
    audio_data = synthesizer.synthesize(text, options)

    # Verify the result is bytes
    assert isinstance(audio_data, bytes)
    assert audio_data == b"mock audio data"

    # Verify the mock manager was called correctly
    mock_manager.synthesize.assert_called_once()
    call_args = mock_manager.synthesize.call_args[1]
    assert call_args["speaker"] == "Claribel Dervla"
    assert call_args["language"] == "en"
    assert "Say this in a happy and cheerful tone:" in call_args["text"]
    assert "In a friendly conversational style:" in call_args["text"]
    assert "Hello, world!" in call_args["text"]

    # Verify _convert_audio was called correctly
    synthesizer._convert_audio.assert_called_once_with(mock_audio, "wav", 22050)


# Import time for the test_synthesize function
import time 