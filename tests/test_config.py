"""
Tests for configuration module.
"""
import pytest
from pydantic import ValidationError

from src.config import TTSModelConfig, SynthesisOptions, ServerConfig, AppConfig


@pytest.mark.unit
@pytest.mark.config
def test_tts_model_config_defaults():
    """Test TTSModelConfig defaults."""
    config = TTSModelConfig()
    assert config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
    assert config.device == "cpu"
    assert config.compute_type == "float32"
    assert config.cpu_threads == 4
    assert config.num_workers == 1
    assert "models" in config.download_root  # Check that it contains 'models' instead of hardcoding the path


@pytest.mark.unit
@pytest.mark.config
def test_tts_model_config_validation():
    """Test TTSModelConfig validation."""
    # Valid config
    config = TTSModelConfig(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        device="cuda",
        compute_type="float16",
        cpu_threads=8,
        num_workers=2,
        download_root="/data/models",
    )
    assert config.model_name == "tts_models/en/ljspeech/tacotron2-DDC"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.cpu_threads == 8
    assert config.num_workers == 2
    assert config.download_root == "/data/models"
    
    # Invalid model name
    with pytest.raises(ValidationError):
        TTSModelConfig(model_name="")
    
    # Invalid device
    with pytest.raises(ValidationError):
        TTSModelConfig(device="gpu")
    
    # Invalid compute type
    with pytest.raises(ValidationError):
        TTSModelConfig(compute_type="float64")


@pytest.mark.unit
@pytest.mark.config
def test_synthesis_options_defaults():
    """Test SynthesisOptions defaults."""
    options = SynthesisOptions()
    assert options.language == "en"
    assert options.voice == "default"
    assert options.speed == 1.0
    assert options.pitch == 1.0
    assert options.energy == 1.0
    assert options.emotion is None
    assert options.format == "wav"
    assert options.sample_rate == 22050


@pytest.mark.unit
@pytest.mark.config
def test_synthesis_options_validation():
    """Test SynthesisOptions validation."""
    # Valid options
    options = SynthesisOptions(
        language="fr",
        voice="speaker_1",
        speed=1.5,
        pitch=0.8,
        energy=1.2,
        emotion="happy",
        format="mp3",
        sample_rate=44100,
    )
    assert options.language == "fr"
    assert options.voice == "speaker_1"
    assert options.speed == 1.5
    assert options.pitch == 0.8
    assert options.energy == 1.2
    assert options.emotion == "happy"
    assert options.format == "mp3"
    assert options.sample_rate == 44100
    
    # Invalid language
    with pytest.raises(ValidationError):
        SynthesisOptions(language="")
    
    # Invalid format
    with pytest.raises(ValidationError):
        SynthesisOptions(format="flac")
    
    # Invalid speed
    with pytest.raises(ValidationError):
        SynthesisOptions(speed=0.4)
    with pytest.raises(ValidationError):
        SynthesisOptions(speed=2.1) 