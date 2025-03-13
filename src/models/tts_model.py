"""
TTS model module for handling text-to-speech synthesis.
"""
import io
import os
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

from src.config import TTSModelConfig, SynthesisOptions
from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)


class TTSModelManager:
    """Manager for TTS models."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(TTSModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: TTSModelConfig):
        """Initialize the TTS model manager."""
        # Skip initialization if already initialized
        if self._initialized:
            return
        
        self.config = config
        # Create a path to models.json file instead of just the directory
        models_file = os.path.join(config.download_root, "models.json")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(models_file), exist_ok=True)
        # Create an empty models.json file if it doesn't exist
        if not os.path.isfile(models_file):
            with open(models_file, "w") as f:
                f.write("{}")
        
        self.model_manager = ModelManager(models_file)
        self.models: Dict[str, Synthesizer] = {}
        self.default_model = None
        self._initialized = True
        
        logger.info("TTS model manager initialized", config=config.model_dump())
    
    def get_model(self, model_name: Optional[str] = None) -> Synthesizer:
        """Get a TTS model by name."""
        # Use default model if not specified
        if model_name is None:
            model_name = self.config.model_name
        
        # Return cached model if available
        if model_name in self.models:
            return self.models[model_name]
        
        # Load model
        logger.info("Loading TTS model", model_name=model_name)
        start_time = time.time()
        
        try:
            # Get model info
            model_path, config_path, model_item = self.model_manager.download_model(model_name)
            vocoder_name = model_item.get("default_vocoder", None)
            
            # Load vocoder if needed
            if vocoder_name is not None:
                vocoder_path, vocoder_config_path, _ = self.model_manager.download_model(vocoder_name)
            else:
                vocoder_path, vocoder_config_path = None, None
            
            # Create synthesizer
            synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config_path,
                encoder_checkpoint="",
                encoder_config="",
                use_cuda=self.config.device == "cuda",
            )
            
            # Cache model
            self.models[model_name] = synthesizer
            
            # Set default model if not set
            if self.default_model is None:
                self.default_model = synthesizer
            
            load_time = time.time() - start_time
            logger.info("TTS model loaded", model_name=model_name, load_time=load_time)
            
            return synthesizer
        
        except Exception as e:
            logger.error("Failed to load TTS model", model_name=model_name, error=str(e), exc_info=True)
            raise
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available TTS models."""
        return self.model_manager.list_tts_models()
    
    def list_available_voices(self, model_name: Optional[str] = None) -> List[str]:
        """List all available voices for a model."""
        model = self.get_model(model_name)
        
        # Check if model has speakers
        if hasattr(model.tts_model, "speaker_manager") and model.tts_model.speaker_manager is not None:
            return model.tts_model.speaker_manager.speaker_names
        
        return ["default"]
    
    def list_available_languages(self, model_name: Optional[str] = None) -> List[str]:
        """List all available languages for a model."""
        model = self.get_model(model_name)
        
        # Check if model has languages
        if hasattr(model.tts_model, "language_manager") and model.tts_model.language_manager is not None:
            return model.tts_model.language_manager.language_names
        
        # Try to infer from model name
        if model_name and "/" in model_name:
            parts = model_name.split("/")
            if len(parts) > 1:
                return [parts[1]]
        
        return ["en"]


class TTSSynthesizer:
    """TTS synthesizer for converting text to speech."""
    
    def __init__(self, model_config: TTSModelConfig):
        """Initialize the TTS synthesizer."""
        self.model_config = model_config
        self.model_manager = TTSModelManager(model_config)
        logger.info("TTS synthesizer initialized", config=model_config.model_dump())
    
    def synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> bytes:
        """Synthesize speech from text."""
        start_time = time.time()
        
        try:
            # Get model
            model = self.model_manager.get_model(self.model_config.model_name)
            
            # Prepare synthesis parameters
            speaker = options.voice
            if speaker == "default":
                speaker = None
            
            language = options.language
            
            # Synthesize speech
            logger.info(
                "Synthesizing speech",
                text_length=len(text),
                language=language,
                voice=speaker,
                speed=options.speed,
            )
            
            wav = model.tts(
                text=text,
                speaker_name=speaker,
                language_name=language,
                speed=options.speed,
            )
            
            # Convert to bytes
            audio_bytes = self._convert_audio(wav, options.format, options.sample_rate)
            
            synthesis_time = time.time() - start_time
            logger.info(
                "Speech synthesized",
                text_length=len(text),
                audio_size=len(audio_bytes),
                synthesis_time=synthesis_time,
            )
            
            return audio_bytes
        
        except Exception as e:
            logger.error(
                "Failed to synthesize speech",
                text_length=len(text),
                error=str(e),
                exc_info=True,
            )
            raise
    
    def _convert_audio(self, wav: np.ndarray, format: str, sample_rate: int) -> bytes:
        """Convert audio to the specified format."""
        import soundfile as sf
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        
        # Write audio to buffer
        sf.write(buffer, wav, sample_rate, format=format)
        
        # Get bytes from buffer
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        return audio_bytes 