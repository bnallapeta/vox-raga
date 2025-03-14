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
            
            # Handle default speaker for multi-speaker models
            if speaker == "default":
                # Get available speakers
                available_speakers = self.model_manager.list_available_voices()
                
                # Check if this is a multi-speaker model
                if hasattr(model.tts_model, "speaker_manager") and model.tts_model.speaker_manager is not None:
                    # Use the first available speaker if any
                    if available_speakers and len(available_speakers) > 0:
                        speaker = available_speakers[0]
                        logger.info(
                            "Using first available speaker for default",
                            speaker=speaker
                        )
                    else:
                        # If no speakers available, this will likely fail but we'll let the model handle it
                        logger.warning(
                            "No speakers available for multi-speaker model",
                            model=self.model_config.model_name
                        )
                else:
                    # For single-speaker models, set to None
                    speaker = None
            
            language = options.language
            modified_text = text
            
            # Default speech parameters
            speech_speed = options.speed
            speech_pitch = getattr(options, 'pitch', 1.0)
            speech_energy = getattr(options, 'energy', 1.0)
            
            # Apply emotion and style if supported
            if options.emotion or options.style:
                logger.info(
                    "Emotion/style requested",
                    emotion=options.emotion,
                    style=options.style,
                )
                
                # Adjust speech parameters based on emotion
                if options.emotion:
                    # Emotion-specific parameter adjustments
                    emotion_params = {
                        "happy": {"pitch": 1.15, "speed": speech_speed * 1.1, "energy": 1.2},
                        "sad": {"pitch": 0.85, "speed": speech_speed * 0.9, "energy": 0.8},
                        "angry": {"pitch": 1.1, "speed": speech_speed * 1.15, "energy": 1.4},
                        "neutral": {"pitch": 1.0, "speed": speech_speed, "energy": 1.0},
                        "excited": {"pitch": 1.2, "speed": speech_speed * 1.2, "energy": 1.3},
                        "calm": {"pitch": 0.95, "speed": speech_speed * 0.9, "energy": 0.9},
                        "fearful": {"pitch": 1.1, "speed": speech_speed * 1.1, "energy": 0.8},
                        "surprised": {"pitch": 1.3, "speed": speech_speed * 1.1, "energy": 1.2}
                    }
                    
                    # Apply emotion-specific parameters if available
                    if options.emotion in emotion_params:
                        params = emotion_params[options.emotion]
                        speech_pitch = params["pitch"]
                        speech_speed = params["speed"]
                        speech_energy = params["energy"]
                        
                        logger.info(
                            "Applied emotion parameters",
                            emotion=options.emotion,
                            pitch=speech_pitch,
                            speed=speech_speed,
                            energy=speech_energy
                        )
                
                # Create a dictionary of supported emotions with their corresponding SSML-like tags
                emotion_tags = {
                    "happy": "<prosody rate='fast' pitch='high' volume='loud'>",
                    "sad": "<prosody rate='slow' pitch='low' volume='soft'>",
                    "angry": "<prosody rate='fast' pitch='high' volume='loud' contour='(0%,+30Hz)(100%,+15Hz)'>",
                    "neutral": "<prosody rate='medium' pitch='medium' volume='medium'>",
                    "excited": "<prosody rate='fast' pitch='high' volume='loud' contour='(0%,+20Hz)(50%,+40Hz)(100%,+30Hz)'>",
                    "calm": "<prosody rate='slow' pitch='low' volume='soft' contour='(0%,-10Hz)(100%,-20Hz)'>",
                    "fearful": "<prosody rate='fast' pitch='high' volume='soft' contour='(0%,+15Hz)(50%,+30Hz)(100%,+10Hz)'>",
                    "surprised": "<prosody rate='fast' pitch='high' volume='loud' contour='(0%,+30Hz)(50%,+50Hz)(100%,+15Hz)'>"
                }
                
                # Create a dictionary of supported styles with their corresponding SSML-like tags
                style_tags = {
                    "formal": "<voice name='formal'>",
                    "casual": "<voice name='casual'>",
                    "news": "<voice name='news'>",
                    "storytelling": "<voice name='storytelling'>",
                    "conversational": "<voice name='conversational'>",
                    "instructional": "<voice name='instructional'>"
                }
                
                # Create a dictionary of supported emotions with their corresponding prompts
                emotion_prompts = {
                    "happy": "Say this in a happy and cheerful tone: ",
                    "sad": "Express this with sadness: ",
                    "angry": "Say this with anger and intensity: ",
                    "neutral": "Say this in a neutral tone: ",
                    "excited": "Say this with excitement and enthusiasm: ",
                    "calm": "Say this in a calm and soothing voice: ",
                    "fearful": "Express this with fear and worry: ",
                    "surprised": "Say this with surprise and astonishment: "
                }
                
                # Create a dictionary of supported styles with their corresponding prompts
                style_prompts = {
                    "formal": "In a formal and professional manner: ",
                    "casual": "In a casual, conversational tone: ",
                    "news": "In the style of a news broadcaster: ",
                    "storytelling": "Like a storyteller narrating a tale: ",
                    "conversational": "In a friendly conversational style: ",
                    "instructional": "As if giving clear instructions: "
                }
                
                # Apply emotion prompt and tags if specified
                if options.emotion and options.emotion in emotion_prompts:
                    # Apply both prompt and SSML-like tags
                    modified_text = f"{emotion_prompts[options.emotion]}{emotion_tags.get(options.emotion, '')}{modified_text}</prosody>"
                    logger.info(
                        "Applied emotion prompt and tags",
                        emotion=options.emotion,
                        prompt=emotion_prompts[options.emotion]
                    )
                
                # Apply style prompt and tags if specified
                if options.style and options.style in style_prompts:
                    # Apply both prompt and SSML-like tags
                    modified_text = f"{style_prompts[options.style]}{style_tags.get(options.style, '')}{modified_text}</voice>"
                    logger.info(
                        "Applied style prompt and tags",
                        style=options.style,
                        prompt=style_prompts[options.style]
                    )
            
            # Synthesize speech
            logger.info(
                "Synthesizing speech",
                text_length=len(modified_text),
                original_text_length=len(text),
                language=language,
                voice=speaker,
                speed=speech_speed,
                pitch=speech_pitch,
                energy=speech_energy,
                emotion=options.emotion,
                style=options.style,
            )
            
            # Check if the model supports additional parameters
            tts_kwargs = {
                "text": modified_text,
                "speaker_name": speaker,
                "language_name": language,
            }
            
            # VITS model uses length_scale for speed and pitch_scale for pitch
            if hasattr(model.tts_model, "length_scale"):
                tts_kwargs["length_scale"] = 1.0 / speech_speed  # Inverse relationship
                logger.info("Model supports speed control via length_scale", speed=speech_speed)
            
            if hasattr(model.tts_model, "pitch_scale"):
                tts_kwargs["pitch_scale"] = speech_pitch
                logger.info("Model supports pitch control via pitch_scale", pitch=speech_pitch)
            elif hasattr(model.tts_model, "pitch_control"):
                tts_kwargs["pitch"] = speech_pitch
                logger.info("Model supports pitch control directly", pitch=speech_pitch)
            
            # Check for energy support
            if hasattr(model.tts_model, "energy_scale"):
                tts_kwargs["energy_scale"] = speech_energy
                logger.info("Model supports energy control via energy_scale", energy=speech_energy)
            elif hasattr(model.tts_model, "energy_control"):
                tts_kwargs["energy"] = speech_energy
                logger.info("Model supports energy control directly", energy=speech_energy)
            
            # Pass the modified text and all necessary parameters to the TTS model
            wav = model.tts(**tts_kwargs)
            
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