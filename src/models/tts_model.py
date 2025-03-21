"""
TTS model module for handling text-to-speech synthesis.
"""
import io
import os
import time
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from TTS.utils.synthesizer import Synthesizer

from src.config import TTSModelConfig, SynthesisOptions
from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)


class TTSModelManager:
    """Manager for TTS models without depending on ModelManager."""
    
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
        self.models: Dict[str, Synthesizer] = {}
        self.default_model = None
        self._initialized = True
        
        # Check model directory at initialization
        model_parts = self.config.model_name.split("/")
        if len(model_parts) >= 4:
            if model_parts[1] == "multilingual" and model_parts[3] == "xtts_v2":
                # XTTS-v2 model
                model_dir = os.path.join(self.config.download_root, *model_parts)
            else:
                # Regular model structure
                model_dir = os.path.join(self.config.download_root, *model_parts)
        else:
            model_dir = os.path.join(self.config.download_root, self.config.model_name)
        
        if os.path.exists(model_dir):
            logger.info(f"Model directory exists at: {model_dir}")
            
            # List files to verify
            try:
                files = os.listdir(model_dir)
                logger.info(f"Model directory contents: {files}")
            except Exception as e:
                logger.warning(f"Failed to list model directory: {e}")
        else:
            logger.warning(f"Model directory not found at: {model_dir}")
            
            # Check parent directories
            parent_dir = os.path.dirname(model_dir)
            while parent_dir and parent_dir != "/" and parent_dir != self.config.download_root:
                if os.path.exists(parent_dir):
                    logger.info(f"Parent directory exists: {parent_dir}")
                    try:
                        files = os.listdir(parent_dir)
                        logger.info(f"Parent directory contents: {files}")
                    except Exception as e:
                        logger.warning(f"Failed to list parent directory: {e}")
                    break
                parent_dir = os.path.dirname(parent_dir)
            
            try:
                # List root directory
                if os.path.exists(self.config.download_root):
                    logger.info(f"Root directory contents: {os.listdir(self.config.download_root)}")
            except Exception as e:
                logger.warning(f"Failed to list root directory: {e}")
        
        logger.info("TTS model manager initialized", config=config.model_dump())
    
    def get_model_paths(self, model_name: str) -> List[str]:
        """Get all possible paths where a model might be located."""
        # Standard paths in our configuration
        model_dir = os.path.join(self.config.download_root, *model_name.split("/"))
        paths = [model_dir]
        
        # Add Coqui's model path format (transforming tts_models/en/vctk/vits to tts_models--en--vctk--vits)
        coqui_model_id = "--".join(model_name.split("/"))
        
        # Common locations where Coqui might store models
        coqui_paths = [
            f"/usr/local/lib/python3.8/site-packages/TTS/.models/{coqui_model_id}",
            f"/usr/local/lib/python3.9/site-packages/TTS/.models/{coqui_model_id}",
            f"/usr/local/lib/python3.10/site-packages/TTS/.models/{coqui_model_id}",
            f"/usr/local/lib/python3.11/site-packages/TTS/.models/{coqui_model_id}",
        ]
        
        paths.extend(coqui_paths)
        logger.info(f"Checking model paths: {paths}")
        return paths
    
    def get_model(self, model_name: Optional[str] = None) -> Synthesizer:
        """Get a TTS model by name, loading directly from file system."""
        # Use default model if not specified
        if model_name is None:
            model_name = self.config.model_name
        
        # Return cached model if available
        if model_name in self.models:
            return self.models[model_name]
        
        # Load model
        logger.info("Loading TTS model", model_name=model_name)
        
        try:
            synthesizer = self.load_model(model_name)
            
            # Cache model
            self.models[model_name] = synthesizer
            
            # Set default model if not set
            if self.default_model is None:
                self.default_model = synthesizer
            
            logger.info("TTS model loaded successfully", model_name=model_name)
            
            return synthesizer
        
        except Exception as e:
            logger.error("Failed to load TTS model", model_name=model_name, error=str(e), exc_info=True)
            raise
    
    def load_model(self, model_name: str) -> Synthesizer:
        """Load model from any available location."""
        start_time = time.time()
        model_paths = self.get_model_paths(model_name)
        
        # Special handling for XTTS-v2 model
        is_xtts_v2 = model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
        
        for base_path in model_paths:
            if not os.path.isdir(base_path):
                logger.warning(f"Path is not a directory: {base_path}")
                continue
                
            logger.info(f"Found model directory at: {base_path}")
            
            # For XTTS-v2, we use specific file names based on what we know from the model repo
            if is_xtts_v2:
                model_file = os.path.join(base_path, "model.pth")
                config_file = os.path.join(base_path, "config.json")
                speakers_file = os.path.join(base_path, "speakers_xtts.pth")
                dvae_file = os.path.join(base_path, "dvae.pth")
                vocab_file = os.path.join(base_path, "vocab.json")
                
                # Check if all required files exist
                if not all(os.path.isfile(f) for f in [model_file, config_file, speakers_file, dvae_file, vocab_file]):
                    logger.warning(f"Missing required files in {base_path}")
                    continue
                
                logger.info(f"Found all required files for XTTS-v2 in {base_path}")
                try:
                    # TTS v0.22.0 compatible initialization with additional parameters for XTTS-v2
                    synthesizer = Synthesizer(
                        tts_checkpoint=base_path,
                        tts_config_path=config_file,
                        tts_speakers_file=speakers_file,
                        tts_languages_file=None,
                        vocoder_checkpoint=None,
                        vocoder_config=None,
                        encoder_checkpoint="",
                        encoder_config="",
                        use_cuda=self.config.device == "cuda",
                    )
                    
                    load_time = time.time() - start_time
                    logger.info("TTS model loaded successfully", load_time=load_time)
                    return synthesizer
                except Exception as e:
                    logger.error(f"Failed to load XTTS-v2 model: {e}")
                    continue
            else:
                # For other models, try different file name combinations
                model_candidates = ["model_file.pth", "model.pth"]
                config_candidates = ["config.json", "config_file.json"]
                speakers_candidates = ["speakers_map.json", "speakers.json", "speakers.pth"]
                
                model_file = None
                config_file = None
                speakers_file = None
                
                # Find model file
                for candidate in model_candidates:
                    file_path = os.path.join(base_path, candidate)
                    if os.path.isfile(file_path):
                        model_file = file_path
                        break
                
                # Find config file
                for candidate in config_candidates:
                    file_path = os.path.join(base_path, candidate)
                    if os.path.isfile(file_path):
                        config_file = file_path
                        break
                
                # Find speakers file
                for candidate in speakers_candidates:
                    file_path = os.path.join(base_path, candidate)
                    if os.path.isfile(file_path):
                        speakers_file = file_path
                        break
                
                if model_file and config_file:
                    try:
                        synthesizer = Synthesizer(
                            tts_checkpoint=model_file,
                            tts_config_path=config_file,
                            tts_speakers_file=speakers_file,
                            tts_languages_file=None,
                            vocoder_checkpoint=None,
                            vocoder_config=None,
                            encoder_checkpoint="",
                            encoder_config="",
                            use_cuda=self.config.device == "cuda",
                        )
                        
                        load_time = time.time() - start_time
                        logger.info("TTS model loaded successfully", load_time=load_time)
                        return synthesizer
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        continue
        
        # If we get here, we couldn't load the model from any path
        model_type = "XTTS-v2" if is_xtts_v2 else model_name
        raise ValueError(f"Could not find or load model: {model_type}. Please ensure model files are correctly placed in the mounted volume.")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available TTS models by scanning the filesystem."""
        models = []
        tts_models_dir = os.path.join(self.config.download_root, "tts_models")
        
        if not os.path.exists(tts_models_dir):
            logger.warning(f"TTS models directory not found: {tts_models_dir}")
            return models
        
        try:
            # Walk through the directory structure to find models
            for lang_dir in os.listdir(tts_models_dir):
                lang_path = os.path.join(tts_models_dir, lang_dir)
                if not os.path.isdir(lang_path):
                    continue
                
                for dataset_dir in os.listdir(lang_path):
                    dataset_path = os.path.join(lang_path, dataset_dir)
                    if not os.path.isdir(dataset_path):
                        continue
                    
                    for model_dir in os.listdir(dataset_path):
                        model_path = os.path.join(dataset_path, model_dir)
                        if not os.path.isdir(model_path):
                            continue
                        
                        # Check if this directory contains a valid model
                        has_model = any(
                            os.path.exists(os.path.join(model_path, f)) 
                            for f in ["model.pth", "model_file.pth"]
                        )
                        has_config = any(
                            os.path.exists(os.path.join(model_path, f)) 
                            for f in ["config.json", "config_file.json"]
                        )
                        
                        if has_model and has_config:
                            model_name = f"tts_models/{lang_dir}/{dataset_dir}/{model_dir}"
                            models.append({
                                "model_name": model_name,
                                "language": lang_dir,
                                "dataset": dataset_dir,
                                "model_type": model_dir,
                                "description": f"TTS model for {lang_dir} language from {dataset_dir} dataset using {model_dir}"
                            })
            
            return models
        except Exception as e:
            logger.warning(f"Error listing TTS models: {str(e)}")
            return models
    
    def list_available_voices(self, model_name: Optional[str] = None) -> List[str]:
        """List all available voices for a model."""
        try:
            model = self.get_model(model_name)
            
            # Check if model has speakers
            if hasattr(model.tts_model, "speaker_manager") and model.tts_model.speaker_manager is not None:
                return model.tts_model.speaker_manager.speaker_names
            
            # If we can't get speakers from the model but we know it's the XTTS v2 model,
            # return a list of known XTTS speakers
            if model_name == "tts_models/multilingual/multi-dataset/xtts_v2" or (not model_name and self.config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"):
                # Return list of common XTTS speakers
                return ["random", "female-en-1", "male-en-1", "female-es-1", "male-es-1", "female-fr-1", "male-fr-1"]
            
            # Return a single placeholder for single-speaker models
            return ["speaker_00"]
        except Exception as e:
            logger.warning(f"Error listing voices: {str(e)}")
            
            # If we can't get the model but know it's XTTS v2, return known speakers
            if model_name == "tts_models/multilingual/multi-dataset/xtts_v2" or (not model_name and self.config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"):
                return ["random", "female-en-1", "male-en-1", "female-es-1", "male-es-1", "female-fr-1", "male-fr-1"]
                
            # Return a single placeholder
            return ["speaker_00"]
    
    def list_available_languages(self, model_name: Optional[str] = None) -> List[str]:
        """List all available languages for a model."""
        try:
            model = self.get_model(model_name)
            
            # Check if model has languages
            if hasattr(model.tts_model, "language_manager") and model.tts_model.language_manager is not None:
                return model.tts_model.language_manager.language_names
            
            # XTTS-v2 is multilingual but doesn't always expose language names properly
            if model_name and "multilingual" in model_name:
                return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hu"]
            
            # Try to infer from model name
            if model_name and "/" in model_name:
                parts = model_name.split("/")
                if len(parts) > 1:
                    return [parts[1]]
            
            return ["en"]
        except Exception as e:
            logger.warning(f"Error listing languages: {str(e)}")
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
                
                # Use the first available speaker if any
                if available_speakers and len(available_speakers) > 0:
                    speaker = available_speakers[0]
                    logger.info(
                        "Using first available speaker instead of default",
                        speaker=speaker
                    )
                else:
                    # Use None for single-speaker models
                    logger.info("No speakers available, using None for voice parameter")
                    speaker = None
                
            # For single-speaker models, set to None if a speaker is specified but not needed
            if not hasattr(model.tts_model, "speaker_manager") or model.tts_model.speaker_manager is None:
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
            
            # Special handling for XTTS-v2
            is_xtts_v2 = self.model_config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"
            
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
                model=self.model_config.model_name,
                is_xtts=is_xtts_v2
            )
            
            # Check if the model supports additional parameters
            tts_kwargs = {
                "text": modified_text,
            }
            
            # Only add speaker_name if it's not None (XTTS-v2 needs this)
            if speaker is not None:
                tts_kwargs["speaker_name"] = speaker
            
            # Only add language_name if the model supports it
            if language is not None and hasattr(model.tts_model, "language_manager"):
                tts_kwargs["language_name"] = language
            
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
            
            # For XTTS-v2, which can generate reference speech for voice cloning, we don't need to add it for standard usage
            
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