"""
TTS model module for handling text-to-speech synthesis.
"""
import io
import os
import time
import json
import inspect
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.io import load_fsspec

from src.config import TTSModelConfig, SynthesisOptions
from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)

# Custom implementation of load_fsspec with weights_only=False
def custom_load_fsspec(path, map_location=None, cache=True, **kwargs):
    """Custom implementation of load_fsspec that sets weights_only=False."""
    # Make sure weights_only is set to False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    
    # Call the original load_fsspec
    return load_fsspec(path, map_location=map_location, cache=cache, **kwargs)

# SimpleSpeakerManager for XTTS models
class SimpleSpeakerManager:
    def __init__(self, speakers_file=None):
        # Default speakers as fallback
        self._name_to_id = {
            "Claribel Dervla": 0,
            "Daisy Studious": 1,
            "Gracie Wise": 2,
            "Tammie Ema": 3,
            "Alison Dietlinde": 4,
            "Andrew Chipper": 5,
        }
        self.embedding = {}
        
        # Try to load actual embeddings if available
        if speakers_file and os.path.isfile(speakers_file):
            try:
                logger.info(f"Loading speaker embeddings from {speakers_file}")
                
                # Use torch.load with weights_only=False
                try:
                    speakers_data = torch.load(speakers_file, weights_only=False)
                except:
                    # Fallback to default load
                    logger.warning("torch.load with weights_only=False failed, trying without parameter")
                    speakers_data = torch.load(speakers_file)
                
                # Check if speakers_data is a dictionary
                if isinstance(speakers_data, dict):
                    logger.info(f"Loaded speaker embeddings dictionary with {len(speakers_data)} entries")
                    
                    # Store the embeddings
                    self.embedding = speakers_data
                    
                    # Update name_to_id mapping
                    speaker_names = sorted(list(speakers_data.keys()))
                    self._name_to_id = {name: i for i, name in enumerate(speaker_names)}
                    
                    # Log the first few speakers for debugging
                    speaker_names = list(self._name_to_id.keys())
                    if len(speaker_names) > 0:
                        logger.info(f"First {min(5, len(speaker_names))} speakers: {speaker_names[:5]}")
                        
                        # Verify we can access the first speaker's embedding
                        first_speaker = speaker_names[0]
                        embed = speakers_data.get(first_speaker)
                        if embed is not None:
                            logger.info(f"Found embedding for {first_speaker}")
                            # Check the format of the embedding
                            if isinstance(embed, torch.Tensor):
                                logger.info(f"Embedding is a torch.Tensor with shape {embed.shape}")
                            elif isinstance(embed, np.ndarray):
                                logger.info(f"Embedding is a numpy array with shape {embed.shape}")
                            else:
                                logger.info(f"Embedding is of type {type(embed)}")
                        else:
                            logger.warning(f"No embedding found for speaker {first_speaker}")
                else:
                    logger.warning(f"Loaded data is not a dictionary: {type(speakers_data)}")
            except Exception as e:
                logger.error(f"Error loading speaker embeddings: {str(e)}")
    
    @property
    def name_to_id(self):
        return self._name_to_id
    
    @name_to_id.setter
    def name_to_id(self, value):
        self._name_to_id = value
    
    @property
    def speaker_names(self):
        return list(self._name_to_id.keys())
    
    def get_speakers(self):
        """Get a list of available speakers."""
        return self.speaker_names
    
    def get_d_vector_by_name(self, name):
        """Get the d-vector (speaker embedding) for a given speaker name."""
        if name not in self.embedding:
            logger.warning(f"Speaker '{name}' not found in embeddings")
            return None
        
        try:
            # Get the embedding
            embed = self.embedding[name]
            
            # Log the type of embedding we're working with
            logger.info(f"Got embedding for '{name}' of type {type(embed)}")
            
            # XTTS v2 stores embeddings as dictionaries with various possible keys
            if isinstance(embed, dict):
                # Try different known keys, in order of preference
                if 'x_vector' in embed:
                    logger.info(f"Found x_vector for '{name}' in dictionary embedding")
                    embed = embed['x_vector']
                elif 'speaker_embedding' in embed:
                    logger.info(f"Found 'speaker_embedding' key for '{name}' in dictionary embedding.")
                    embed = embed['speaker_embedding']
                elif 'gpt_cond_latent' in embed:
                    logger.info(f"Found 'gpt_cond_latent' key for '{name}' in dictionary embedding.")
                    # Use gpt_cond_latent as a fallback
                    embed = embed['gpt_cond_latent']
                else:
                    # If none of the known keys are present, use the first value in the dict
                    logger.warning(f"No known embedding key found for '{name}', using first available value")
                    first_key = next(iter(embed))
                    embed = embed[first_key]
                    logger.info(f"Using '{first_key}' with type {type(embed)}")
            
            # Convert to tensor if it's not already
            if not isinstance(embed, torch.Tensor):
                if isinstance(embed, np.ndarray):
                    embed = torch.from_numpy(embed)
                    logger.info(f"Converted numpy array to tensor with shape {embed.shape}")
                elif isinstance(embed, list):
                    embed = torch.tensor(embed)
                    logger.info(f"Converted list to tensor with shape {embed.shape}")
                elif isinstance(embed, dict):
                    # If we still have a dict here, something went wrong in the above logic
                    logger.warning(f"Embedding is still a dict after processing: {embed.keys()}")
                    # Create a dummy embedding as a last resort
                    embed = torch.zeros((1, 512))  # Common embedding size for TTS models
                    logger.info(f"Created dummy embedding with shape {embed.shape}")
                else:
                    logger.warning(f"Unknown embedding type {type(embed)}")
                    # Create a dummy embedding
                    embed = torch.zeros((1, 512))
                    logger.info(f"Created dummy embedding with shape {embed.shape}")
            
            # Ensure correct shape: [1, D] for speaker embedding
            if len(embed.shape) == 1:
                # If it's a 1D tensor, add a batch dimension
                embed = embed.unsqueeze(0)
                logger.info(f"Added batch dimension, new shape: {embed.shape}")
            
            # Final check to make sure we have a valid tensor
            if not torch.isfinite(embed).all():
                logger.warning(f"Embedding contains non-finite values, replacing with zeros")
                embed = torch.zeros_like(embed)
            
            logger.info(f"Returning embedding for '{name}' with shape {embed.shape}")
            
            return embed
        except Exception as e:
            logger.error(f"Error processing embedding for '{name}': {e}")
            # Create a dummy embedding as a last resort
            embed = torch.zeros((1, 512))
            logger.info(f"Created dummy embedding with shape {embed.shape} after error")
            return embed

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
    
    def get_model(self, model_name: str = None) -> Synthesizer:
        """Get a loaded model by name, loading it if necessary.
        
        Args:
            model_name: Optional name of the model to load. If None, loads the default model.
            
        Returns:
            Loaded synthesizer
        """
        # If model_name is None, use the default model name from config
        if model_name is None:
            model_name = self.config.model_name
            
        # Check if model is already loaded
        if model_name in self.models:
            logger.info(f"Using cached model: {model_name}")
            return self.models[model_name]
        
        # Load the model
        model = self.load_model(model_name)
        
        # Cache the model
        self.models[model_name] = model
        
        return model
    
    def _patch_xtts_synthesizer(self, synthesizer: Synthesizer):
        """Patch the XTTS synthesizer to handle specific issues."""
        logger.info("Patching XTTS synthesizer to handle specific issues")
        
        # See if we can generate reference audio embeddings for common speakers
        self._prepare_reference_embeddings(synthesizer)
        
        # Store the original tts method
        original_tts = synthesizer.tts
        
        # Create a patched version that handles the use_d_vector_file attribute issue
        def patched_tts(text, language=None, language_name=None, speaker_name=None, speaker_wav=None, ref_audio_path=None, **kwargs):
            """Patched tts method that handles specific issues with the XTTS model."""
            # Immediately add the missing attribute to the tts_config
            if not hasattr(synthesizer.tts_config, "use_d_vector_file"):
                logger.info("Adding 'use_d_vector_file' attribute to tts_config")
                # Add the attribute directly to avoid errors
                setattr(synthesizer.tts_config, "use_d_vector_file", False)
            
            try:
                # Use language_name if provided, otherwise fall back to language
                effective_language = language_name or language or getattr(synthesizer.tts_config, "language", None)
                
                # Clean up kwargs to avoid duplicate parameters
                if 'language' in kwargs:
                    del kwargs['language']
                if 'language_name' in kwargs:
                    del kwargs['language_name']
                
                # If we have a language, set it as language_name
                if effective_language:
                    kwargs['language_name'] = effective_language
                
                # If speaker_name is provided, try to use the corresponding embedding
                embedding = None
                if speaker_name and hasattr(synthesizer, "speaker_manager") and synthesizer.speaker_manager:
                    logger.info(f"Looking up embedding for speaker '{speaker_name}' from synthesizer.speaker_manager")
                    try:
                        embedding = synthesizer.speaker_manager.get_d_vector_by_name(speaker_name)
                        if embedding is not None:
                            logger.info(f"Found embedding for '{speaker_name}' in synthesizer.speaker_manager")
                    except Exception as e:
                        logger.error(f"Error getting embedding from synthesizer.speaker_manager: {e}")
                
                # Also check the tts_model.speaker_manager if it exists
                if embedding is None and speaker_name and hasattr(synthesizer.tts_model, "speaker_manager") and synthesizer.tts_model.speaker_manager:
                    logger.info(f"Looking up embedding for speaker '{speaker_name}' from tts_model.speaker_manager")
                    try:
                        embedding = synthesizer.tts_model.speaker_manager.get_d_vector_by_name(speaker_name)
                        if embedding is not None:
                            logger.info(f"Found embedding for '{speaker_name}' in tts_model.speaker_manager")
                    except Exception as e:
                        logger.error(f"Error getting embedding from tts_model.speaker_manager: {e}")
                
                # Try direct inference with embedding if available
                if embedding is not None:
                    logger.info("Using speaker embedding for direct inference")
                    try:
                        if hasattr(synthesizer.tts_model, "inference") and hasattr(synthesizer.tts_model, "get_conditioning_latents"):
                            # For XTTS models, we need both speaker_embedding and gpt_cond_latent
                            # First, try to get a reference audio path if none is provided
                            if ref_audio_path is None and speaker_wav is None:
                                # Try to find a reference audio file
                                if hasattr(synthesizer, "output_path"):
                                    model_dir = os.path.dirname(os.path.dirname(synthesizer.output_path))
                                    if os.path.exists(model_dir):
                                        reference_dir = os.path.join(model_dir, "reference_audio")
                                        if os.path.exists(reference_dir):
                                            # Find any WAV or MP3 file in the reference directory
                                            for file in os.listdir(reference_dir):
                                                if file.endswith(".wav") or file.endswith(".mp3"):
                                                    ref_audio_path = os.path.join(reference_dir, file)
                                                    logger.info(f"Using reference audio from {ref_audio_path}")
                                                    break
                            
                            # Use either ref_audio_path or speaker_wav to get conditioning latents
                            if ref_audio_path is not None or speaker_wav is not None:
                                try:
                                    # Use get_conditioning_latents to get gpt_cond_latent
                                    logger.info("Getting conditioning latents")
                                    if ref_audio_path is not None:
                                        # Try different parameter names since ref_audio_path is not accepted
                                        try:
                                            # Try with reference_wav first
                                            gpt_cond_latent, speaker_embedding = synthesizer.tts_model.get_conditioning_latents(
                                                reference_wav=ref_audio_path
                                            )
                                            logger.info("Used reference_wav parameter successfully")
                                        except (TypeError, ValueError) as e:
                                            logger.info(f"reference_wav failed: {e}, trying reference_audio")
                                            try:
                                                # Try with reference_audio next
                                                gpt_cond_latent, speaker_embedding = synthesizer.tts_model.get_conditioning_latents(
                                                    reference_audio=ref_audio_path
                                                )
                                                logger.info("Used reference_audio parameter successfully")
                                            except (TypeError, ValueError) as e:
                                                logger.info(f"reference_audio failed: {e}, trying without parameter name")
                                                # As a last resort, try just passing the path as first argument
                                                gpt_cond_latent, speaker_embedding = synthesizer.tts_model.get_conditioning_latents(
                                                    ref_audio_path
                                                )
                                                logger.info("Used positional argument successfully")
                                        
                                        logger.info(f"Successfully obtained conditioning latents from reference audio")
                                    else:
                                        # Try different parameter names for raw_audio as well
                                        try:
                                            # First try with audio
                                            gpt_cond_latent, _ = synthesizer.tts_model.get_conditioning_latents(
                                                audio=speaker_wav
                                            )
                                            logger.info("Used audio parameter successfully")
                                        except (TypeError, ValueError) as e:
                                            logger.info(f"audio failed: {e}, trying raw_audio")
                                            try:
                                                # Then try with raw_audio
                                                gpt_cond_latent, _ = synthesizer.tts_model.get_conditioning_latents(
                                                    raw_audio=speaker_wav
                                                )
                                                logger.info("Used raw_audio parameter successfully")
                                            except (TypeError, ValueError) as e:
                                                logger.info(f"raw_audio failed: {e}, trying wav_audio")
                                                # Try another possible parameter name
                                                gpt_cond_latent, _ = synthesizer.tts_model.get_conditioning_latents(
                                                    wav_audio=speaker_wav
                                                )
                                                logger.info("Used wav_audio parameter successfully")
                                    logger.info(f"Successfully obtained conditioning latents from reference audio")
                                    
                                    # Now call inference with the conditioning latents
                                    logger.info("Calling tts_model.inference with embedding and gpt_cond_latent")
                                    output = synthesizer.tts_model.inference(
                                        text=text,
                                        speaker_embedding=embedding,
                                        gpt_cond_latent=gpt_cond_latent,
                                        **kwargs
                                    )
                                    logger.info(f"Direct inference output type: {type(output)}")
                                    
                                    # If the output is a dictionary, extract the audio data
                                    if isinstance(output, dict):
                                        logger.info(f"Output is a dictionary with keys: {list(output.keys())}")
                                        # Look for common keys that might contain the audio data
                                        if 'wav' in output:
                                            output = output['wav']
                                            logger.info(f"Extracted 'wav' from output dictionary, new type: {type(output)}")
                                        elif 'audio' in output:
                                            output = output['audio']
                                            logger.info(f"Extracted 'audio' from output dictionary, new type: {type(output)}")
                                        elif 'waveform' in output:
                                            output = output['waveform']
                                            logger.info(f"Extracted 'waveform' from output dictionary, new type: {type(output)}")
                                        elif len(output) > 0:
                                            # If we can't find a known key, try the first value
                                            first_key = next(iter(output))
                                            first_value = output[first_key]
                                            if isinstance(first_value, (np.ndarray, torch.Tensor)):
                                                output = first_value
                                                logger.info(f"Using first value with key '{first_key}', new type: {type(output)}")
                                            else:
                                                logger.warning(f"First value is not an array, type: {type(first_value)}")
                                                # Return empty audio as fallback
                                                output = np.zeros(22050)
                                        else:
                                            logger.warning("Output dictionary is empty, using silence")
                                            output = np.zeros(22050)
                                    
                                    # Convert torch tensor to numpy if needed
                                    if isinstance(output, torch.Tensor):
                                        output = output.cpu().numpy()
                                        logger.info(f"Converted torch tensor to numpy array, shape: {output.shape}")
                                    
                                    # Check if we have valid audio data
                                    if isinstance(output, np.ndarray):
                                        logger.info(f"Direct inference successful, audio shape: {output.shape}")
                                        return output
                                    else:
                                        logger.warning(f"Output is not a numpy array, type: {type(output)}, using silence")
                                        return np.zeros(22050)  # 1 second of silence
                                except Exception as e:
                                    logger.error(f"Error getting conditioning latents: {e}")
                            else:
                                logger.warning("No reference audio available, creating a dummy gpt_cond_latent")
                                # Create a dummy gpt_cond_latent based on typical dimensions
                                # XTTS typically uses a 1024-dimensional latent
                                gpt_cond_latent = torch.zeros((1, 1024))
                                
                            # Now call inference with both speaker_embedding and gpt_cond_latent
                            if gpt_cond_latent is not None:
                                logger.info(f"Calling inference with speaker_embedding and gpt_cond_latent")
                                output = synthesizer.tts_model.inference(
                                    text=text,
                                    language=effective_language,
                                    speaker_embedding=embedding,
                                    gpt_cond_latent=gpt_cond_latent,
                                    **kwargs
                                )
                                logger.info(f"Direct inference output type: {type(output)}")
                                
                                # If the output is a dictionary, extract the audio data
                                if isinstance(output, dict):
                                    logger.info(f"Output is a dictionary with keys: {list(output.keys())}")
                                    # Look for common keys that might contain the audio data
                                    if 'wav' in output:
                                        output = output['wav']
                                        logger.info(f"Extracted 'wav' from output dictionary, new type: {type(output)}")
                                    elif 'audio' in output:
                                        output = output['audio']
                                        logger.info(f"Extracted 'audio' from output dictionary, new type: {type(output)}")
                                    elif 'waveform' in output:
                                        output = output['waveform']
                                        logger.info(f"Extracted 'waveform' from output dictionary, new type: {type(output)}")
                                    elif len(output) > 0:
                                        # If we can't find a known key, try the first value
                                        first_key = next(iter(output))
                                        first_value = output[first_key]
                                        if isinstance(first_value, (np.ndarray, torch.Tensor)):
                                            output = first_value
                                            logger.info(f"Using first value with key '{first_key}', new type: {type(output)}")
                                        else:
                                            logger.warning(f"First value is not an array, type: {type(first_value)}")
                                            # Return empty audio as fallback
                                            output = np.zeros(22050)
                                    else:
                                        logger.warning("Output dictionary is empty, using silence")
                                        output = np.zeros(22050)
                                
                                # Convert torch tensor to numpy if needed
                                if isinstance(output, torch.Tensor):
                                    output = output.cpu().numpy()
                                    logger.info(f"Converted torch tensor to numpy array, shape: {output.shape}")
                                
                                # Check if we have valid audio data
                                if isinstance(output, np.ndarray):
                                    logger.info(f"Direct inference successful, audio shape: {output.shape}")
                                    return output
                                else:
                                    logger.warning(f"Output is not a numpy array, type: {type(output)}, using silence")
                                    return np.zeros(22050)  # 1 second of silence
                    except Exception as e:
                        logger.error(f"Error during direct inference with embedding: {e}")
                
                # If we have a reference audio, but not in the correct format, convert it
                if speaker_wav is not None and not isinstance(speaker_wav, np.ndarray):
                    logger.info(f"Converting speaker_wav from {type(speaker_wav)} to numpy array")
                    try:
                        import io
                        import soundfile as sf
                        wav_io = io.BytesIO(speaker_wav)
                        wav_data, sr = sf.read(wav_io)
                        speaker_wav = wav_data
                        logger.info(f"Converted speaker_wav to numpy array, shape: {speaker_wav.shape}")
                    except Exception as e:
                        logger.error(f"Error converting speaker_wav: {e}")
                
                # For XTTS, ensure we have a valid reference_audio_path or speaker_wav
                if hasattr(synthesizer.tts_model, "get_conditioning_latents") and speaker_wav is None and ref_audio_path is None:
                    # If we're using XTTS, we need a reference audio or speaker embedding
                    # Check if there are any reference audio files in a known location
                    model_dir = os.path.dirname(os.path.dirname(synthesizer.output_path)) if hasattr(synthesizer, "output_path") else None
                    if model_dir and os.path.exists(model_dir):
                        reference_dir = os.path.join(model_dir, "reference_audio")
                        if os.path.exists(reference_dir):
                            # Find any WAV or MP3 file in the reference directory
                            for file in os.listdir(reference_dir):
                                if file.endswith(".wav") or file.endswith(".mp3"):
                                    ref_audio_path = os.path.join(reference_dir, file)
                                    logger.info(f"Using reference audio from {ref_audio_path}")
                                    break
                
                # Try the original method if we have reference audio
                if speaker_wav is not None or ref_audio_path is not None:
                    logger.info(f"Trying original tts method with reference audio")
                    try:
                        return original_tts(text, **kwargs)
                    except Exception as e:
                        logger.error(f"Error with original tts method using reference audio: {e}")
                
                # Last resort: try with the first available speaker embedding
                logger.info("Trying with first available speaker as last resort")
                available_speakers = []
                
                # Try to get speakers from synthesizer.speaker_manager
                if hasattr(synthesizer, "speaker_manager") and synthesizer.speaker_manager:
                    available_speakers = list(synthesizer.speaker_manager.name_to_id)
                # Or try tts_model.speaker_manager
                elif hasattr(synthesizer.tts_model, "speaker_manager") and synthesizer.tts_model.speaker_manager:
                    available_speakers = list(synthesizer.tts_model.speaker_manager.name_to_id)
                
                if available_speakers:
                    fallback_speaker = available_speakers[0]
                    logger.info(f"Using fallback speaker '{fallback_speaker}'")
                    
                    # Try synthesizer.speaker_manager first
                    if hasattr(synthesizer, "speaker_manager") and synthesizer.speaker_manager:
                        try:
                            fallback_embedding = synthesizer.speaker_manager.get_d_vector_by_name(fallback_speaker)
                            if fallback_embedding is not None and hasattr(synthesizer.tts_model, "inference"):
                                logger.info(f"Using fallback embedding from synthesizer.speaker_manager")
                                output = synthesizer.tts_model.inference(
                                    text=text,
                                    **kwargs
                                )
                                return output
                        except Exception as e:
                            logger.error(f"Error with fallback speaker from synthesizer.speaker_manager: {e}")
                    
                    # Then try tts_model.speaker_manager
                    if hasattr(synthesizer.tts_model, "speaker_manager") and synthesizer.tts_model.speaker_manager:
                        try:
                            fallback_embedding = synthesizer.tts_model.speaker_manager.get_d_vector_by_name(fallback_speaker)
                            if fallback_embedding is not None and hasattr(synthesizer.tts_model, "inference"):
                                logger.info(f"Using fallback embedding from tts_model.speaker_manager")
                                output = synthesizer.tts_model.inference(
                                    text=text,
                                    **kwargs
                                )
                                return output
                        except Exception as e:
                            logger.error(f"Error with fallback speaker from tts_model.speaker_manager: {e}")
                
                # If all else fails, try the original method one more time without any speaker info
                logger.info("Trying original tts method without any speaker info as absolute last resort")
                try:
                    return original_tts(text, **kwargs)
                except Exception as e:
                    logger.error(f"Final fallback failed: {e}")
                    raise RuntimeError(f"Failed to synthesize speech: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error in patched_tts: {e}", exc_info=True)
                raise e
        
        # Replace the original tts method with our patched version
        synthesizer.tts = patched_tts
        logger.info("XTTS synthesizer patched successfully")
    
    def _prepare_reference_embeddings(self, synthesizer: Synthesizer):
        """Generate and store reference embeddings from audio files if available."""
        # Check if we have some reference audio files
        model_dir = os.path.join(self.config.download_root, "tts_models", "multilingual", "multi-dataset", "xtts_v2")
        reference_dir = os.path.join(model_dir, "reference_audio")
        
        # Create the directory if it doesn't exist
        os.makedirs(reference_dir, exist_ok=True)
        
        # Check if we have reference audio files
        reference_files = []
        if os.path.exists(reference_dir):
            for file in os.listdir(reference_dir):
                if file.endswith(".wav") or file.endswith(".mp3"):
                    reference_files.append(os.path.join(reference_dir, file))
        
        # If no reference files exist, create a default one for testing
        if not reference_files:
            logger.info("No reference audio files found. Creating a default one for testing.")
            default_ref_path = os.path.join(reference_dir, "default_reference.wav")
            
            # Create a simple sine wave as the reference audio
            try:
                import numpy as np
                import soundfile as sf
                
                # Create a 3-second sine wave at 440 Hz
                sr = 24000  # Sample rate
                duration = 3  # seconds
                frequency = 440  # Hz
                t = np.linspace(0, duration, int(sr * duration), False)
                sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
                
                # Save as WAV file
                sf.write(default_ref_path, sine_wave, sr)
                logger.info(f"Created default reference audio at {default_ref_path}")
                
                reference_files.append(default_ref_path)
            except Exception as e:
                logger.error(f"Failed to create default reference audio: {e}")
        
        # If we have reference files, compute embeddings
        if reference_files:
            logger.info(f"Found {len(reference_files)} reference audio files")
            
            # Ensure speaker manager exists
            if not hasattr(synthesizer.tts_model, "speaker_manager") or synthesizer.tts_model.speaker_manager is None:
                synthesizer.tts_model.speaker_manager = SimpleSpeakerManager()
            
            # For each reference file, compute an embedding and store it
            for ref_file in reference_files:
                try:
                    # Get the speaker name from the filename
                    speaker_name = os.path.basename(ref_file).split(".")[0]
                    
                    # Skip if we already have this embedding
                    if speaker_name in synthesizer.tts_model.speaker_manager.embedding:
                        logger.info(f"Already have embedding for {speaker_name}")
                        continue
                    
                    logger.info(f"Generating embedding for {speaker_name} from {ref_file}")
                    
                    # Use the model's encoder to generate the embedding
                    if hasattr(synthesizer.tts_model, "get_conditioning_latents"):
                        # Try to load the audio and compute the embedding
                        try:
                            # This will use the model's internal functions to process the audio
                            gpt_cond_latent, speaker_embedding = synthesizer.tts_model.get_conditioning_latents(
                                ref_audio_path=ref_file
                            )
                            
                            # Store the embedding
                            synthesizer.tts_model.speaker_manager.embedding[speaker_name] = {"x_vector": speaker_embedding}
                            logger.info(f"Generated and stored embedding for {speaker_name}")
                            
                            # Make sure the speaker is in the name_to_id mapping
                            if speaker_name not in synthesizer.tts_model.speaker_manager.name_to_id:
                                next_id = len(synthesizer.tts_model.speaker_manager.name_to_id)
                                synthesizer.tts_model.speaker_manager.name_to_id[speaker_name] = next_id
                                # Don't need to append to speaker_names as it's derived from name_to_id
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for {speaker_name}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing reference file {ref_file}: {e}")
        else:
            logger.info(f"No reference audio files found in {reference_dir}. Add .wav files to generate embeddings.")
    
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
                    # To handle PyTorch 2.6 compatibility issues, we need to explicitly allow classes from TTS
                    try:
                        # Try to register XttsConfig as safe global if using PyTorch 2.6+
                        if hasattr(torch.serialization, 'add_safe_globals'):
                            from TTS.tts.configs.xtts_config import XttsConfig
                            logger.info("Adding XttsConfig to PyTorch safe globals")
                            torch.serialization.add_safe_globals([XttsConfig])
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Could not add XttsConfig to safe globals: {e}")
                    
                    # We need to make sure weights_only=False for both config and model loading
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
                    
                    if not hasattr(synthesizer.tts_config, "use_d_vector_file"):
                        logger.info("Adding 'use_d_vector_file' attribute to XttsConfig right after instantiation")
                        setattr(synthesizer.tts_config, "use_d_vector_file", False)
                    
                    # Ensure we have a speaker manager for XTTS-v2
                    if not hasattr(synthesizer.tts_model, "speaker_manager") or synthesizer.tts_model.speaker_manager is None:
                        logger.warning("Speaker manager not initialized for XTTS-v2, creating a simple one")
                        # Create and set a simple speaker manager
                        synthesizer.tts_model.speaker_manager = SimpleSpeakerManager(speakers_file=speakers_file)
                        logger.info(f"Created simple speaker manager with speakers: {list(synthesizer.tts_model.speaker_manager.name_to_id)}")
                    
                    load_time = time.time() - start_time
                    logger.info("TTS model loaded successfully", load_time=load_time)
                    return synthesizer
                except Exception as e:
                    logger.error(f"Failed to load XTTS-v2 model: {e}")
                    # Try a more direct approach with explicit weights_only=False
                    try:
                        logger.info("Attempting to load model with explicit weights_only=False")
                        # Monkey patch torch.load temporarily
                        original_torch_load = torch.load
                        
                        def patched_torch_load(f, *args, **kwargs):
                            kwargs['weights_only'] = False
                            return original_torch_load(f, *args, **kwargs)
                        
                        # Replace torch.load with our patched version
                        torch.load = patched_torch_load
                        
                        # Try loading again
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
                        if not hasattr(synthesizer.tts_config, "use_d_vector_file"):
                            logger.info("Adding 'use_d_vector_file' attribute to XttsConfig right after instantiation")
                            setattr(synthesizer.tts_config, "use_d_vector_file", False)                        
                        
                        # Restore original torch.load
                        torch.load = original_torch_load
                        
                        # Ensure we have a speaker manager for XTTS-v2
                        if not hasattr(synthesizer.tts_model, "speaker_manager") or synthesizer.tts_model.speaker_manager is None:
                            logger.warning("Speaker manager not initialized for XTTS-v2, creating a simple one")
                            # Create and set a simple speaker manager
                            synthesizer.tts_model.speaker_manager = SimpleSpeakerManager(speakers_file=speakers_file)
                            logger.info(f"Created simple speaker manager with speakers: {list(synthesizer.tts_model.speaker_manager.name_to_id)}")
                        
                        load_time = time.time() - start_time
                        logger.info("TTS model loaded successfully with patched torch.load", load_time=load_time)
                        return synthesizer
                    except Exception as e2:
                        logger.error(f"Also failed to load with patched torch.load: {e2}")
                        # Restore original torch.load if exception occurred
                        if 'original_torch_load' in locals():
                            torch.load = original_torch_load
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
                        # Try loading with standard approach
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
                        if not hasattr(synthesizer.tts_config, "use_d_vector_file"):
                            logger.info("Adding 'use_d_vector_file' attribute to XttsConfig right after instantiation")
                            setattr(synthesizer.tts_config, "use_d_vector_file", False)                        
                        
                        load_time = time.time() - start_time
                        logger.info("TTS model loaded successfully", load_time=load_time)
                        return synthesizer
                    except Exception as e:
                        logger.error(f"Failed to load model: {e}")
                        # Try with monkey-patched torch.load
                        try:
                            logger.info("Attempting to load model with explicit weights_only=False")
                            # Monkey patch torch.load temporarily
                            original_torch_load = torch.load
                            
                            def patched_torch_load(f, *args, **kwargs):
                                kwargs['weights_only'] = False
                                return original_torch_load(f, *args, **kwargs)
                            
                            # Replace torch.load with our patched version
                            torch.load = patched_torch_load
                            
                            # Try loading again
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
                            if not hasattr(synthesizer.tts_config, "use_d_vector_file"):
                                logger.info("Adding 'use_d_vector_file' attribute to XttsConfig right after instantiation")
                                setattr(synthesizer.tts_config, "use_d_vector_file", False)
                                                            
                            # Restore original torch.load
                            torch.load = original_torch_load
                            
                            # Ensure we have a speaker manager for XTTS-v2
                            if not hasattr(synthesizer.tts_model, "speaker_manager") or synthesizer.tts_model.speaker_manager is None:
                                logger.warning("Speaker manager not initialized for XTTS-v2, creating a simple one")
                                # Create and set a simple speaker manager
                                synthesizer.tts_model.speaker_manager = SimpleSpeakerManager(speakers_file=speakers_file)
                                logger.info(f"Created simple speaker manager with speakers: {list(synthesizer.tts_model.speaker_manager.name_to_id)}")
                            
                            load_time = time.time() - start_time
                            logger.info("TTS model loaded successfully with patched torch.load", load_time=load_time)
                            return synthesizer
                        except Exception as e2:
                            logger.error(f"Also failed to load with patched torch.load: {e2}")
                            # Restore original torch.load if exception occurred
                            if 'original_torch_load' in locals():
                                torch.load = original_torch_load
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
                return list(model.tts_model.speaker_manager.name_to_id)
            
            # If we can't get speakers from the model but we know it's the XTTS v2 model,
            # return a list of known XTTS speakers
            if model_name == "tts_models/multilingual/multi-dataset/xtts_v2" or (not model_name and self.config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"):
                # Return list of common XTTS speakers
                return ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Andrew Chipper"]
            
            # Return a single placeholder for single-speaker models
            return ["speaker_00"]
        except Exception as e:
            logger.warning(f"Error listing voices: {str(e)}")
            
            # If we can't get the model but know it's XTTS v2, return known speakers
            if model_name == "tts_models/multilingual/multi-dataset/xtts_v2" or (not model_name and self.config.model_name == "tts_models/multilingual/multi-dataset/xtts_v2"):
                return ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Andrew Chipper"]
                
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
        
        # For XTTS-v2, ensure we have all speaker embeddings loaded
        if "xtts_v2" in model_config.model_name:
            try:
                model = self.model_manager.get_model()
                
                # Check if we need to initialize the speaker manager
                if not hasattr(model.tts_model, "speaker_manager") or model.tts_model.speaker_manager is None:
                    logger.info("Initializing speaker manager for XTTS-v2")
                    model_dir = os.path.join(model_config.download_root, *model_config.model_name.split("/"))
                    speakers_file = os.path.join(model_dir, "speakers_xtts.pth")
                    
                    if os.path.isfile(speakers_file):
                        # Create a speaker manager with the speakers file
                        model.tts_model.speaker_manager = SimpleSpeakerManager(speakers_file=speakers_file)
                        logger.info(f"Speaker manager created with {len(model.tts_model.speaker_manager.name_to_id)} speakers")
            except Exception as e:
                logger.warning(f"Failed to initialize speaker manager: {e}")
                
        logger.info("TTS synthesizer initialized", config=model_config.model_dump())
    
    def synthesize(self, text: str, speaker: Optional[str] = None, language: Optional[str] = None, reference_wav: Optional[bytes] = None) -> np.ndarray:
        """Synthesize speech from text using the loaded model.
        
        Args:
            text: Text to synthesize
            speaker: Speaker ID or name (if supported by model)
            language: Language code (if supported by model)
            reference_wav: Reference audio as bytes (if supported by model)
            
        Returns:
            numpy array of audio samples
        """
        import os
        import tempfile
        import numpy as np
        import soundfile as sf
        
        # Start with empty result in case of errors
        result_audio = np.zeros(22050)  # 1 second of silence at 22050 Hz
        
        # Track temporary files for cleanup
        temp_files = []
        
        try:
            logger.info(f"[DEBUG] Synthesizing text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            logger.info(f"[DEBUG] Initial speaker value: '{speaker}'")
            
            # Get the model
            model = self.model_manager.get_model()
            
            # Get the model directory path
            model_dir = os.path.join(self.model_config.download_root, "tts_models", "multilingual", "multi-dataset", "xtts_v2")
            
            # Prepare reference audio (either from parameter or create a new one)
            reference_audio_path = None
            
            if reference_wav is not None:
                # Save the provided reference audio to a temporary file
                logger.info("[DEBUG] Using provided reference_wav data")
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = temp_file.name
                temp_file.write(reference_wav)
                temp_file.close()
                reference_audio_path = temp_path
                temp_files.append(temp_path)
                logger.info(f"[DEBUG] Saved reference audio to temporary file: {reference_audio_path}")
            else:
                # Look for existing reference audio files in the model's directory
                reference_dir = os.path.join(model_dir, "reference_audio")
                # Create the directory if it doesn't exist
                os.makedirs(reference_dir, exist_ok=True)
                
                # Check for existing reference files
                if os.path.exists(reference_dir):
                    for file in os.listdir(reference_dir):
                        if file.endswith(".wav") or file.endswith(".mp3"):
                            reference_audio_path = os.path.join(reference_dir, file)
                            logger.info(f"[DEBUG] Using existing reference audio file: {reference_audio_path}")
                            break
            
            # Create a new reference audio file if none was found
            if reference_audio_path is None:
                logger.info("[DEBUG] No reference audio found. Creating a default one.")
                reference_dir = os.path.join(model_dir, "reference_audio")
                os.makedirs(reference_dir, exist_ok=True)
                
                default_ref_path = os.path.join(reference_dir, "default_reference.wav")
                
                # Create a 3-second sine wave
                sr = 24000  # Sample rate
                duration = 3  # seconds
                frequency = 440  # Hz
                t = np.linspace(0, duration, int(sr * duration), False)
                sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
                
                # Save the WAV file
                sf.write(default_ref_path, sine_wave, sr)
                logger.info(f"[DEBUG] Created default reference audio at {default_ref_path}")
                reference_audio_path = default_ref_path
            
            # Set the language parameter (use language_name to match the XTTS API)
            effective_language = language or "en"
            
            # Try different approaches to synthesize speech
            
            # Approach 1: Try using model.tts directly with the speaker_wav parameter
            try:
                logger.info(f"[DEBUG] Attempt 1: Using model.tts with reference_audio_path: {reference_audio_path}")
                
                # Prepare kwargs with speaker_wav and language
                kwargs = {
                    "speaker_wav": reference_audio_path,
                    "language_name": effective_language
                }
                
                logger.info(f"[DEBUG] Calling model.tts with kwargs: {kwargs}")
                result_audio = model.tts(text=text, **kwargs)
                
                # Check if we have valid audio data
                if result_audio is not None and isinstance(result_audio, (np.ndarray, torch.Tensor)) and len(result_audio) > 0:
                    logger.info(f"[DEBUG] Approach 1 successful, got audio of shape: {result_audio.shape if hasattr(result_audio, 'shape') else 'unknown'}")
                    
                    # Convert torch tensor to numpy if needed
                    if isinstance(result_audio, torch.Tensor):
                        result_audio = result_audio.cpu().numpy()
                        
                    return result_audio
                else:
                    logger.warning(f"[DEBUG] Approach 1 returned invalid audio data: {type(result_audio)}")
            except Exception as e:
                logger.error(f"[DEBUG] Approach 1 failed with error: {str(e)}")
            
            # Approach 2: Try to access the TTS model's get_conditioning_latents and inference methods directly
            try:
                logger.info(f"[DEBUG] Attempt 2: Using direct inference with conditioning latents")
                
                if hasattr(model, "tts_model") and hasattr(model.tts_model, "get_conditioning_latents") and hasattr(model.tts_model, "inference"):
                    # Get conditioning latents from the reference audio
                    logger.info(f"[DEBUG] Getting conditioning latents from {reference_audio_path}")
                    
                    # Try with different parameter names since APIs can vary
                    gpt_cond_latent = None
                    speaker_embedding = None
                    success = False
                    
                    try:
                        logger.info("[DEBUG] Trying get_conditioning_latents with reference_wav parameter")
                        gpt_cond_latent, speaker_embedding = model.tts_model.get_conditioning_latents(reference_wav=reference_audio_path)
                        success = True
                    except (TypeError, ValueError) as e:
                        logger.warning(f"[DEBUG] Failed with reference_wav parameter: {e}, trying reference_audio")
                        try:
                            logger.info("[DEBUG] Trying get_conditioning_latents with reference_audio parameter")
                            gpt_cond_latent, speaker_embedding = model.tts_model.get_conditioning_latents(reference_audio=reference_audio_path)
                            success = True
                        except (TypeError, ValueError) as e:
                            logger.warning(f"[DEBUG] Failed with reference_audio parameter: {e}, trying with no parameter name")
                            # Try with positional argument
                            try:
                                logger.info("[DEBUG] Trying get_conditioning_latents with positional parameter")
                                gpt_cond_latent, speaker_embedding = model.tts_model.get_conditioning_latents(reference_audio_path)
                                success = True
                            except Exception as e:
                                logger.warning(f"[DEBUG] Failed with positional parameter: {e}")
                    
                    if not success:
                        logger.warning("[DEBUG] All attempts to get conditioning latents failed")
                        raise ValueError("Could not get conditioning latents with any parameter format")
                    
                    logger.info(f"[DEBUG] Successfully obtained conditioning latents")
                    
                    # Call inference directly with the conditioning latents
                    logger.info(f"[DEBUG] Calling tts_model.inference directly with conditioning latents")
                    output = model.tts_model.inference(
                        text=text,
                        language=effective_language,
                        speaker_embedding=speaker_embedding,
                        gpt_cond_latent=gpt_cond_latent
                    )
                    
                    # Process the output
                    if isinstance(output, dict):
                        logger.info(f"[DEBUG] Output is a dictionary with keys: {list(output.keys())}")
                        # Extract the audio data from the dictionary
                        if 'wav' in output:
                            result_audio = output['wav']
                            logger.info(f"[DEBUG] Found 'wav' in output dictionary")
                        elif 'audio' in output:
                            result_audio = output['audio']
                            logger.info(f"[DEBUG] Found 'audio' in output dictionary")
                        elif 'waveform' in output:
                            result_audio = output['waveform']
                            logger.info(f"[DEBUG] Found 'waveform' in output dictionary")
                        elif len(output) > 0:
                            # Try the first value
                            first_key = next(iter(output))
                            first_value = output[first_key]
                            if isinstance(first_value, (np.ndarray, torch.Tensor)):
                                result_audio = first_value
                                logger.info(f"[DEBUG] Using first value from output with key: {first_key}")
                    elif isinstance(output, (np.ndarray, torch.Tensor)):
                        result_audio = output
                        logger.info(f"[DEBUG] Output is directly audio data of type: {type(output)}")
                    
                    # Convert torch tensor to numpy if needed
                    if isinstance(result_audio, torch.Tensor):
                        result_audio = result_audio.cpu().numpy()
                        logger.info(f"[DEBUG] Converted torch tensor to numpy array")
                    
                    # Check if we have valid audio data
                    if result_audio is not None and isinstance(result_audio, np.ndarray) and len(result_audio) > 0:
                        logger.info(f"[DEBUG] Approach 2 successful, got audio of shape: {result_audio.shape}")
                        return result_audio
                    else:
                        logger.warning(f"[DEBUG] Approach 2 returned invalid audio data: {type(result_audio)}")
                else:
                    logger.warning("[DEBUG] Model doesn't have required methods for approach 2")
            except Exception as e:
                logger.error(f"[DEBUG] Approach 2 failed with error: {str(e)}")
            
            # Approach 3: Try original tts method with reference_audio parameter
            try:
                logger.info(f"[DEBUG] Attempt 3: Using model.tts with 'reference_audio' parameter")
                result_audio = model.tts(text=text, reference_audio=reference_audio_path, language_name=effective_language)
                
                # Convert torch tensor to numpy if needed
                if isinstance(result_audio, torch.Tensor):
                    result_audio = result_audio.cpu().numpy()
                    logger.info(f"[DEBUG] Approach 3: Converted torch tensor to numpy array")
                
                # Check if we have valid audio data
                if result_audio is not None and isinstance(result_audio, np.ndarray) and len(result_audio) > 0:
                    logger.info(f"[DEBUG] Approach 3 successful, got audio of shape: {result_audio.shape}")
                    return result_audio
                else:
                    logger.warning(f"[DEBUG] Approach 3 returned invalid audio data: {type(result_audio)}")
            except Exception as e:
                logger.error(f"[DEBUG] Approach 3 failed with error: {str(e)}")
            
            # Final approach: Try to use tts method with minimal parameters
            try:
                logger.info(f"[DEBUG] Final attempt: Using model.tts with minimal parameters")
                # Try with just text and language
                result_audio = model.tts(text=text, language_name=effective_language)
                
                # Convert torch tensor to numpy if needed
                if isinstance(result_audio, torch.Tensor):
                    result_audio = result_audio.cpu().numpy()
                    logger.info(f"[DEBUG] Final approach: Converted torch tensor to numpy array")
                
                # Check if we have valid audio data
                if result_audio is not None and isinstance(result_audio, np.ndarray) and len(result_audio) > 0:
                    logger.info(f"[DEBUG] Final approach successful, got audio of shape: {result_audio.shape}")
                    return result_audio
                else:
                    logger.warning(f"[DEBUG] Final approach returned invalid audio data: {type(result_audio)}")
            except Exception as e:
                logger.error(f"[DEBUG] Final approach failed with error: {str(e)}")
            
            # If all approaches failed, return silence
            logger.error("[DEBUG] All synthesis approaches failed, returning silence")
            return result_audio
            
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error in synthesize: {str(e)}")
            return result_audio
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                    logger.debug(f"[DEBUG] Deleted temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"[DEBUG] Failed to delete temp file {temp_file}: {e}")
    
    def _convert_audio(self, wav: np.ndarray, format: str, sample_rate: int) -> bytes:
        """Convert audio to the specified format."""
        import soundfile as sf
        
        try:
            # Check if wav is valid
            if wav is None:
                logger.warning(f"Audio data is None, creating silence")
                wav = np.zeros(sample_rate)  # 1 second of silence
            elif isinstance(wav, dict):
                # Try to extract audio data from dictionary
                logger.warning(f"Audio data is a dictionary with keys: {list(wav.keys())}")
                if 'wav' in wav:
                    wav = wav['wav']
                    logger.info(f"Extracted 'wav' from dictionary, new type: {type(wav)}")
                elif 'audio' in wav:
                    wav = wav['audio']
                    logger.info(f"Extracted 'audio' from dictionary, new type: {type(wav)}")
                elif 'waveform' in wav:
                    wav = wav['waveform']
                    logger.info(f"Extracted 'waveform' from dictionary, new type: {type(wav)}")
                elif len(wav) > 0:
                    # If we can't find a known key, try the first value
                    first_key = next(iter(wav))
                    first_value = wav[first_key]
                    if isinstance(first_value, (np.ndarray, torch.Tensor)):
                        wav = first_value
                        logger.info(f"Using first value with key '{first_key}', new type: {type(wav)}")
                    else:
                        logger.warning(f"First value is not an array, creating silence")
                        wav = np.zeros(sample_rate)
                else:
                    logger.warning("Dictionary is empty, creating silence")
                    wav = np.zeros(sample_rate)
            elif isinstance(wav, torch.Tensor):
                # Convert torch tensor to numpy array
                wav = wav.cpu().numpy()
                logger.info(f"Converted torch tensor to numpy array, shape: {wav.shape}")
            elif not isinstance(wav, np.ndarray):
                logger.warning(f"Invalid audio data type: {type(wav)}, creating silence")
                wav = np.zeros(sample_rate)  # 1 second of silence
            elif len(wav) == 0:
                logger.warning(f"Empty audio data, creating silence")
                wav = np.zeros(sample_rate)  # 1 second of silence
            
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Log audio data for debugging
            logger.info(f"Converting audio data, shape: {wav.shape}, min: {wav.min()}, max: {wav.max()}, format: {format}, sample_rate: {sample_rate}")
            
            # Write audio to buffer
            sf.write(buffer, wav, sample_rate, format=format)
            
            # Get bytes from buffer
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # Check for very small file size that might indicate an issue
            if len(audio_bytes) < 1000:
                logger.warning(f"Suspiciously small audio file: {len(audio_bytes)} bytes, trying manual WAV creation")
                # Try to manually create a WAV file
                audio_bytes = self._create_manual_wav(wav, sample_rate)
            
            # Log size
            logger.info(f"Converted audio to {len(audio_bytes)} bytes")
            
            return audio_bytes
        except Exception as e:
            logger.error(f"Error converting audio: {e}", exc_info=True)
            # Return manually created WAV as fallback
            return self._create_manual_wav(wav, sample_rate)
    
    def _create_manual_wav(self, wav: np.ndarray, sample_rate: int) -> bytes:
        """Create a WAV file manually, bypassing soundfile."""
        logger.info("Creating WAV file manually")
        try:
            # Ensure we have valid audio data
            if wav is None or not isinstance(wav, np.ndarray):
                logger.warning("Invalid audio data for manual WAV creation, using silence")
                wav = np.zeros(sample_rate)
            
            # Normalize audio to -1.0 to 1.0 range if it's not already
            if wav.max() > 1.0 or wav.min() < -1.0:
                logger.info("Normalizing audio data to -1.0 to 1.0 range")
                max_val = max(abs(wav.max()), abs(wav.min()))
                if max_val > 0:
                    wav = wav / max_val
            
            # Convert to 16-bit PCM
            wav_pcm = (wav * 32767).astype(np.int16)
            
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Write WAV header
            # RIFF header
            buffer.write(b'RIFF')
            buffer.write((36 + len(wav_pcm) * 2).to_bytes(4, byteorder='little'))  # File size - 8
            buffer.write(b'WAVE')
            
            # fmt chunk
            buffer.write(b'fmt ')
            buffer.write((16).to_bytes(4, byteorder='little'))  # Chunk size
            buffer.write((1).to_bytes(2, byteorder='little'))  # Format code (PCM)
            buffer.write((1).to_bytes(2, byteorder='little'))  # Channels (mono)
            buffer.write(sample_rate.to_bytes(4, byteorder='little'))  # Sample rate
            buffer.write((sample_rate * 2).to_bytes(4, byteorder='little'))  # Byte rate
            buffer.write((2).to_bytes(2, byteorder='little'))  # Block align
            buffer.write((16).to_bytes(2, byteorder='little'))  # Bits per sample
            
            # data chunk
            buffer.write(b'data')
            buffer.write((len(wav_pcm) * 2).to_bytes(4, byteorder='little'))  # Chunk size
            
            # Write audio data
            buffer.write(wav_pcm.tobytes())
            
            # Get bytes from buffer
            buffer.seek(0)
            wav_bytes = buffer.read()
            
            logger.info(f"Manually created WAV file: {len(wav_bytes)} bytes")
            return wav_bytes
        except Exception as e:
            logger.error(f"Error creating manual WAV: {e}", exc_info=True)
            # Last resort: return a minimal valid WAV file
            return self._create_empty_wav(sample_rate)
    
    def _create_empty_wav(self, sample_rate: int) -> bytes:
        """Create an empty WAV file with 1 second of silence."""
        try:
            import soundfile as sf
            
            # Create silence
            silence = np.zeros(sample_rate)
            
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Write audio to buffer
            sf.write(buffer, silence, sample_rate, format="wav")
            
            # Get bytes from buffer
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            logger.info(f"Created empty WAV file, size: {len(audio_bytes)} bytes")
            
            return audio_bytes
        except Exception as e:
            logger.error(f"Error creating empty WAV: {e}", exc_info=True)
            # Return minimal valid WAV header for 1 second of silence at given sample rate
            # This is a manually constructed minimal WAV header for PCM format
            header_size = 44
            data_size = sample_rate * 2  # 16-bit mono
            total_size = header_size + data_size
            
            header = bytearray([
                # RIFF header
                0x52, 0x49, 0x46, 0x46,  # "RIFF"
                (total_size - 8) & 0xff, ((total_size - 8) >> 8) & 0xff, ((total_size - 8) >> 16) & 0xff, ((total_size - 8) >> 24) & 0xff,  # Size
                0x57, 0x41, 0x56, 0x45,  # "WAVE"
                
                # fmt chunk
                0x66, 0x6d, 0x74, 0x20,  # "fmt "
                0x10, 0x00, 0x00, 0x00,  # Chunk size: 16
                0x01, 0x00,              # Format: PCM
                0x01, 0x00,              # Channels: 1
                sample_rate & 0xff, (sample_rate >> 8) & 0xff, (sample_rate >> 16) & 0xff, (sample_rate >> 24) & 0xff,  # Sample rate
                (sample_rate * 2) & 0xff, ((sample_rate * 2) >> 8) & 0xff, ((sample_rate * 2) >> 16) & 0xff, ((sample_rate * 2) >> 24) & 0xff,  # Byte rate
                0x02, 0x00,              # Block align
                0x10, 0x00,              # Bits per sample: 16
                
                # data chunk
                0x64, 0x61, 0x74, 0x61,  # "data"
                data_size & 0xff, (data_size >> 8) & 0xff, (data_size >> 16) & 0xff, (data_size >> 24) & 0xff  # Size
            ])
            
            # Add silence data (all zeros)
            silence_data = bytearray(data_size)
            
            return bytes(header + silence_data) 

    def get_available_speakers(self) -> List[str]:
        """Get the list of available speakers for this model."""
        import os
        import torch
        
        # Default to empty list of speakers
        speakers = []
        
        # Try to get the model to access its path
        try:
            model = self.model_manager.get_model()
            
            # First check the speaker_manager.speakers dictionary directly
            # This is the most reliable source and avoids the KeyError issue
            if hasattr(model.tts_model, "speaker_manager") and model.tts_model.speaker_manager:
                if hasattr(model.tts_model.speaker_manager, "speakers"):
                    speakers_dict = getattr(model.tts_model.speaker_manager, "speakers", {})
                    if speakers_dict:
                        # Get speakers from the dictionary keys
                        speakers = list(speakers_dict.keys())
                        logger.info(f"Found {len(speakers)} speakers directly from speaker_manager.speakers")
                        
                        # Sort the speakers for consistent ordering
                        speakers.sort()
                        return speakers
            
            # If no speakers from direct dictionary, try the standard approach
            # Path to the speakers file (look in the same directory as the model)
            if hasattr(model, "output_path"):
                model_dir = os.path.dirname(model.output_path)
                speaker_file_path = os.path.join(model_dir, "speakers_xtts.pth")
                
                # Check if the speakers file exists
                if os.path.exists(speaker_file_path):
                    try:
                        # Load the embeddings
                        logger.info(f"Loading speaker data from {speaker_file_path}")
                        speaker_data = torch.load(speaker_file_path)
                        
                        # Get the list of speakers
                        speakers = list(speaker_data.keys())
                        logger.info(f"Found {len(speakers)} speakers in {speaker_file_path}")
                        
                        # Sort the speakers for consistent ordering
                        speakers.sort()
                    except Exception as e:
                        logger.error(f"Error loading speakers from {speaker_file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting model for speakers: {str(e)}")
        
        # If we couldn't get speakers from the model path, try the standard location
        if not speakers:
            # Path for XTTS-v2 models
            model_dir = os.path.join(self.model_config.download_root, "tts_models", "multilingual", "multi-dataset", "xtts_v2")
            speaker_file_path = os.path.join(model_dir, "speakers_xtts.pth")
            
            if os.path.exists(speaker_file_path):
                try:
                    # Load the embeddings from the default location
                    logger.info(f"Loading speaker data from default location: {speaker_file_path}")
                    speaker_data = torch.load(speaker_file_path)
                    
                    # Get the list of speakers
                    speakers = list(speaker_data.keys())
                    logger.info(f"Found {len(speakers)} speakers in default location")
                    
                    # Sort the speakers for consistent ordering
                    speakers.sort()
                except Exception as e:
                    logger.error(f"Error loading speakers from default location: {str(e)}")
        
        # If we still don't have speakers, use the list_available_voices method as fallback
        if not speakers:
            logger.info("Using list_available_voices as fallback")
            speakers = self.model_manager.list_available_voices()
            
        return speakers