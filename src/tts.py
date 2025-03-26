import time
from src.logging_setup import get_logger
from src.config import SynthesisOptions
from src.models.tts_model import TTSModelManager

# Get logger
logger = get_logger(__name__)

class TTSSynthesizer:
    def __init__(self, model_config, model_manager=None):
        self.model_config = model_config
        self.model_manager = model_manager if model_manager is not None else TTSModelManager(model_config)

    def _convert_audio(self, wav, format, sample_rate):
        """Convert audio to the specified format."""
        import io
        import numpy as np
        import soundfile as sf
        
        try:
            logger.info(f"Converting audio format={format}, sample_rate={sample_rate}, wav_shape={wav.shape if isinstance(wav, np.ndarray) else 'not array'}")
            
            # Ensure wav is a numpy array
            if not isinstance(wav, np.ndarray):
                logger.warning(f"Expected numpy array but got {type(wav)}, attempting to convert")
                if wav is None:
                    logger.error("Received None for wav, returning empty audio")
                    return b''
                try:
                    wav = np.array(wav)
                except:
                    logger.error(f"Could not convert {type(wav)} to numpy array, returning empty audio")
                    return b''
            
            # Check if audio data is valid
            if len(wav) == 0:
                logger.warning("Empty audio data received, returning empty bytes")
                return b''
            
            # Create in-memory buffer
            buffer = io.BytesIO()
            
            # Write audio to buffer
            sf.write(buffer, wav, sample_rate, format=format)
            
            # Get bytes from buffer
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # Log audio size
            logger.info(f"Converted audio to {format}, size: {len(audio_bytes)} bytes")
            
            return audio_bytes
        except Exception as e:
            logger.error(f"Error converting audio: {e}", exc_info=True)
            # Return empty audio in case of error
            return b''

    def synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> bytes:
        """Synthesize speech from text."""
        start_time = time.time()
        
        try:
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
                
                # Apply emotion prompt if specified
                if options.emotion and options.emotion in emotion_prompts:
                    modified_text = f"{emotion_prompts[options.emotion]}{modified_text}"
                    logger.info(
                        "Applied emotion prompt",
                        emotion=options.emotion,
                        prompt=emotion_prompts[options.emotion]
                    )
                
                # Apply style prompt if specified
                if options.style and options.style in style_prompts:
                    modified_text = f"{style_prompts[options.style]}{modified_text}"
                    logger.info(
                        "Applied style prompt",
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
                model=self.model_config.model_name
            )
            
            # Call the model manager's synthesize method
            wav = self.model_manager.synthesize(
                text=modified_text,
                speaker=speaker,
                language=language
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