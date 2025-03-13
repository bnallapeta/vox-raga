"""
TTS API endpoints for speech synthesis.
"""
import io
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Response
from fastapi.responses import StreamingResponse

from src.config import SynthesisRequest, SynthesisOptions, config
from src.models.tts_model import TTSSynthesizer, TTSModelManager
from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["tts"])

# Create TTS synthesizer
synthesizer = TTSSynthesizer(config.model)
model_manager = TTSModelManager(config.model)


@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    Synthesize speech from text.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Perform synthesis
        audio_bytes = synthesizer.synthesize(
            text=request.text,
            options=request.options,
        )
        
        # Record latency
        latency = time.time() - start_time
        
        # Determine content type based on format
        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg"
        }.get(request.options.format, "application/octet-stream")
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers={
                "X-Processing-Time": str(latency),
                "X-Language": request.options.language,
                "X-Voice": request.options.voice
            }
        )
    except Exception as e:
        # Log error
        logger.error("Synthesis error", error=str(e), exc_info=True)
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis error: {str(e)}"
        )


@router.get("/voices")
async def list_voices() -> Dict[str, List[str]]:
    """
    List available voices.
    """
    try:
        voices = model_manager.list_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error("Failed to list voices", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list voices: {str(e)}"
        )


@router.get("/languages")
async def list_languages() -> Dict[str, List[str]]:
    """
    List available languages.
    """
    try:
        languages = model_manager.list_available_languages()
        return {"languages": languages}
    except Exception as e:
        logger.error("Failed to list languages", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list languages: {str(e)}"
        )


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current configuration.
    """
    return {
        "model": config.model.model_dump(),
        "server": config.server.model_dump(),
    } 