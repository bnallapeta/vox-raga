"""
TTS API endpoints for speech synthesis.
"""
import io
import time
import zipfile
import uuid
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse

from src.config import SynthesisRequest, BatchSynthesisRequest, SynthesisOptions, config
from src.models.tts_model import TTSSynthesizer, TTSModelManager
from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["tts"])

# Create TTS synthesizer
synthesizer = TTSSynthesizer(config.model)
model_manager = TTSModelManager(config.model)

# In-memory storage for async synthesis jobs
# In a production system, this would be replaced with a proper database or queue
synthesis_jobs = {}


@router.websocket("/synthesize/ws")
async def websocket_synthesize(websocket: WebSocket):
    """
    WebSocket endpoint for streaming synthesis.
    
    The client should send a JSON message with the following format:
    {
        "text": "Text to synthesize",
        "options": {
            "language": "en",
            "voice": "default",
            "speed": 1.0,
            "format": "wav"
        }
    }
    
    The server will respond with a binary message containing the audio data.
    """
    await websocket.accept()
    
    try:
        # Receive the request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        # Validate the request
        if "text" not in request_data:
            await websocket.send_json({"error": "Missing 'text' field"})
            await websocket.close()
            return
        
        # Create the request object
        text = request_data["text"]
        options = SynthesisOptions(**(request_data.get("options", {})))
        
        # Log the request
        logger.info(
            "WebSocket synthesis request",
            text_length=len(text),
            language=options.language,
            voice=options.voice,
        )
        
        # Perform synthesis
        audio_bytes = synthesizer.synthesize(
            text=text,
            options=options,
        )
        
        # Send the audio data
        await websocket.send_bytes(audio_bytes)
        
        # Close the connection
        await websocket.close()
        
        logger.info("WebSocket synthesis completed")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in WebSocket request")
        await websocket.send_json({"error": "Invalid JSON"})
        await websocket.close()
    except Exception as e:
        logger.error("WebSocket synthesis error", error=str(e), exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except:
            # Client might have already disconnected
            pass


@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    Synthesize speech from text.
    
    Returns an audio stream in the requested format.
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


@router.post("/synthesize/async")
async def synthesize_speech_async(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Asynchronously synthesize speech from text.
    
    Returns a job ID that can be used to check the status and retrieve the result.
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create a job entry
    synthesis_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.model_dump(),
        "result": None,
        "error": None,
    }
    
    # Start the synthesis task in the background
    background_tasks.add_task(
        process_synthesis_job,
        job_id=job_id,
        text=request.text,
        options=request.options,
    )
    
    # Return the job ID
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "message": "Synthesis job started",
        }
    )


@router.get("/synthesize/status/{job_id}")
async def get_synthesis_status(job_id: str) -> JSONResponse:
    """
    Get the status of an asynchronous synthesis job.
    """
    # Check if the job exists
    if job_id not in synthesis_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    # Get the job
    job = synthesis_jobs[job_id]
    
    # Return the status
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "error": job["error"],
        }
    )


@router.get("/synthesize/result/{job_id}")
async def get_synthesis_result(job_id: str) -> StreamingResponse:
    """
    Get the result of an asynchronous synthesis job.
    """
    # Check if the job exists
    if job_id not in synthesis_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    # Get the job
    job = synthesis_jobs[job_id]
    
    # Check if the job is completed
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed: {job_id}, current status: {job['status']}"
        )
    
    # Check if there's a result
    if not job["result"]:
        raise HTTPException(
            status_code=500,
            detail=f"No result available for job: {job_id}"
        )
    
    # Get the request options
    options = SynthesisOptions(**job["request"]["options"])
    
    # Determine content type based on format
    content_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg"
    }.get(options.format, "application/octet-stream")
    
    # Return the result
    return StreamingResponse(
        io.BytesIO(job["result"]),
        media_type=content_type,
        headers={
            "X-Language": options.language,
            "X-Voice": options.voice
        }
    )


async def process_synthesis_job(job_id: str, text: str, options: SynthesisOptions):
    """
    Process a synthesis job asynchronously.
    """
    try:
        # Update job status
        synthesis_jobs[job_id]["status"] = "processing"
        
        # Perform synthesis
        audio_bytes = synthesizer.synthesize(
            text=text,
            options=options,
        )
        
        # Update job with result
        synthesis_jobs[job_id]["status"] = "completed"
        synthesis_jobs[job_id]["result"] = audio_bytes
        
        logger.info("Async synthesis completed", job_id=job_id)
    except Exception as e:
        # Log error
        logger.error("Async synthesis error", job_id=job_id, error=str(e), exc_info=True)
        
        # Update job with error
        synthesis_jobs[job_id]["status"] = "failed"
        synthesis_jobs[job_id]["error"] = str(e)


@router.post("/batch_synthesize")
async def batch_synthesize_speech(
    request: BatchSynthesisRequest,
    background_tasks: BackgroundTasks,
) -> StreamingResponse:
    """
    Synthesize speech from multiple texts.
    
    Returns a ZIP file containing audio files for each text.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Process each text
            for i, text in enumerate(request.texts):
                # Perform synthesis
                audio_bytes = synthesizer.synthesize(
                    text=text,
                    options=request.options,
                )
                
                # Add to ZIP file
                filename = f"audio_{i+1}.{request.options.format}"
                zip_file.writestr(filename, audio_bytes)
        
        # Record latency
        latency = time.time() - start_time
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Return ZIP file
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "X-Processing-Time": str(latency),
                "X-Language": request.options.language,
                "X-Voice": request.options.voice,
                "Content-Disposition": f"attachment; filename=batch_synthesis.zip"
            }
        )
    except Exception as e:
        # Log error
        logger.error("Batch synthesis error", error=str(e), exc_info=True)
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Batch synthesis error: {str(e)}"
        )


@router.post("/batch_synthesize/async")
async def batch_synthesize_speech_async(
    request: BatchSynthesisRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Asynchronously synthesize speech from multiple texts.
    
    Returns a job ID that can be used to check the status and retrieve the result.
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Create a job entry
    synthesis_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.model_dump(),
        "result": None,
        "error": None,
        "is_batch": True,
    }
    
    # Start the synthesis task in the background
    background_tasks.add_task(
        process_batch_synthesis_job,
        job_id=job_id,
        texts=request.texts,
        options=request.options,
    )
    
    # Return the job ID
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "message": "Batch synthesis job started",
        }
    )


@router.get("/batch_synthesize/result/{job_id}")
async def get_batch_synthesis_result(job_id: str) -> StreamingResponse:
    """
    Get the result of an asynchronous batch synthesis job.
    """
    # Check if the job exists
    if job_id not in synthesis_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    # Get the job
    job = synthesis_jobs[job_id]
    
    # Check if the job is a batch job
    if not job.get("is_batch", False):
        raise HTTPException(
            status_code=400,
            detail=f"Not a batch job: {job_id}"
        )
    
    # Check if the job is completed
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed: {job_id}, current status: {job['status']}"
        )
    
    # Check if there's a result
    if not job["result"]:
        raise HTTPException(
            status_code=500,
            detail=f"No result available for job: {job_id}"
        )
    
    # Return the result
    return StreamingResponse(
        io.BytesIO(job["result"]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=batch_synthesis_{job_id}.zip"
        }
    )


async def process_batch_synthesis_job(job_id: str, texts: List[str], options: SynthesisOptions):
    """
    Process a batch synthesis job asynchronously.
    """
    try:
        # Update job status
        synthesis_jobs[job_id]["status"] = "processing"
        
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Process each text
            for i, text in enumerate(texts):
                # Perform synthesis
                audio_bytes = synthesizer.synthesize(
                    text=text,
                    options=options,
                )
                
                # Add to ZIP file
                filename = f"audio_{i+1}.{options.format}"
                zip_file.writestr(filename, audio_bytes)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Update job with result
        synthesis_jobs[job_id]["status"] = "completed"
        synthesis_jobs[job_id]["result"] = zip_buffer.getvalue()
        
        logger.info("Async batch synthesis completed", job_id=job_id)
    except Exception as e:
        # Log error
        logger.error("Async batch synthesis error", job_id=job_id, error=str(e), exc_info=True)
        
        # Update job with error
        synthesis_jobs[job_id]["status"] = "failed"
        synthesis_jobs[job_id]["error"] = str(e)


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


@router.get("/voices/{language}")
async def list_voices_by_language(language: str) -> Dict[str, List[str]]:
    """
    List available voices for a specific language.
    """
    try:
        # Get all voices
        all_voices = model_manager.list_available_voices()
        
        # Filter voices by language
        # This is a simplified implementation - in a real system, you would have a mapping
        # of voices to languages or query the model for language-specific voices
        language_voices = []
        
        # For VCTK model, all voices are English
        if language.lower() == "en" and "p" in all_voices[0]:
            language_voices = all_voices
        # For multilingual models, we would have a more sophisticated mapping
        elif language.lower() in ["fr", "de", "es", "it"]:
            # Return a subset of voices that might support this language
            # This is just a placeholder - real implementation would depend on the model
            language_voices = all_voices[:3] if all_voices else []
        
        return {"voices": language_voices}
    except Exception as e:
        logger.error("Failed to list voices by language", language=language, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list voices for language {language}: {str(e)}"
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