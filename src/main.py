"""
Main application module for the TTS service.
"""
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api import health, tts
from src.config import config
from src.logging_setup import configure_logging, get_logger
from src.models.tts_model import TTSModelManager

# Configure logging
configure_logging(config.server.log_level)

# Get logger
logger = get_logger(__name__)

# Apply monkey patch for torch.load to handle weights_only
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

# Apply the monkey patch
torch.load = patched_torch_load
logger.info("Applied monkey patch for torch.load to set weights_only=False by default")

# Try to register TTS models with PyTorch safe globals if using PyTorch 2.6+
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        logger.info("Adding TTS model classes to PyTorch safe globals")
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
            logger.info("Added XttsConfig to PyTorch safe globals")
        except ImportError:
            logger.warning("Could not import XttsConfig")
except Exception as e:
    logger.warning(f"Failed to add TTS model classes to PyTorch safe globals: {e}")

# Create necessary directories
os.makedirs(config.server.cache_dir, exist_ok=True)
os.makedirs(config.model.download_root, exist_ok=True)

# Log paths and environment
logger.info(f"Cache directory: {os.path.abspath(config.server.cache_dir)}")
logger.info(f"Models directory: {os.path.abspath(config.model.download_root)}")
logger.info(f"Running with device: {config.model.device}")
logger.info(f"Model name: {config.model.model_name}")

# Check if we're in a container
in_container = os.path.exists("/.dockerenv")
logger.info(f"Running in container: {in_container}")

# Log environment for debugging
for env_var in ['MODEL_DOWNLOAD_ROOT', 'SERVER_CACHE_DIR', 'MODEL_NAME', 'MODEL_DEVICE']:
    if env_var in os.environ:
        logger.info(f"{env_var} set to: {os.environ[env_var]}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    """
    # Startup
    logger.info("Starting TTS service", config=config.model_dump())
    
    # Preload model - try, but continue even if it fails to allow for fallbacks
    model_manager = TTSModelManager(config.model)
    try:
        logger.info("Pre-loading TTS model at startup")
        model_manager.get_model()
        logger.info("TTS model pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load TTS model at startup: {str(e)}", exc_info=True)
        logger.warning("Service will continue to start, but TTS functionality may not work correctly until models are available")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TTS service")


# Create FastAPI app
app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service for speech synthesis",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        url=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# Add metrics endpoint
if config.server.metrics_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router)
app.include_router(tts.router)

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("TTS service started")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("TTS service stopped")


if __name__ == "__main__":
    """Run the application directly."""
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
