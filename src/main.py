"""
Main application module for the TTS service.
"""
import os
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    """
    # Startup
    logger.info("Starting TTS service", config=config.model_dump())
    
    # Preload model
    model_manager = TTSModelManager(config.model)
    model_manager.get_model()
    
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
