"""
Health check API endpoints.
"""
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from src.logging_setup import get_logger

# Get logger
logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str = "0.1.0"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    """Readiness check endpoint."""
    logger.debug("Readiness check requested")
    return HealthResponse(status="ready")


@router.get("/live", response_model=HealthResponse)
async def liveness_check() -> HealthResponse:
    """Liveness check endpoint."""
    logger.debug("Liveness check requested")
    return HealthResponse(status="alive") 