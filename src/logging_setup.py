"""
Logging setup for the TTS service.
"""
import logging
import sys
import time
from typing import Dict, Any, Optional

import structlog
from structlog.types import Processor


def add_timestamp(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add timestamp to the event dict."""
    event_dict["timestamp"] = time.time()
    return event_dict


def add_service_name(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service name to the event dict."""
    event_dict["service"] = "tts-service"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    # Convert string log level to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )
    
    # Configure structlog
    processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_service_name,
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger with the given name."""
    return structlog.get_logger(name)
