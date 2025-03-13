"""
Tests for logging setup.
"""
import logging
import pytest
from unittest.mock import patch, MagicMock

import structlog
from structlog.testing import LogCapture

from src.logging_setup import configure_logging, get_logger, add_timestamp, add_service_name


@pytest.mark.unit
@pytest.mark.utils
def test_add_timestamp():
    """Test add_timestamp processor."""
    event_dict = {}
    result = add_timestamp(None, None, event_dict)
    
    assert "timestamp" in result
    assert isinstance(result["timestamp"], float)


@pytest.mark.unit
@pytest.mark.utils
def test_add_service_name():
    """Test add_service_name processor."""
    event_dict = {}
    result = add_service_name(None, None, event_dict)
    
    assert "service" in result
    assert result["service"] == "tts-service"


@pytest.mark.unit
@pytest.mark.utils
def test_configure_logging():
    """Test configure_logging function."""
    with patch("src.logging_setup.logging.basicConfig") as mock_basic_config:
        with patch("src.logging_setup.structlog.configure") as mock_configure:
            # Test with valid log level
            configure_logging("INFO")
            
            # Verify logging.basicConfig was called with correct parameters
            mock_basic_config.assert_called_once()
            args, kwargs = mock_basic_config.call_args
            assert kwargs["level"] == logging.INFO
            assert kwargs["format"] == "%(message)s"
            assert kwargs["stream"] is not None
            
            # Verify structlog.configure was called
            mock_configure.assert_called_once()
            
            # Test with invalid log level
            mock_basic_config.reset_mock()
            mock_configure.reset_mock()
            
            with pytest.raises(ValueError):
                configure_logging("INVALID_LEVEL")


@pytest.mark.unit
@pytest.mark.utils
def test_get_logger():
    """Test get_logger function."""
    with patch("src.logging_setup.structlog.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        logger = get_logger("test_logger")
        
        mock_get_logger.assert_called_once_with("test_logger")
        assert logger == mock_logger


@pytest.mark.unit
@pytest.mark.utils
def test_logger_output():
    """Test logger output."""
    # Configure logging with a log capture
    log_capture = LogCapture()
    with patch("src.logging_setup.structlog.configure") as mock_configure:
        def configure_with_capture(processors, **kwargs):
            # Replace the original configure with one that uses our log capture
            structlog.configure(
                processors=[log_capture] + processors,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        mock_configure.side_effect = configure_with_capture
        
        # Configure logging
        configure_logging("INFO")
        
        # Get a logger and log a message
        logger = get_logger("test_logger")
        logger.info("Test message", key1="value1", key2=123)
        
        # Check the captured log
        assert len(log_capture.entries) == 1
        entry = log_capture.entries[0]
        
        assert entry["event"] == "Test message"
        assert entry["key1"] == "value1"
        assert entry["key2"] == 123
        assert "timestamp" in entry
        assert entry["service"] == "tts-service"
        assert entry["level"] == "info"
        assert entry["logger"] == "test_logger" 