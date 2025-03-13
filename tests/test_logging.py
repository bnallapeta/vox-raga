"""
Tests for logging module.
"""
import logging
import pytest
from unittest.mock import patch, MagicMock

import structlog

from src.logging_setup import configure_logging, get_logger, add_timestamp, add_service_name


class LogCapture:
    """Capture logs for testing."""
    
    def __init__(self):
        self.logs = []
    
    def __call__(self, logger, method_name, event_dict):
        """Capture the log."""
        self.logs.append(event_dict)
        return event_dict


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
    with patch("src.logging_setup.structlog.configure") as mock_configure:
        # Call the function
        configure_logging("info")
        
        # Verify that structlog.configure was called
        assert mock_configure.called
        
        # Get the call arguments
        args, kwargs = mock_configure.call_args
        
        # Verify the processors
        assert "processors" in kwargs
        processors = kwargs["processors"]
        
        # Check that our custom processors are included
        processor_names = [p.__name__ if hasattr(p, "__name__") else str(p) for p in processors]
        assert "add_service_name" in processor_names
        # TimeStamper is used instead of add_timestamp
        assert any("TimeStamper" in p for p in processor_names)


@pytest.mark.unit
@pytest.mark.utils
def test_get_logger():
    """Test get_logger function."""
    with patch("src.logging_setup.structlog.get_logger") as mock_get_logger:
        # Set up the mock
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call the function
        logger = get_logger("test_module")
        
        # Verify that structlog.get_logger was called with the right name
        mock_get_logger.assert_called_once_with("test_module")
        
        # Verify that the returned logger is the mock
        assert logger == mock_logger


@pytest.mark.unit
@pytest.mark.utils
def test_logger_output():
    """Test logger output."""
    # Configure logging with a log capture
    log_capture = LogCapture()
    
    # Use a direct approach instead of patching structlog.configure
    # This avoids the recursion issue
    original_configure = structlog.configure
    
    try:
        # Define a custom processors list with our log capture
        def custom_configure(**kwargs):
            processors = kwargs.get('processors', [])
            kwargs['processors'] = [log_capture] + processors
            return original_configure(**kwargs)
        
        # Replace structlog.configure temporarily
        structlog.configure = custom_configure
        
        # Configure logging
        configure_logging("debug")
        
        # Get a logger
        logger = get_logger("test_logger")
        
        # Log some messages
        logger.debug("Debug message", key1="value1")
        logger.info("Info message", key2="value2")
        logger.warning("Warning message", key3="value3")
        logger.error("Error message", key4="value4")
        
        # Verify that the logs were captured
        assert len(log_capture.logs) == 4
        
        # Check the log contents
        assert log_capture.logs[0]["event"] == "Debug message"
        assert log_capture.logs[0]["key1"] == "value1"
        assert log_capture.logs[0]["level"] == "debug"
        
        assert log_capture.logs[1]["event"] == "Info message"
        assert log_capture.logs[1]["key2"] == "value2"
        assert log_capture.logs[1]["level"] == "info"
        
        assert log_capture.logs[2]["event"] == "Warning message"
        assert log_capture.logs[2]["key3"] == "value3"
        assert log_capture.logs[2]["level"] == "warning"
        
        assert log_capture.logs[3]["event"] == "Error message"
        assert log_capture.logs[3]["key4"] == "value4"
        assert log_capture.logs[3]["level"] == "error"
    finally:
        # Restore the original configure function
        structlog.configure = original_configure 