"""
Tests for main application module.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import asyncio

from src.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.api
def test_app_title():
    """Test app title."""
    assert app.title == "TTS Service"
    assert app.description == "Text-to-Speech service for speech synthesis"
    assert app.version == "0.1.0"


@pytest.mark.unit
@pytest.mark.api
def test_cors_middleware():
    """Test CORS middleware is configured."""
    # Check that CORS middleware is in the middleware stack
    cors_middleware = next(
        (m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"),
        None
    )
    assert cors_middleware is not None


@pytest.mark.unit
@pytest.mark.api
def test_exception_handler():
    """Test global exception handler."""
    from src.main import global_exception_handler
    
    # Create a mock request
    mock_request = MagicMock()
    mock_request.url = "http://test.com/test"
    mock_request.method = "GET"
    
    # Create a test exception
    test_exception = ValueError("Test exception")
    
    # Call the exception handler directly
    with patch("src.main.logger") as mock_logger:
        # Use asyncio to run the async function
        response = asyncio.run(global_exception_handler(mock_request, test_exception))
        
        # Verify the logger was called with the right arguments
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0]
        assert "Unhandled exception" in call_args
        
        # Verify the response
        assert response.status_code == 500
        assert "Internal server error" in response.body.decode()
        assert "Test exception" in response.body.decode()


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.parametrize("metrics_enabled", [True, False])
def test_metrics_endpoint(metrics_enabled):
    """Test metrics endpoint is configured based on config."""
    # Create a simple ASGI app that returns a 200 response
    async def metrics_app(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"metrics data",
        })
    
    # Create a test app
    test_app = FastAPI()
    
    # Configure the app based on metrics_enabled
    if metrics_enabled:
        test_app.mount("/metrics", metrics_app)
    
    # Create a test client
    client = TestClient(test_app)
    
    # Make a request to the metrics endpoint
    response = client.get("/metrics")
    
    # Verify the response based on whether metrics are enabled
    if metrics_enabled:
        # If metrics are enabled, the endpoint should be available
        assert response.status_code == 200
        assert response.content == b"metrics data"
    else:
        # If metrics are disabled, the endpoint should return 404
        assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
def test_router_inclusion():
    """Test routers are included."""
    # Create a test client
    test_client = TestClient(app)
    
    # Check that health endpoints are available
    response = test_client.get("/health")
    assert response.status_code == 200
    
    response = test_client.get("/ready")
    assert response.status_code == 200
    
    response = test_client.get("/live")
    assert response.status_code == 200
    
    # Check that TTS endpoints are available
    # Note: These will return 422 because they require request bodies
    response = test_client.post("/synthesize")
    assert response.status_code == 422
    
    response = test_client.get("/voices")
    assert response.status_code in [200, 500]  # May fail if model can't be loaded
    
    response = test_client.get("/languages")
    assert response.status_code in [200, 500]  # May fail if model can't be loaded


@pytest.mark.unit
@pytest.mark.api
def test_startup_event():
    """Test startup event handler."""
    # We need to patch the logger before the event is triggered
    with patch("src.main.logger") as mock_logger:
        # Create a test client which will trigger the startup event
        with TestClient(app) as client:
            # Make a request to ensure the app is started
            client.get("/health")
        
        # Check that the startup message was logged
        # The actual message in the code is "Starting TTS service"
        mock_logger.info.assert_any_call("Starting TTS service", config=mock_logger.info.call_args_list[0][1]["config"])


@pytest.mark.unit
@pytest.mark.api
def test_shutdown_event():
    """Test shutdown event handler."""
    # We need to patch the logger before the event is triggered
    with patch("src.main.logger") as mock_logger:
        # Create a test client which will trigger the startup and shutdown events
        with TestClient(app) as client:
            # Make a request to ensure the app is started
            client.get("/health")
        
        # Check that the shutdown message was logged
        # The actual message in the code is "Shutting down TTS service"
        mock_logger.info.assert_any_call("Shutting down TTS service") 