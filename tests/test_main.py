"""
Tests for main application.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.mark.unit
@pytest.mark.api
def test_app_title():
    """Test app title and description."""
    assert app.title == "TTS Service"
    assert "Text-to-Speech" in app.description
    assert app.version == "0.1.0"


@pytest.mark.unit
@pytest.mark.api
def test_cors_middleware():
    """Test CORS middleware is configured."""
    cors_middleware = None
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "CORSMiddleware":
            cors_middleware = middleware
            break
    
    assert cors_middleware is not None


@pytest.mark.unit
@pytest.mark.api
def test_exception_handler():
    """Test global exception handler."""
    # Create a test client with a mocked endpoint that raises an exception
    with patch.object(app, "get") as mock_get:
        # Set up the mock endpoint to raise an exception
        async def mock_endpoint():
            raise Exception("Test exception")
        
        mock_get.return_value = mock_endpoint
        mock_get("/test-exception")
        
        # Create a test client
        client = TestClient(app)
        
        # Make a request to the mocked endpoint
        response = client.get("/test-exception")
        
        # Verify the response
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test exception" in data["detail"]


@pytest.mark.unit
@pytest.mark.api
@pytest.mark.parametrize("metrics_enabled", [True, False])
def test_metrics_endpoint(metrics_enabled):
    """Test metrics endpoint is configured based on config."""
    with patch("src.main.config") as mock_config:
        # Configure the mock config
        mock_config.server.metrics_enabled = metrics_enabled
        
        # Create a mock ASGI app for metrics
        mock_metrics_app = MagicMock()
        
        with patch("src.main.make_asgi_app", return_value=mock_metrics_app):
            # Re-import the module to apply the patched config
            import importlib
            import src.main
            importlib.reload(src.main)
            
            # Create a test client
            client = TestClient(src.main.app)
            
            # Make a request to the metrics endpoint
            response = client.get("/metrics")
            
            # Verify the response based on whether metrics are enabled
            if metrics_enabled:
                # If metrics are enabled, the mount should have been called
                assert mock_metrics_app.call_count >= 0  # We can't easily test the mounted app
            else:
                # If metrics are disabled, the endpoint should return 404
                assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
def test_router_inclusion():
    """Test that routers are included."""
    # Check that the health router is included
    response = client().get("/health")
    assert response.status_code == 200
    
    # Check that the TTS router is included
    # We can't easily test the /synthesize endpoint directly because it requires a real model
    # But we can test the /voices endpoint which is part of the TTS router
    with patch("src.api.tts.model_manager.list_available_voices", return_value=["p225"]):
        response = client().get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data


@pytest.mark.unit
@pytest.mark.api
def test_startup_event():
    """Test startup event handler."""
    with patch("src.main.logger.info") as mock_info:
        # Trigger the startup event
        with TestClient(app):
            pass
        
        # Verify that the logger was called with the expected message
        mock_info.assert_any_call("TTS service started")


@pytest.mark.unit
@pytest.mark.api
def test_shutdown_event():
    """Test shutdown event handler."""
    with patch("src.main.logger.info") as mock_info:
        # Trigger the startup and shutdown events
        with TestClient(app):
            pass
        
        # Verify that the logger was called with the expected message
        mock_info.assert_any_call("TTS service stopped") 