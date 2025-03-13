"""
Tests for metrics module.
"""
import pytest
from unittest.mock import patch, MagicMock

from prometheus_client import Counter, Histogram, Gauge, Summary

from src.utils import metrics


@pytest.mark.unit
@pytest.mark.utils
def test_metrics_initialization():
    """Test metrics initialization."""
    # Verify that all metrics are initialized
    assert isinstance(metrics.SYNTHESIS_REQUESTS, Counter)
    assert isinstance(metrics.SYNTHESIS_LATENCY, Histogram)
    assert isinstance(metrics.SYNTHESIS_TEXT_LENGTH, Histogram)
    assert isinstance(metrics.SYNTHESIS_AUDIO_SIZE, Histogram)
    assert isinstance(metrics.MODEL_LOAD_LATENCY, Histogram)
    assert isinstance(metrics.MODEL_CACHE_SIZE, Gauge)
    assert isinstance(metrics.MODEL_MEMORY_USAGE, Gauge)
    assert isinstance(metrics.SYSTEM_MEMORY_USAGE, Gauge)
    assert isinstance(metrics.SYSTEM_CPU_USAGE, Gauge)


@pytest.mark.unit
@pytest.mark.utils
def test_synthesis_requests_counter():
    """Test SYNTHESIS_REQUESTS counter."""
    with patch.object(metrics.SYNTHESIS_REQUESTS, 'labels', return_value=MagicMock()) as mock_labels:
        # Verify that the counter is initialized with success and error labels
        metrics.SYNTHESIS_REQUESTS.labels(status="success")
        metrics.SYNTHESIS_REQUESTS.labels(status="error")
        
        # Verify that the labels method was called with the correct arguments
        mock_labels.assert_any_call(status="success")
        mock_labels.assert_any_call(status="error")


@pytest.mark.unit
@pytest.mark.utils
def test_synthesis_latency_histogram():
    """Test SYNTHESIS_LATENCY histogram."""
    with patch.object(metrics.SYNTHESIS_LATENCY, 'observe') as mock_observe:
        # Observe a latency value
        metrics.SYNTHESIS_LATENCY.observe(1.5)
        
        # Verify that the observe method was called with the correct value
        mock_observe.assert_called_once_with(1.5)


@pytest.mark.unit
@pytest.mark.utils
def test_synthesis_text_length_histogram():
    """Test SYNTHESIS_TEXT_LENGTH histogram."""
    with patch.object(metrics.SYNTHESIS_TEXT_LENGTH, 'observe') as mock_observe:
        # Observe a text length value
        metrics.SYNTHESIS_TEXT_LENGTH.observe(100)
        
        # Verify that the observe method was called with the correct value
        mock_observe.assert_called_once_with(100)


@pytest.mark.unit
@pytest.mark.utils
def test_synthesis_audio_size_histogram():
    """Test SYNTHESIS_AUDIO_SIZE histogram."""
    with patch.object(metrics.SYNTHESIS_AUDIO_SIZE, 'observe') as mock_observe:
        # Observe an audio size value
        metrics.SYNTHESIS_AUDIO_SIZE.observe(50000)
        
        # Verify that the observe method was called with the correct value
        mock_observe.assert_called_once_with(50000)


@pytest.mark.unit
@pytest.mark.utils
def test_model_load_latency_histogram():
    """Test MODEL_LOAD_LATENCY histogram."""
    with patch.object(metrics.MODEL_LOAD_LATENCY, 'observe') as mock_observe:
        # Observe a model load latency value
        metrics.MODEL_LOAD_LATENCY.observe(2.5)
        
        # Verify that the observe method was called with the correct value
        mock_observe.assert_called_once_with(2.5)


@pytest.mark.unit
@pytest.mark.utils
def test_model_cache_size_gauge():
    """Test MODEL_CACHE_SIZE gauge."""
    with patch.object(metrics.MODEL_CACHE_SIZE, 'set') as mock_set:
        # Set a model cache size value
        metrics.MODEL_CACHE_SIZE.set(3)
        
        # Verify that the set method was called with the correct value
        mock_set.assert_called_once_with(3)


@pytest.mark.unit
@pytest.mark.utils
def test_model_memory_usage_gauge():
    """Test MODEL_MEMORY_USAGE gauge."""
    with patch.object(metrics.MODEL_MEMORY_USAGE, 'labels', return_value=MagicMock()) as mock_labels:
        # Set a model memory usage value
        mock_gauge = mock_labels.return_value
        metrics.MODEL_MEMORY_USAGE.labels(model_name="test_model").set(1024 * 1024 * 100)  # 100 MB
        
        # Verify that the labels method was called with the correct arguments
        mock_labels.assert_called_once_with(model_name="test_model")
        mock_gauge.set.assert_called_once_with(1024 * 1024 * 100)


@pytest.mark.unit
@pytest.mark.utils
def test_system_memory_usage_gauge():
    """Test SYSTEM_MEMORY_USAGE gauge."""
    with patch.object(metrics.SYSTEM_MEMORY_USAGE, 'set') as mock_set:
        # Set a system memory usage value
        metrics.SYSTEM_MEMORY_USAGE.set(1024 * 1024 * 500)  # 500 MB
        
        # Verify that the set method was called with the correct value
        mock_set.assert_called_once_with(1024 * 1024 * 500)


@pytest.mark.unit
@pytest.mark.utils
def test_system_cpu_usage_gauge():
    """Test SYSTEM_CPU_USAGE gauge."""
    with patch.object(metrics.SYSTEM_CPU_USAGE, 'set') as mock_set:
        # Set a system CPU usage value
        metrics.SYSTEM_CPU_USAGE.set(75.5)  # 75.5%
        
        # Verify that the set method was called with the correct value
        mock_set.assert_called_once_with(75.5) 