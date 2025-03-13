"""
Metrics module for Prometheus metrics.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
SYNTHESIS_REQUESTS = Counter(
    "tts_synthesis_requests_total",
    "Total number of synthesis requests",
    ["status"]
)

SYNTHESIS_LATENCY = Histogram(
    "tts_synthesis_latency_seconds",
    "Latency of synthesis requests",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

SYNTHESIS_TEXT_LENGTH = Histogram(
    "tts_synthesis_text_length",
    "Length of text in synthesis requests",
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)

SYNTHESIS_AUDIO_SIZE = Histogram(
    "tts_synthesis_audio_size_bytes",
    "Size of audio in synthesis responses",
    buckets=[1024, 10240, 102400, 1048576, 10485760]
)

# Model metrics
MODEL_LOAD_LATENCY = Histogram(
    "tts_model_load_latency_seconds",
    "Latency of model loading",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

MODEL_CACHE_SIZE = Gauge(
    "tts_model_cache_size",
    "Number of models in cache"
)

MODEL_MEMORY_USAGE = Gauge(
    "tts_model_memory_usage_bytes",
    "Memory usage of models",
    ["model_name"]
)

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    "tts_system_memory_usage_bytes",
    "System memory usage"
)

SYSTEM_CPU_USAGE = Gauge(
    "tts_system_cpu_usage_percent",
    "System CPU usage"
)

# Initialize request status counters
SYNTHESIS_REQUESTS.labels(status="success")
SYNTHESIS_REQUESTS.labels(status="error") 