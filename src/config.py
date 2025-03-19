"""
Configuration module for the TTS service.
"""
import os
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict, field_validator


class TTSModelConfig(BaseModel):
    """Configuration for the TTS model."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = Field(default="tts_models/en/vctk/vits", description="TTS model name")
    device: str = Field(default="cpu", description="Device to use")
    compute_type: str = Field(default="float32", description="Compute type")
    cpu_threads: int = Field(default=4, ge=1, description="Number of CPU threads")
    num_workers: int = Field(default=1, ge=1, description="Number of workers")
    download_root: str = Field(default="/app/models", description="Root directory for model downloads")
    
    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        # This would be expanded with actual model validation
        if not v or len(v) < 5:
            raise ValueError(f"Invalid model name: {v}")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v
    
    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        valid_types = ["int8", "float16", "float32"]
        if v not in valid_types:
            raise ValueError(f"Invalid compute type: {v}. Must be one of {valid_types}")
        return v


class SynthesisOptions(BaseModel):
    """Options for speech synthesis."""
    language: str = Field(default="en", description="Language code")
    voice: str = Field(default="default", description="Voice identifier")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice pitch multiplier")
    energy: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice energy/volume")
    emotion: Optional[str] = Field(default=None, description="Emotion to convey (e.g., happy, sad, angry, neutral)")
    style: Optional[str] = Field(default=None, description="Speaking style (e.g., formal, casual, news, storytelling)")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        # This would be expanded with actual language code validation
        if len(v) < 2 or len(v) > 5:
            raise ValueError(f"Invalid language code: {v}")
        return v
    
    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        
        valid_emotions = ["happy", "sad", "angry", "neutral", "excited", "calm", "fearful", "surprised"]
        if v.lower() not in valid_emotions:
            raise ValueError(f"Invalid emotion: {v}. Must be one of {valid_emotions}")
        return v.lower()
    
    @field_validator("style")
    @classmethod
    def validate_style(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        
        valid_styles = ["formal", "casual", "news", "storytelling", "conversational", "instructional"]
        if v.lower() not in valid_styles:
            raise ValueError(f"Invalid style: {v}. Must be one of {valid_styles}")
        return v.lower()
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["wav", "mp3", "ogg"]
        if v not in valid_formats:
            raise ValueError(f"Invalid audio format: {v}. Must be one of {valid_formats}")
        return v


class SynthesisRequest(BaseModel):
    """Request model for speech synthesis."""
    text: str = Field(..., description="Text to synthesize")
    options: SynthesisOptions = Field(default_factory=SynthesisOptions, description="Synthesis options")


class BatchSynthesisRequest(BaseModel):
    """Request model for batch speech synthesis."""
    texts: List[str] = Field(..., description="List of texts to synthesize")
    options: SynthesisOptions = Field(default_factory=SynthesisOptions, description="Synthesis options")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8888, description="Port to bind to")
    log_level: str = Field(default="info", description="Log level")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    metrics_enabled: bool = Field(default=True, description="Enable metrics")
    cache_dir: str = Field(default="/app/cache", description="Cache directory")
    max_cache_size_mb: int = Field(default=1024, description="Maximum cache size in MB")


class AppConfig(BaseModel):
    """Application configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    model: TTSModelConfig = Field(default_factory=TTSModelConfig, description="Model configuration")


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    # Server config
    server_config = ServerConfig(
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8888")),
        log_level=os.getenv("SERVER_LOG_LEVEL", "info"),
        cors_origins=os.getenv("SERVER_CORS_ORIGINS", "*").split(","),
        metrics_enabled=os.getenv("SERVER_METRICS_ENABLED", "true").lower() == "true",
        cache_dir=os.getenv("SERVER_CACHE_DIR", "/app/cache"),
        max_cache_size_mb=int(os.getenv("SERVER_MAX_CACHE_SIZE_MB", "1024")),
    )
    
    # Model config
    model_config = TTSModelConfig(
        model_name=os.getenv("MODEL_NAME", "tts_models/en/vctk/vits"),
        device=os.getenv("MODEL_DEVICE", "cpu"),
        compute_type=os.getenv("MODEL_COMPUTE_TYPE", "float32"),
        cpu_threads=int(os.getenv("MODEL_CPU_THREADS", "4")),
        num_workers=int(os.getenv("MODEL_NUM_WORKERS", "1")),
        download_root=os.getenv("MODEL_DOWNLOAD_ROOT", "/app/models"),
    )
    
    return AppConfig(server=server_config, model=model_config)


# Create a global config instance
config = load_config()
