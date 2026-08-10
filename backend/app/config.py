"""Application configuration."""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API
    PROJECT_NAME: str = "DermaVision AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_USE_openssl_rand_hex_32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@db:5432/skin_classifier"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Storage
    STORAGE_TYPE: str = "local"  # local or s3
    STORAGE_ROOT: str = "./uploads"
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # Model
    MODEL_PATH: str = "./models/current_model.pth"
    MODEL_VERSION: str = "v1.0"
    DEVICE: str = "cpu"  # cpu or cuda
    BATCH_SIZE: int = 8
    IMAGE_SIZE: int = 384
    NUM_CLASSES: int = 4  # Chickenpox, Measles, Monkeypox, Normal
    
    # Upload limits
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png"}
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    
    # CORS
    CORS_ORIGINS: list = ["*"]  # Allow all origins in development
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
