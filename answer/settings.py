import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, PostgresDsn, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    DB_DSN: PostgresDsn = "postgresql://postgres@localhost:5432/postgres"
    ROOT_PATH: str = "/" + os.getenv("APP_NAME", "")
    
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    CHROMA_DIR: str
    GIGA_KEY_PATH: str
    
    GIGA_MAX_TOKENS: int = 500
    PROFANITY_CHECK: bool = True
    
    ensemble_k: int = 5
    retrivier_k: int = 10                     

@lru_cache
def get_settings() -> Settings:
    return Settings()
