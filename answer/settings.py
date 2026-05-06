import os
from functools import lru_cache

from pydantic import ConfigDict, Field, PostgresDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DB_DSN: PostgresDsn = "postgresql://postgres@localhost:5432/postgres"
    BOT_TOKEN: str = ""
    BASE_URL: str = ""
    BASE_DESCRIPTION: str = (
        "\nЯ - бот помощник для студентов физического факультета МГУ.\n"
        "Задай любой вопрос по стипендиям, учебным правам, социальным программам "
        "и иным особенностям обучения - я постараюсь тебе помочь."
    )
    CONTEXT_DEPTH: int = 3
    WEBHOOK_PATH: str = ""
    ROOT_PATH: str = "/" + os.getenv("APP_NAME", "")

    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    QDRANT_API_KEY: str
    QDRANT_HOST: str
    QDRANT_PORT: int
    QDRANT_HTTPS: bool
    QDRANT_TIMEOUT: int
    collection_name: str = Field(validation_alias="COLLECTION_NAME")

    SERVICE_ACCOUNT_ID: str = "null"
    PRIVATE_KEY: str = "null"
    KEY_ID: str = "null"
    LLM_MAX_OUTPUT: int = 500

    HOST: str = "127.0.0.1"
    PORT: int = 8080

    ensemble_k: int = 10
    retrivier_k: int = 20
    MAX_BUTTONS: int = 5
    max_length: int = 200
    warning_message: str = "<i>Ответ сгенерирован ИИ и может содержать неточности.</i>"


@lru_cache
def get_settings() -> Settings:
    return Settings()