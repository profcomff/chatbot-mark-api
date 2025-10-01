"""Общие модели для API взаимодействия между слоями приложения."""

import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UserInput(BaseModel):
    """Модель входных данных от пользователя."""

    text: str
    generate_ai_response: bool = False
    user_chat_id: str = ""


class SearchResult(BaseModel):
    """Модель результата поиска."""

    topic: str
    full_text: str
    metadata: Optional[Dict[str, Any]] = None


class ResponseResult(BaseModel):
    """Модель ответа системы."""

    results: List[SearchResult]
    ai_answer: Optional[str] = None
    message: Optional[str] = None


# Новые модели для API эндпоинтов
class CreateUserRequest(BaseModel):
    """Модель запроса на создание пользователя."""

    chat_id: str


class UserResponse(BaseModel):
    """Модель ответа с информацией о пользователе."""

    id: int
    chat_id: str
    create_ts: datetime.datetime
    is_deleted: bool

    class Config:
        # Настройка для правильной сериализации datetime с timezone
        json_encoders = {datetime.datetime: lambda v: v.isoformat() if v else None}


class SaveConversationRequest(BaseModel):
    """Модель запроса на сохранение диалога."""

    user_chat_id: str
    request: str
    response: str
    is_response_with_buttons: bool = False


class ConversationContextResponse(BaseModel):
    """Модель ответа с контекстом диалогов."""

    context: str
    conversations_count: int
