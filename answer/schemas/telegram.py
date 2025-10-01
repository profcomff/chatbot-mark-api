import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserPost(BaseModel):
    chat_id: str
    create_ts: datetime.datetime


class UserInfo(BaseModel):
    id: int
    chat_id: str
    create_ts: datetime.datetime
    is_deleted: bool


class TelegramUserInput(BaseModel):
    """Валидация входящих данных от пользователя Telegram"""

    text: str
    user_id: int
    chat_id: str
    username: str | None
    first_name: str | None
    last_name: str | None


class CallbackDataInput(BaseModel):
    """Валидация данных callback кнопок"""

    callback_data: str
    user_id: int
    chat_id: str


class QuestionValidation(BaseModel):
    """Валидация вопроса пользователя"""

    question: str
