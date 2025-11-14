import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from answer.models.base import BaseDbModel


class User(BaseDbModel):
    """
    Таблица данных юзера
    """

    id: Mapped[int] = mapped_column(primary_key=True, comment="Идентификатор пользователя")
    chat_id: Mapped[str] = mapped_column(unique=True, comment="Тг айди чата с пользователем")
    create_ts: Mapped[datetime.datetime] = mapped_column(DateTime, comment="Таймстемп создания пользователя")
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False, comment="Флаг софтделита"
    )


class Conversation(BaseDbModel):
    """
    Таблица контекста диалога для каждого юзера
    """

    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="Идентификатор записи диалога")
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("user.id"))
    request: Mapped[str] = mapped_column(
        String, nullable=False, default="request_text", server_default='request_text', comment="Строка запроса"
    )
    response: Mapped[str] = mapped_column(
        String, nullable=False, default="response_text", server_default='response_text', comment="Строка ответа"
    )
    create_ts: Mapped[datetime.datetime] = mapped_column(DateTime, comment="Таймстемп создания пары request/response")
    is_response_with_buttons: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        comment="Генерировался ли в режиме возврата эндпоинтов (False - значит - чисто генерированный ai ответ)",
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", default=False, comment="Флаг софтделита"
    )
