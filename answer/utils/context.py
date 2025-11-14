import datetime
import logging
from typing import Optional

from sqlalchemy import and_, desc
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from answer.models.db import Conversation, User
from answer.routes.user import get_user_by_chat_id
from answer.schemas.db_models import StatusMessage
from answer.schemas.telegram import UserInfo
from answer.settings import Settings, get_settings


logger = logging.getLogger(__name__)
settings: Settings = get_settings()
engine = create_engine(str(settings.DB_DSN), pool_pre_ping=True, pool_recycle=300)
Session: DbSession = sessionmaker(bind=engine)


async def get_conversation_context(user_chat_id: str) -> str:
    """
    Получает последние CONTEXT_DEPTH диалогов пользователя и формирует контекстную строку.

    Args:
        user_chat_id: Идентификатор чата пользователя в Telegram

    Returns:
        Строка с историей диалогов в формате промта
    """
    try:
        user_result = await get_user_by_chat_id(user_chat_id)

        if isinstance(user_result, StatusMessage):
            logger.info(f"Пользователь с chat_id {user_chat_id} не найден")
            return ""

        user_info: UserInfo = user_result

        with Session() as session:
            conversations = (
                session.query(Conversation)
                .filter(and_(Conversation.user_id == user_info.id, Conversation.is_deleted == False))
                .order_by(desc(Conversation.create_ts))
                .limit(settings.CONTEXT_DEPTH)
                .all()
            )

            if not conversations:
                logger.info(f"Диалоги для пользователя {user_chat_id} не найдены")
                return ""

            conversations = list(reversed(conversations))

            context_parts = []
            for conv in conversations:
                context_parts.append(f"Пользователь: {conv.request}")
                context_parts.append(f"Ассистент: {conv.response}")

            context_string = "\n".join(context_parts)

            logger.info(f"Сформирован контекст для пользователя {user_chat_id} с {len(conversations)} диалогами")
            return context_string

    except Exception as e:
        logger.error(f"Ошибка получения контекста диалогов для пользователя {user_chat_id}: {e}", exc_info=True)
        return ""


async def build_enhanced_query(user_input: str, user_chat_id: str) -> str:
    """
    Создает расширенный запрос, объединяя текущий вопрос пользователя с контекстом предыдущих диалогов.

    Args:
        user_input: Текущий вопрос пользователя
        user_chat_id: Идентификатор чата пользователя в Telegram

    Returns:
        Расширенный запрос с контекстом
    """
    try:
        context = await get_conversation_context(user_chat_id)

        if not context:
            return user_input

        enhanced_query = f"""Контекст предыдущих диалогов:
{context}

Текущий вопрос: {user_input}"""

        logger.info(f"Сформирован расширенный запрос для пользователя {user_chat_id}")
        return enhanced_query

    except Exception as e:
        logger.error(f"Ошибка формирования расширенного запроса для пользователя {user_chat_id}: {e}", exc_info=True)
        return user_input


async def save_conversation(
    user_chat_id: str, request: str, response: str, is_response_with_buttons: bool = False
) -> bool:
    """
    Сохраняет диалог пользователя в базе данных.

    Args:
        user_chat_id: Идентификатор чата пользователя в Telegram
        request: Запрос пользователя
        response: Ответ системы
        is_response_with_buttons: Был ли ответ с кнопками

    Returns:
        True если сохранение прошло успешно, False иначе
    """
    try:
        user_result = await get_user_by_chat_id(user_chat_id)

        if isinstance(user_result, StatusMessage):
            logger.warning(f"Пользователь с chat_id {user_chat_id} не найден для сохранения диалога")
            return False

        user_info: UserInfo = user_result

        with Session() as session:
            conversation = Conversation(
                user_id=user_info.id,
                request=request,
                response=response,
                is_response_with_buttons=is_response_with_buttons,
                create_ts=datetime.datetime.now(datetime.timezone.utc),
            )

            session.add(conversation)
            session.commit()

            logger.info(f"Диалог сохранен для пользователя {user_chat_id}")
            return True

    except Exception as e:
        logger.error(f"Ошибка сохранения диалога для пользователя {user_chat_id}: {e}", exc_info=True)
        return False
