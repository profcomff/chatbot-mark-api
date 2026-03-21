"""Сервисный слой для бота — прямые вызовы бизнес-логики без HTTP."""

import datetime
import logging
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import and_, desc
from sqlalchemy.orm import Session as DbSession

from answer.models.db import Conversation, User
from answer.services.search_service import get_search_service
from answer.settings import get_settings
from llm.llm import get_answer
from search.filter import length_filter
from search.search import get_context

logger = logging.getLogger(__name__)
settings = get_settings()


class BotService:
    """Сервис для обработки сообщений бота и управления данными."""

    def __init__(self):
        self._search_service = get_search_service()

    def _get_app_state(self) -> Optional[Dict[str, Any]]:
        """Получает состояние приложения с инициализированными компонентами."""
        return self._search_service._app_state

    def _get_db_session(self) -> DbSession:
        """Создаёт новую сессию базы данных."""
        from sqlalchemy.engine import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(str(settings.DB_DSN), pool_pre_ping=True, pool_recycle=300)
        return sessionmaker(bind=engine)()

    def get_or_create_user(self, chat_id: str) -> Tuple[Optional[Dict], bool]:
        """
        Получает пользователя или создаёт нового.

        Args:
            chat_id: Telegram chat ID пользователя

        Returns:
            Кортеж (user_data, is_new_user) или (None, False) при ошибке
        """
        try:
            session = self._get_db_session()
            with session:
                existing_user = session.query(User).filter(User.chat_id == chat_id).first()
                if existing_user:
                    logger.info(f"Найден существующий пользователь: {chat_id}")
                    return {
                        "id": existing_user.id,
                        "chat_id": existing_user.chat_id,
                        "create_ts": existing_user.create_ts,
                        "is_deleted": existing_user.is_deleted,
                    }, False

                new_user = User(
                    chat_id=chat_id,
                    create_ts=datetime.datetime.now(datetime.timezone.utc),
                    is_deleted=False,
                )
                session.add(new_user)
                session.commit()
                session.refresh(new_user)

                logger.info(f"Создан новый пользователь: {chat_id}")
                return {
                    "id": new_user.id,
                    "chat_id": new_user.chat_id,
                    "create_ts": new_user.create_ts,
                    "is_deleted": new_user.is_deleted,
                }, True

        except Exception as e:
            logger.error(f"Ошибка получения/создания пользователя: {e}", exc_info=True)
            return None, False

    async def generate_response(
        self, text: str, chat_id: str = "", generate_ai_response: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует ответ на запрос пользователя.

        Args:
            text: Текст запроса
            chat_id: ID чата (для контекста)
            generate_ai_response: Флаг генерации AI-ответа

        Returns:
            Словарь с результатами поиска и/или AI-ответом, или None при ошибке
        """
        try:
            app_state = self._get_app_state()
            if not app_state:
                logger.error("App state не инициализирован")
                return None

            ensemble_retriever = (
                app_state["ensemble_retriever"]
                if generate_ai_response
                else app_state.get("filtered_ensemble_retriever", app_state["ensemble_retriever"])
            )

            processed_text = app_state["text_preprocessor"].preprocess(text)

            results, combined_text = get_context(
                query=processed_text,
                key_words_dict=app_state["keywords_dict"],
                ensemble_retriever=ensemble_retriever,
                vector_store=app_state["vector_store"],
                ensemble_k=settings.ensemble_k,
                verbose=True,
            )

            if results is None:
                logger.error("Ошибка генерации ответа от get_context")
                return None

            formatted_results = [
                {
                    "topic": getattr(r, "topic", ""),
                    "full_text": getattr(r, "full_text", str(r)),
                    "metadata": getattr(r, "metadata", {}),
                }
                for r in results
            ]

            response: Dict[str, Any] = {"results": formatted_results}

            if generate_ai_response:
                if length_filter(text=text, max_len=settings.max_length):
                    ai_answer = get_answer(
                        context=combined_text,
                        question=text,
                        settings=settings,
                    )
                    if ai_answer:
                        response["ai_answer"] = ai_answer
                else:
                    response["ai_answer"] = (
                        "Ваш запрос слишком длинный :( Сделайте короче или используйте режим без GPT."
                    )
            elif len(formatted_results) == 0:
                response["ai_answer"] = "Извините, я не понял Ваш запрос. Попробуйте использовать GPT версию."

            return response

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            return None

    def save_conversation(
        self, user_chat_id: str, request: str, response: str, is_response_with_buttons: bool = False
    ) -> bool:
        """
        Сохраняет диалог в базу данных.

        Args:
            user_chat_id: ID чата пользователя
            request: Текст запроса
            response: Текст ответа
            is_response_with_buttons: Флаг ответа с кнопками

        Returns:
            True при успехе, False при ошибке
        """
        try:
            session = self._get_db_session()
            with session:
                user = session.query(User).filter(User.chat_id == user_chat_id).one_or_none()
                if not user:
                    logger.error(f"Пользователь не найден: {user_chat_id}")
                    return False

                conversation = Conversation(
                    user_id=user.id,
                    request=request,
                    response=response,
                    is_response_with_buttons=is_response_with_buttons,
                    create_ts=datetime.datetime.now(datetime.timezone.utc),
                    is_deleted=False,
                )

                session.add(conversation)
                session.commit()

                logger.info(f"Диалог сохранен для пользователя {user_chat_id}")
                return True

        except Exception as e:
            logger.error(f"Ошибка сохранения диалога: {e}", exc_info=True)
            return False

    def get_user(self, chat_id: str) -> Optional[Dict]:
        """
        Получает пользователя по chat_id.

        Args:
            chat_id: Telegram chat ID

        Returns:
            Данные пользователя или None если не найден
        """
        try:
            session = self._get_db_session()
            with session:
                user = session.query(User).filter(User.chat_id == chat_id).one_or_none()
                if user is None:
                    return None

                return {
                    "id": user.id,
                    "chat_id": user.chat_id,
                    "create_ts": user.create_ts,
                    "is_deleted": user.is_deleted,
                }

        except Exception as e:
            logger.error(f"Ошибка получения пользователя: {e}", exc_info=True)
            return None

    def get_conversation_context(self, chat_id: str) -> str:
        """
        Получает контекст последних диалогов пользователя.

        Args:
            chat_id: Telegram chat ID

        Returns:
            Строка с контекстом диалогов
        """
        try:
            session = self._get_db_session()
            with session:
                user = session.query(User).filter(User.chat_id == chat_id).one_or_none()
                if user is None:
                    return ""

                conversations = (
                    session.query(Conversation)
                    .filter(and_(Conversation.user_id == user.id, Conversation.is_deleted == False))
                    .order_by(desc(Conversation.create_ts))
                    .limit(settings.CONTEXT_DEPTH)
                    .all()
                )

                if not conversations:
                    return ""

                conversations = list(reversed(conversations))
                context_parts = []
                for conv in conversations:
                    context_parts.append(f"Пользователь: {conv.request}")
                    context_parts.append(f"Ассистент: {conv.response}")

                return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Ошибка получения контекста диалогов: {e}", exc_info=True)
            return ""


_bot_service = BotService()


def get_bot_service() -> BotService:
    """Возвращает экземпляр BotService."""
    return _bot_service
