"""Утилиты для валидации входящих данных от пользователей Telegram"""

import logging
from typing import Optional

from aiogram.types import CallbackQuery, Message
from pydantic import ValidationError

from answer.schemas.telegram import CallbackDataInput, QuestionValidation, TelegramUserInput


logger = logging.getLogger(__name__)


class ValidationResult:
    """Результат валидации"""

    def __init__(self, is_valid: bool, data: Optional[dict] = None, error: Optional[str] = None):
        self.is_valid = is_valid
        self.data = data
        self.error = error


def validate_message(message: Message) -> ValidationResult:
    """
    Валидация входящего сообщения от пользователя

    Args:
        message: Объект сообщения Telegram

    Returns:
        ValidationResult: Результат валидации с данными или ошибкой
    """
    try:
        if not message.text:
            return ValidationResult(is_valid=False, error="Сообщение должно содержать текст")

        user_data = {
            'text': message.text.strip(),
            'user_id': message.from_user.id,
            'chat_id': str(message.chat.id),
            'username': message.from_user.username,
            'first_name': message.from_user.first_name,
            'last_name': message.from_user.last_name,
        }

        validated_data = TelegramUserInput.model_validate(user_data)

        logger.info(f"Валидация сообщения успешна для пользователя {message.from_user.id}")

        return ValidationResult(is_valid=True, data=validated_data.model_dump())

    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error.get('loc', ['unknown'])[0]
            msg = error.get('msg', 'Unknown error')
            error_messages.append(f"{field}: {msg}")

        error_text = "; ".join(error_messages)
        logger.warning(f"Ошибка валидации сообщения от пользователя {message.from_user.id}: {error_text}")

        return ValidationResult(is_valid=False, error=f"Ошибка валидации данных: {error_text}")

    except Exception as e:
        logger.error(f"Неожиданная ошибка валидации сообщения: {e}", exc_info=True)
        return ValidationResult(is_valid=False, error="Произошла ошибка при обработке сообщения")


def validate_callback_query(callback: CallbackQuery) -> ValidationResult:
    """
    Валидация входящего callback query от пользователя

    Args:
        callback: Объект callback query Telegram

    Returns:
        ValidationResult: Результат валидации с данными или ошибкой
    """
    try:
        if not callback.data:
            return ValidationResult(is_valid=False, error="Callback query должен содержать данные")

        callback_data = {
            'callback_data': callback.data.strip(),
            'user_id': callback.from_user.id,
            'chat_id': str(callback.message.chat.id),
        }

        validated_data = CallbackDataInput.model_validate(callback_data)

        logger.info(f"Валидация callback успешна для пользователя {callback.from_user.id}")

        return ValidationResult(is_valid=True, data=validated_data.model_dump())

    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            field = error.get('loc', ['unknown'])[0]
            msg = error.get('msg', 'Unknown error')
            error_messages.append(f"{field}: {msg}")

        error_text = "; ".join(error_messages)
        logger.warning(f"Ошибка валидации callback от пользователя {callback.from_user.id}: {error_text}")

        return ValidationResult(is_valid=False, error=f"Ошибка валидации callback данных: {error_text}")

    except Exception as e:
        logger.error(f"Неожиданная ошибка валидации callback: {e}", exc_info=True)
        return ValidationResult(is_valid=False, error="Произошла ошибка при обработке callback")


def validate_question(question_text: str) -> ValidationResult:
    """
    Валидация текста вопроса пользователя

    Args:
        question_text: Текст вопроса

    Returns:
        ValidationResult: Результат валидации с данными или ошибкой
    """
    try:
        validated_question = QuestionValidation.model_validate({'question': question_text.strip()})

        logger.info("Валидация вопроса успешна")

        return ValidationResult(is_valid=True, data={'question': validated_question.question})

    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            msg = error.get('msg', 'Unknown error')
            error_messages.append(msg)

        error_text = "; ".join(error_messages)
        logger.warning(f"Ошибка валидации вопроса: {error_text}")

        return ValidationResult(is_valid=False, error=error_text)

    except Exception as e:
        logger.error(f"Неожиданная ошибка валидации вопроса: {e}", exc_info=True)
        return ValidationResult(is_valid=False, error="Произошла ошибка при валидации вопроса")


def get_safe_user_info(message: Message) -> dict:
    """Получение безопасной информации о пользователе для логирования"""
    return {
        'user_id': message.from_user.id,
        'chat_id': str(message.chat.id),
        'username': message.from_user.username[:20] if message.from_user.username else None,
        'first_name': message.from_user.first_name[:20] if message.from_user.first_name else None,
        'has_text': bool(message.text),
        'text_length': len(message.text) if message.text else 0,
    }


def sanitize_text_for_logging(text: str, max_length: int = 100) -> str:
    """Безопасная обрезка текста для логирования"""
    if not text:
        return ""

    safe_text = text.replace('\n', ' ').replace('\r', ' ').strip()

    if len(safe_text) > max_length:
        safe_text = safe_text[: max_length - 3] + "..."

    return safe_text
