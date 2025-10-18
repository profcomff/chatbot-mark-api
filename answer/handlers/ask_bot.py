import logging

import httpx
from aiogram import Router
from aiogram.types import Message

from answer.settings import Settings, get_settings
from answer.utils.validation import (
    get_safe_user_info,
    validate_message,
    validate_question,
)


logger = logging.getLogger(__name__)
router = Router()
settings: Settings = get_settings()


async def call_internal_api(text: str, chat_id: str = "", generate_ai_response: bool = False):
    """Вызов внутреннего API через HTTP-запрос к эндпоинту /greet"""
    try:
        request_data = {"text": text, "generate_ai_response": generate_ai_response, "user_chat_id": chat_id}

        base_url = f"http://{settings.HOST}:{settings.PORT}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/greet", json=request_data, headers={"Content-Type": "application/json"}, timeout=30.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"HTTP ошибка {response.status_code}: {response.text}")
                return None

    except Exception as e:
        logger.error(f"Ошибка HTTP-запроса к внутреннему API: {e}", exc_info=True)
        return None


async def save_conversation_api(user_chat_id: str, request: str, response: str, is_response_with_buttons: bool = False):
    """Сохранение диалога через API"""
    try:
        base_url = f"http://{settings.HOST}:{settings.PORT}"
        request_data = {
            "user_chat_id": user_chat_id,
            "request": request,
            "response": response,
            "is_response_with_buttons": is_response_with_buttons,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/conversations",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )

            if response.status_code == 200:
                logger.info(f"Диалог успешно сохранен для пользователя {user_chat_id}")
                return True
            else:
                logger.error(f"Ошибка сохранения диалога: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        logger.error(f"Ошибка HTTP-запроса сохранения диалога: {e}", exc_info=True)
        return False


@router.message()
async def handle_any_message(message: Message):
    """Обработчик любого вопроса"""
    try:
        if not message.text or message.text.startswith('/'):
            return
        message_validation = validate_message(message)
        if not message_validation.is_valid:
            safe_user = get_safe_user_info(message)
            logger.warning(f"Невалидное сообщение от пользователя {safe_user['user_id']}: {message_validation.error}")
            await message.answer(
                f"❌ Ошибка в сообщении: {message_validation.error}\n\n" "Пожалуйста, попробуйте еще раз."
            )
            return

        user_question = message_validation.data['text']

        question_validation = validate_question(user_question)
        if not question_validation.is_valid:
            logger.warning(f"Невалидный вопрос от пользователя {message.from_user.id}: {question_validation.error}")
            await message.answer(f"❌ {question_validation.error}\n\n" "Пожалуйста, переформулируйте ваш вопрос.")
            return

        validated_question = question_validation.data['question']

        search_message = await message.answer("🔍 Ищу информацию и готовлю развернутый ответ...")

        api_result = await call_internal_api(
            text=validated_question, chat_id=str(message.chat.id), generate_ai_response=True
        )

        if not api_result or not api_result.get("ai_answer"):
            await search_message.delete()

            await message.answer(
                "😕 К сожалению, не удалось получить ответ на ваш вопрос.\n"
                "Попробуйте переформулировать вопрос и задать его снова."
            )
            return

        answer = api_result["ai_answer"]

        await save_conversation_api(str(message.chat.id), validated_question, answer, is_response_with_buttons=False)
        await search_message.delete()
        await message.answer(f"💡 <b>Ответ:</b>\n\n{answer}")

        logger.info(f"Отправлен развернутый ответ пользователю {message.from_user.id}")

    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}", exc_info=True)
        await message.answer("Произошла ошибка при обработке вашего вопроса. Попробуйте еще раз.")
