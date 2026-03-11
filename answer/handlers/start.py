import logging

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from answer.handlers.keyboards import get_base_menu
from answer.services.bot_service import get_bot_service
from answer.settings import Settings, get_settings
from answer.utils.validation import get_safe_user_info, validate_callback_query, validate_message

logger = logging.getLogger(__name__)
start_router = Router()
settings: Settings = get_settings()
bot_service = get_bot_service()


@start_router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        validation_result = validate_message(message)
        if not validation_result.is_valid:
            safe_user = get_safe_user_info(message)
            logger.warning(
                f"Невалидное /start сообщение от пользователя {safe_user['user_id']}: {validation_result.error}"
            )
            await message.answer("❌ Ошибка при обработке команды. Попробуйте еще раз.")
            return

        logger.info(f"Received /start command from user {message.from_user.id}")
        chat_id = str(message.chat.id)
        user_data, is_new_user = bot_service.get_or_create_user(chat_id)

        if user_data:
            if is_new_user:
                message_text = (
                    f"Привет, <b>{message.from_user.full_name}</b>! Меня зовут Марк." + settings.BASE_DESCRIPTION
                )
                logger.info(f"Created new user with chat_id: {chat_id}")
            else:
                message_text = (
                    f"Привет, Марк помнит тебя, <b>{message.from_user.full_name}</b>!" + settings.BASE_DESCRIPTION
                )
                logger.info(f"User with chat_id {chat_id} already exists")
        else:
            message_text = f"Привет, <b>{message.from_user.full_name}</b>! Меня зовут Марк." + settings.BASE_DESCRIPTION
            logger.warning(f"Failed to create/get user via API for chat_id: {chat_id}")

        menu_builder = await get_base_menu()
        await message.answer(text=message_text, reply_markup=menu_builder.as_markup())
        logger.info("Response sent successfully")

    except Exception as e:
        logger.error(f"Error in start handler: {e}", exc_info=True)
        await message.answer("Произошла ошибка. Попробуйте позже.\n Если проблема сохранится, обратитесь в поддержку")


@start_router.callback_query(F.data == "back_to_menu")
async def back_to_menu(callback: CallbackQuery):
    """Возврат в главное меню"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        menu_builder = await get_base_menu()
        message_text = f"🏠 <b>Главное меню</b>" + settings.BASE_DESCRIPTION
        await callback.message.answer(text=message_text, reply_markup=menu_builder.as_markup())
        await callback.answer()
        logger.info("User returned to main menu")
    except Exception as e:
        logger.error(f"Error in back_to_menu handler: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при возврате в меню")
