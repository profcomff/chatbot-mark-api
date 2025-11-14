import logging

from aiogram import F, Router
from aiogram.types import CallbackQuery

from answer.handlers.keyboards import get_menu_from_help, get_menu_from_info
from answer.utils.validation import validate_callback_query


logger = logging.getLogger(__name__)

router: Router = Router()


@router.callback_query(F.data == "help")
async def ask_for_help(callback: CallbackQuery):
    """Обработчик запроса помощи от пользователя"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        help_text = """Раздел в разработке"""

        # 🆘 <b>Поддержка</b>

        # Если у вас возникли вопросы или проблемы, вы можете:

        # 📧 Написать на почту: pochtazatichka@profcomff.com
        # 💬 Обратиться в техническую поддержку через официальные каналы
        # 🔧 Описать проблему максимально подробно для быстрого решения

        # Мы постараемся ответить в кратчайшие сроки!

        base_menu = await get_menu_from_help()
        await callback.message.answer(text=help_text, reply_markup=base_menu.as_markup())
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в ask_for_help: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при получении справки")


@router.callback_query(F.data == "info")
async def get_faq(callback: CallbackQuery):
    """Обработчик FAQ"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        faq_text = """
❓ <b>Часто задаваемые вопросы (FAQ)</b>

<b>Что умеет Марк?</b>
Марк может отвечать на вопросы, связанные с деятельностью Профкома и студенческой жизнью.

<b>Как задать вопрос?</b>
Напишите свой вопрос.

<b>Марк не понимает мой вопрос. Что делать?</b>
Попробуйте переформулировать вопрос или обратитесь в поддержку.

<b>Конфиденциальность</b>
Марк не сохраняет персональные данные пользователей без необходимости.

<b>Техническая поддержка</b>
Если у вас технические проблемы, воспользуйтесь кнопкой "Поддержка".
        """
        base_menu = await get_menu_from_info()
        await callback.message.answer(text=faq_text, reply_markup=base_menu.as_markup())
        await callback.answer()
    except Exception as e:
        logger.error(f"Ошибка в get_faq: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при получении FAQ")
