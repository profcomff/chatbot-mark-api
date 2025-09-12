import asyncio
import logging

from aiogram import Router
from aiogram.utils.keyboard import InlineKeyboardBuilder

from answer.routes.user import get_user_by_chat_id
from answer.settings import Settings, get_settings


settings: Settings = get_settings()
router: Router = Router()
logger = logging.getLogger(__name__)


async def get_base_menu() -> InlineKeyboardBuilder:
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="Спросить Марка", callback_data="ask_llm")
    builder.button(text="FAQ", callback_data="info")
    builder.button(text="Поддержка", callback_data="help")
    builder.adjust(1, 2)
    return builder


async def get_menu_from_info() -> InlineKeyboardBuilder:
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="Спросить Марка", callback_data="ask_llm")
    builder.button(text="Поддержка", callback_data="help")
    builder.button(text="В главное меню", callback_data="back_to_menu")
    builder.adjust(1, 2)
    return builder


async def get_menu_from_help() -> InlineKeyboardBuilder:
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="Спросить Марка", callback_data="ask_llm")
    builder.button(text="FAQ", callback_data="info")
    builder.button(text="В главное меню", callback_data="back_to_menu")
    builder.adjust(1, 2)
    return builder


async def get_ask_bot_keyboard() -> InlineKeyboardBuilder:
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="Спросить Марка", callback_data="ask_llm")
    builder.button(text="FAQ", callback_data="info")
    builder.button(text="В главное меню", callback_data="back_to_menu")
    builder.adjust(1, 2)
    return builder


async def get_response_type_keyboard() -> InlineKeyboardBuilder:
    """Клавиатура выбора типа ответа"""
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="📄 Развернутый ответ", callback_data="response_detailed")
    builder.button(text="🔗 Релевантные кнопки", callback_data="response_buttons")
    builder.button(text="🔙 Назад", callback_data="back_to_menu")
    builder.adjust(2, 1)
    return builder


async def get_topics_keyboard(topics_list: list, page: int = 0, total_results: int = None) -> InlineKeyboardBuilder:
    """Клавиатура с релевантными топиками с поддержкой пагинации"""
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()

    max_buttons = settings.MAX_BUTTONS
    start_idx = page * max_buttons
    end_idx = start_idx + max_buttons
    page_topics = topics_list[start_idx:end_idx]

    for i, result in enumerate(page_topics):
        topic_name = result.get("topic", f"Топик {start_idx + i + 1}")
        if len(topic_name) > 50:
            topic_name = topic_name[:47] + "..."
        builder.button(text=topic_name, callback_data=f"topic_{start_idx + i}")

    nav_buttons = []

    if page > 0:
        nav_buttons.append(("⬅️", f"page_{page - 1}"))

    total_pages = (len(topics_list) + max_buttons - 1) // max_buttons
    if total_pages > 1:
        start_idx = page * max_buttons
        end_idx = min(start_idx + max_buttons, len(topics_list))
        page_info = f"📄 {page + 1}/{total_pages}"
        nav_buttons.append((page_info, "page_info"))

    if end_idx < len(topics_list):
        nav_buttons.append(("➡️", f"page_{page + 1}"))

    for text, callback_data in nav_buttons:
        builder.button(text=text, callback_data=callback_data)

    builder.button(text="🔍 Новый поиск", callback_data="ask_llm")
    builder.button(text="📋 Главное меню", callback_data="back_to_menu")

    if nav_buttons:
        builder.adjust(1, len(nav_buttons), 2)
    else:
        builder.adjust(1, 2)

    return builder


async def get_no_results_keyboard() -> InlineKeyboardBuilder:
    """Клавиатура для случаев, когда результаты не найдены"""
    builder: InlineKeyboardBuilder = InlineKeyboardBuilder()
    builder.button(text="🔍 Попробовать еще раз", callback_data="ask_llm")
    builder.button(text="📋 Главное меню", callback_data="back_to_menu")
    builder.adjust(2)
    return builder
