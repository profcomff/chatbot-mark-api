import asyncio
import json
import logging

import httpx
from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from answer.handlers.keyboards import (
    get_base_menu,
    get_no_results_keyboard,
    get_response_type_keyboard,
    get_topics_keyboard,
)
from answer.handlers.states import QuestionState, TopicState
from answer.settings import Settings, get_settings
from answer.utils.validation import (
    get_safe_user_info,
    sanitize_text_for_logging,
    validate_callback_query,
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


@router.callback_query(F.data == "ask_llm")
async def ask_bot_mark(callback: CallbackQuery, state: FSMContext):
    """Обработчик нажатия кнопки 'Спросить Марка'"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        await state.set_state(QuestionState.waiting_for_question)
        await callback.message.answer(
            "❓ <b>Задайте ваш вопрос</b>\n\n" "Напишите, что вас интересует, и я помогу найти ответ!"
        )
        await callback.answer()
        logger.info(f"Пользователь {callback.from_user.id} перешел в режим ввода вопроса")
    except Exception as e:
        logger.error(f"Ошибка в ask_bot_mark: {e}", exc_info=True)
        await callback.answer("Произошла ошибка. Попробуйте позже.")


@router.message(StateFilter(QuestionState.waiting_for_question))
async def process_question(message: Message, state: FSMContext):
    """Обработчик получения вопроса от пользователя"""
    try:
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

        await state.update_data(
            user_question=validated_question, chat_id=str(message.chat.id), user_id=message.from_user.id
        )

        keyboard = await get_response_type_keyboard()
        await message.answer(
            f"📝 <b>Ваш вопрос:</b> {validated_question}\n\n" "🔽 <b>Выберите тип ответа:</b>",
            reply_markup=keyboard.as_markup(),
        )

        await state.set_state(QuestionState.waiting_for_response_type)
        safe_question = sanitize_text_for_logging(validated_question, 50)
        logger.info(f"Получен валидный вопрос от пользователя {message.from_user.id}: {safe_question}")

    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}", exc_info=True)
        await message.answer("Произошла ошибка при обработке вопроса. Попробуйте еще раз.")
        await state.clear()


@router.callback_query(F.data == "response_detailed", StateFilter(QuestionState.waiting_for_response_type))
async def handle_detailed_response(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора развернутого ответа"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            await state.clear()
            return

        data = await state.get_data()
        user_question = data.get("user_question")
        chat_id = data.get("chat_id")

        if not user_question or not chat_id:
            logger.warning(f"Отсутствуют данные в состоянии для пользователя {callback.from_user.id}")
            await callback.answer("❌ Ошибка: данные вопроса не найдены")
            await state.clear()
            return

        question_validation = validate_question(user_question)
        if not question_validation.is_valid:
            logger.warning(
                f"Невалидный сохраненный вопрос от пользователя {callback.from_user.id}: {question_validation.error}"
            )
            await callback.answer("❌ Ошибка: невалидные данные вопроса")
            await state.clear()
            return

        await callback.message.edit_text("🔍 Ищу информацию и готовлю развернутый ответ...")

        api_result = await call_internal_api(text=user_question, chat_id=chat_id, generate_ai_response=True)

        if not api_result or not api_result.get("ai_answer"):
            no_results_keyboard = await get_no_results_keyboard()
            await callback.message.edit_text(
                "😕 К сожалению, не удалось получить ответ на ваш вопрос.\n"
                "Попробуйте переформулировать вопрос или выберите действие:",
                reply_markup=no_results_keyboard.as_markup(),
            )
            await state.clear()
            return

        answer = api_result["ai_answer"]

        await save_conversation_api(chat_id, user_question, answer, is_response_with_buttons=False)

        await callback.message.edit_text(f"💡 <b>Ответ:</b>\n\n{answer}")

        menu_keyboard = await get_base_menu()
        await callback.message.answer("❓ Есть еще вопросы?", reply_markup=menu_keyboard.as_markup())

        await state.clear()
        logger.info(f"Отправлен развернутый ответ пользователю {callback.from_user.id}")

    except Exception as e:
        logger.error(f"Ошибка получения развернутого ответа: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при получении ответа.")
        await state.clear()


@router.callback_query(F.data == "response_buttons", StateFilter(QuestionState.waiting_for_response_type))
async def handle_buttons_response(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора ответа с кнопками"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            await state.clear()
            return

        data = await state.get_data()
        user_question = data.get("user_question")
        chat_id = data.get("chat_id")

        await callback.message.edit_text("🔍 Ищу релевантные топики...")

        api_result = await call_internal_api(text=user_question, chat_id="", generate_ai_response=False)

        if not api_result or not api_result.get("results"):
            no_results_keyboard = await get_no_results_keyboard()
            await callback.message.edit_text(
                "😕 К сожалению, не удалось найти релевантные топики по вашему вопросу.\n"
                "Попробуйте переформулировать вопрос или выберите действие:",
                reply_markup=no_results_keyboard.as_markup(),
            )
            await state.clear()
            return

        results = api_result["results"]
        logger.info(f"Поиск релевантных топиков без контекста для вопроса: {user_question[:50]}...")

        if not results:
            no_results_keyboard = await get_no_results_keyboard()
            await callback.message.edit_text(
                "😕 К сожалению, не удалось найти релевантные топики по вашему вопросу.\n"
                "Попробуйте переформулировать вопрос или выберите действие:",
                reply_markup=no_results_keyboard.as_markup(),
            )
            await state.clear()
            return

        max_buttons = settings.MAX_BUTTONS
        total_found = len(results)
        current_page = 0

        logger.info(
            f"Найдено {total_found} результатов, показываем страницу {current_page + 1} (макс на странице: {max_buttons})"
        )

        await state.update_data(search_results=results, current_page=current_page, total_results=total_found)
        await state.set_state(TopicState.showing_topics)

        topics_keyboard = await get_topics_keyboard(results, page=current_page, total_results=total_found)

        start_idx = current_page * max_buttons
        end_idx = min(start_idx + max_buttons, total_found)
        shown_count = end_idx - start_idx

        message_text = f"🎯 <b>Найдены релевантные топики по вашему вопросу:</b>\n<i>{user_question}</i>\n\n"

        if total_found > max_buttons:
            total_pages = (total_found + max_buttons - 1) // max_buttons
            message_text += f"Показаны результаты {start_idx + 1}-{end_idx} из {total_found} (страница {current_page + 1} из {total_pages}).\n\n"

        message_text += "Выберите интересующий вас раздел:"

        await callback.message.edit_text(message_text, reply_markup=topics_keyboard.as_markup())

        page_topics = results[start_idx:end_idx]
        response_text = f"Найдены топики (страница {current_page + 1}, показано {shown_count} из {total_found}): {', '.join([r['topic'] for r in page_topics])}"
        await save_conversation_api(chat_id, user_question, response_text, is_response_with_buttons=True)

        logger.info(f"Показаны релевантные топики пользователю {callback.from_user.id}")

    except Exception as e:
        logger.error(f"Ошибка получения релевантных топиков: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при поиске топиков.")
        await state.clear()


@router.callback_query(F.data.startswith("page_"), StateFilter(TopicState.showing_topics))
async def handle_page_navigation(callback: CallbackQuery, state: FSMContext):
    """Обработчик навигации по страницам с топиками"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        if callback.data == "page_info":
            await callback.answer()
            return

        new_page = int(callback.data.split("_")[1])

        data = await state.get_data()
        search_results = data.get("search_results", [])
        user_question = data.get("user_question", "")
        total_results = data.get("total_results", len(search_results))

        max_buttons = settings.MAX_BUTTONS
        total_pages = (len(search_results) + max_buttons - 1) // max_buttons

        if new_page < 0 or new_page >= total_pages:
            await callback.answer("Неверный номер страницы")
            return

        await state.update_data(current_page=new_page)

        topics_keyboard = await get_topics_keyboard(search_results, page=new_page, total_results=total_results)

        start_idx = new_page * max_buttons
        end_idx = min(start_idx + max_buttons, len(search_results))

        message_text = f"🎯 <b>Найдены релевантные топики по вашему вопросу:</b>\n<i>{user_question}</i>\n\n"

        if len(search_results) > max_buttons:
            message_text += f"Показаны результаты {start_idx + 1}-{end_idx} из {len(search_results)} (страница {new_page + 1} из {total_pages}).\n\n"

        message_text += "Выберите интересующий вас раздел:"

        await callback.message.edit_text(message_text, reply_markup=topics_keyboard.as_markup())

        await callback.answer()
        logger.info(f"Пользователь {callback.from_user.id} переключился на страницу {new_page + 1}")

    except Exception as e:
        logger.error(f"Ошибка навигации по страницам: {e}", exc_info=True)
        await callback.answer("Произошла ошибка при переключении страницы.")


@router.callback_query(F.data.startswith("topic_"), StateFilter(TopicState.showing_topics))
async def handle_topic_selection(callback: CallbackQuery, state: FSMContext):
    """Обработчик выбора конкретного топика"""
    try:
        validation_result = validate_callback_query(callback)
        if not validation_result.is_valid:
            logger.warning(f"Невалидный callback от пользователя {callback.from_user.id}: {validation_result.error}")
            await callback.answer("❌ Ошибка валидации данных")
            return

        topic_index = int(callback.data.split("_")[1])

        data = await state.get_data()
        search_results = data.get("search_results", [])
        current_page = data.get("current_page", 0)

        if topic_index >= len(search_results) or topic_index < 0:
            await callback.answer("❌ Ошибка: неверный топик")
            return

        selected_topic = search_results[topic_index]
        topic_name = selected_topic.get("topic", "Неизвестный топик")
        full_text = selected_topic.get("full_text", "Информация недоступна")

        if len(full_text) > 4000:
            full_text = full_text[:3997] + "..."

        response_text = f"📋 <b>{topic_name}</b>\n\n{full_text}"

        if len(response_text) > 4096:
            await callback.message.answer(f"📋 <b>{topic_name}</b>")
            if len(full_text) > 4096:
                full_text = full_text[:4093] + "..."
            await callback.message.answer(full_text)
        else:
            await callback.message.answer(response_text)

        menu_keyboard = await get_base_menu()
        await callback.message.answer("❓ Могу еще чем-то помочь?", reply_markup=menu_keyboard.as_markup())

        await callback.answer()
        await state.clear()

        logger.info(f"Пользователь {callback.from_user.id} выбрал топик #{topic_index + 1}: {topic_name[:50]}...")

    except ValueError:
        logger.error(f"Неверный формат callback_data: {callback.data}")
        await callback.answer("❌ Ошибка обработки запроса")
    except Exception as e:
        logger.error(f"Ошибка выбора топика: {e}", exc_info=True)
        await callback.answer("❌ Произошла ошибка при получении информации по топику.")
        await state.clear()
