"""Сервисный слой для поиска и генерации ответов."""

import logging
from typing import Any, Dict, Optional

import httpx

from answer.schemas.api_models import ResponseResult, SearchResult, UserInput
from answer.settings import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class SearchService:
    """Сервис для поиска документов и генерации ответов."""

    def __init__(self):
        self._app_state: Optional[Dict[str, Any]] = None

    def set_app_state(self, app_state: Dict[str, Any]) -> None:
        """Устанавливает состояние приложения с инициализированными компонентами."""
        self._app_state = app_state

    async def _get_context_via_api(self, user_chat_id: str) -> str:
        """Получает контекст диалогов пользователя через API."""
        try:
            base_url = f"http://{settings.HOST}:{settings.PORT}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/users/{user_chat_id}/context",
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data.get("context", "")
                elif response.status_code == 404:
                    return ""
                else:
                    logger.error(f"Ошибка получения контекста: {response.status_code} - {response.text}")
                    return ""

        except Exception as e:
            logger.error(f"Ошибка HTTP-запроса получения контекста: {e}", exc_info=True)
            return ""

    async def _build_enhanced_query(self, user_input: str, user_chat_id: str) -> str:
        """Создает расширенный запрос с контекстом."""
        try:
            context = await self._get_context_via_api(user_chat_id)

            if not context:
                return user_input

            enhanced_query = f"""Контекст предыдущих диалогов:
{context}

Текущий вопрос: {user_input}"""

            logger.info(f"Сформирован расширенный запрос для пользователя {user_chat_id}")
            return enhanced_query

        except Exception as e:
            logger.error(f"Ошибка формирования расширенного запроса: {e}", exc_info=True)
            return user_input

    async def search_and_generate(self, user_input: UserInput) -> ResponseResult:
        """
        Выполняет поиск документов и опционально генерирует AI-ответ.

        Args:
            user_input: Входные данные от пользователя

        Returns:
            ResponseResult: Результат с найденными документами и/или AI-ответом

        Raises:
            RuntimeError: Если сервис не инициализирован
        """
        if not self._app_state:
            raise RuntimeError("SearchService не инициализирован. Вызовите set_app_state().")

        try:
            enhanced_query = user_input.text
            if user_input.user_chat_id:
                enhanced_query = await self._build_enhanced_query(user_input.text, user_input.user_chat_id)

            results, combined_text = await self._perform_search(enhanced_query)

            search_results = [
                SearchResult(
                    topic=result.get("topic", ""),
                    full_text=result.get("full_text", ""),
                    metadata=result.get("metadata", {}),
                )
                for result in results
            ]

            ai_answer = None
            if user_input.generate_ai_response and combined_text:
                ai_answer = await self._generate_ai_answer(combined_text, user_input.text)

            return ResponseResult(results=search_results, ai_answer=ai_answer)

        except Exception as e:
            logger.error(f"Ошибка в SearchService.search_and_generate: {e}", exc_info=True)
            return ResponseResult(results=[], message=f"Произошла ошибка при обработке запроса: {str(e)}")

    async def _perform_search(self, query: str) -> tuple:
        """Выполняет поиск документов."""
        from search.search import get_context

        return get_context(
            query=query,
            key_words_dict=self._app_state["keywords_dict"],
            ensemble_retriever=self._app_state["ensemble_retriever"],
            vector_store=self._app_state["vector_store"],
            ensemble_k=settings.ensemble_k,
            verbose=True,
        )

    async def _generate_ai_answer(self, context: str, question: str) -> str:
        """Генерирует AI-ответ на основе контекста."""
        from llm.llm import get_answer

        return get_answer(
            context=context, question=question, credentials=self._app_state["credentials"], settings=settings
        )


_search_service = SearchService()


def get_search_service() -> SearchService:
    """Возвращает экземпляр SearchService."""
    return _search_service