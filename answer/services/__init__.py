"""Инициализация модуля services."""

from .bot_service import BotService, get_bot_service
from .search_service import SearchService, get_search_service

__all__ = ['get_search_service', 'SearchService', 'get_bot_service', 'BotService']
