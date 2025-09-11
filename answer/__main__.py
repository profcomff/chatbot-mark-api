import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeDefault, Update
from fastapi import Request

from answer.routes.base import app
from answer.settings import Settings, get_settings


settings: Settings = get_settings()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Starting FastAPI app with bot integration")

        uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
    except (KeyboardInterrupt, SystemExit):
        logger.info("Application stopped")
