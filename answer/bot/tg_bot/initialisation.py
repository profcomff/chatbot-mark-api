import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BotCommandScopeDefault

from answer.handlers.ask_bot import router as router_ask_bot
from answer.handlers.info import router as info_router
from answer.handlers.start import start_router
from answer.settings import Settings, get_settings


settings: Settings = get_settings()
logger = logging.getLogger(__name__)

storage = MemoryStorage()

bot: Bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp: Dispatcher = Dispatcher(storage=storage)

dp.include_router(start_router)
dp.include_router(info_router)
dp.include_router(router_ask_bot)


async def setup_bot():
    logger.info("Setting up bot commands and webhook")

    commands = [BotCommand(command='start', description='Старт')]
    await bot.set_my_commands(commands, BotCommandScopeDefault())
    await bot.set_webhook(f"{settings.BASE_URL}{settings.WEBHOOK_PATH}")

    logger.info("Bot setup completed")


async def bot_startup():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting FastAPI app with bot integration")

    await setup_bot()

    logger.info("Bot initialized and webhook set")
    return bot, dp


async def bot_shutdown():
    if bot:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.session.close()

    logger.info("Bot shutdown completed")


def get_bot_objects():  # вот это здорово придумал конечно))
    """Возвращает объекты бота и диспетчера"""
    return bot, dp
