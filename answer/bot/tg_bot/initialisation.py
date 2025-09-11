import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, BotCommandScopeDefault

from answer.handlers.start import start_router
from answer.settings import Settings, get_settings


settings: Settings = get_settings()
logger = logging.getLogger(__name__)

bot: Bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp: Dispatcher = Dispatcher()

dp.include_router(start_router)


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


async def bot_shutdown():
    if bot:
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.session.close()

    logger.info("Bot shutdown completed")
