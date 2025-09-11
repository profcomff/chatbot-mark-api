import datetime
import logging

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from answer.models.db import User
from answer.settings import Settings, get_settings


logger = logging.getLogger(__name__)
start_router = Router()
settings: Settings = get_settings()
engine = create_engine(str(settings.DB_DSN), pool_pre_ping=True, pool_recycle=300)
Session: DbSession = sessionmaker(bind=engine)


@start_router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    try:
        logger.info(f"Received /start command from user {message.from_user.id}")
        with Session() as session:
            User.create(
                chat_id=message.chat.id, create_ts=datetime.datetime.now(datetime.timezone.utc), session=session
            )
            session.commit()
        await message.answer(f"Привет, <b>{message.from_user.full_name}</b>! Как дела?")
        logger.info("Response sent successfully")
    except Exception as e:
        logger.error(f"Error in start handler: {e}", exc_info=True)
        await message.answer("Произошла ошибка. Попробуйте позже.")
