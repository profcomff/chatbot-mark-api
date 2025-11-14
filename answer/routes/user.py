from sqlalchemy import and_
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from answer.models.db import User
from answer.schemas.db_models import StatusMessage
from answer.schemas.telegram import UserInfo
from answer.settings import Settings, get_settings


settings: Settings = get_settings()
engine = create_engine(str(settings.DB_DSN), pool_pre_ping=True, pool_recycle=300)
Session: DbSession = sessionmaker(bind=engine)


async def get_user_by_chat_id(user_chat_id: str) -> UserInfo | StatusMessage:
    with Session() as session:
        user = session.query(User).filter(User.chat_id == user_chat_id).one_or_none()
        if user is not None:
            result = UserInfo.model_validate(user.__dict__)
            return result
        return StatusMessage(status="User is not found", status_code=404)
