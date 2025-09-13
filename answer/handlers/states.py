from aiogram.fsm.state import State, StatesGroup


class QuestionState(StatesGroup):
    """Состояния для обработки вопросов пользователя"""

    waiting_for_question = State()
    waiting_for_response_type = State()


class TopicState(StatesGroup):
    """Состояния для работы с релевантными топиками"""

    showing_topics = State()
