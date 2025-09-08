class ChatBotException(Exception):
    eng: str
    ru: str

    def __init__(self, eng: str, ru: str) -> None:
        self.eng = eng
        self.ru = ru
        super().__init__(eng)


class ObjectNotFound(ChatBotException):
    def __init__(self, obj: type, obj_id_or_name: int | str):
        super().__init__(
            f"Object {obj.__name__} {obj_id_or_name=} not found",
            f"Объект {obj.__name__}  с идентификатором {obj_id_or_name} не найден",
        )


class UpdateError(ChatBotException):
    def __init__(self, msg: str):
        super().__init__(
            f"{msg} Conflict with update a resource that already exists or has conflicting information.",
            f"{msg} Конфликт с обновлением ресурса, который уже существует или имеет противоречивую информацию.",
        )
