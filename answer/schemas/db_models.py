from pydantic import BaseModel


class StatusMessage(BaseModel):
    status_code: int
    status: str | None = None
    message: str | None = None
