from abc import abstractmethod
from typing import Protocol
import datetime as dt

from pydantic import BaseModel, Field



class BaseProto(BaseModel):
    id: int | None = None
    created_at: dt.datetime = Field(default_factory=dt.datetime.now)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.now)
    
    
    @staticmethod
    async def get(id: int) -> "BaseProto":
        raise NotImplementedError

    @abstractmethod
    async def save(self) -> "BaseProto":
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class BranchProto(BaseProto):
    message_order: int = Field(0, ge=0)
    forked_from_message: "MessageProto | None" = Field(default=None)
    forked_from_message_order: int | None = Field(default=None)
    
    @abstractmethod
    async def get_messages(self, limit: int = 10, offset: int = 0) -> list["MessageProto"]:
        raise NotImplementedError
    
    @abstractmethod
    async def get_turns(self, limit: int = 10, offset: int = 0) -> list["TurnProto"]:
        raise NotImplementedError



class TurnProto(BaseProto):
    branch: BranchProto = Field(...)
    messages: list["MessageProto"] = Field(default_factory=list)
    local_state: dict = Field(default_factory=dict)
    
    @abstractmethod
    async def get_messages(self, limit: int = 10, offset: int = 0) -> list["MessageProto"]:
        raise NotImplementedError




class MessageProto(BaseProto):
    role: str = Field(...)
    name: str = Field(...)
    content: str = Field(...)
    blocks: list[dict] | None = Field(default=None)
    run_id: str | None = Field(default=None)
    platform_id: str | None = Field(default=None)
    ref_id: str | None = Field(default=None)
    branch_order: int = Field(...)
    branch: BranchProto = Field()
    turn: TurnProto = Field()

    
    