


import uuid
from datetime import datetime
from typing import Any, List, Union

from chatboard.app_manager import app_manager
from chatboard.state.history import History
from chatboard.llms.messages import BaseMessage                                      
from pydantic import BaseModel, Field


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    updated_at: datetime = Field(default_factory=datetime.now)


class Context(BaseModel):
    key: str | int
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    curr_prompt: str | None = None
    curr_prompt_cls: Any = None
    curr_prompt_gen: Any = None
    curr_description: str | None = None

    message: str | None = None
    history: History = Field(default_factory=History)
    created_at: datetime = Field(default_factory=datetime.now)
    session: Session = Field(default_factory=Session)
    add_session_id: bool = True
    
    
    @property
    def session_id(self):
        if not self.add_session_id:
            return None
        return self.session.id
    


    async def init_context(self):
        raise NotImplementedError
    
    class Config:
        arbitrary_types_allowed = True


    def clear_current_prompt(self):
        self.curr_prompt = None
        self.curr_prompt_cls = None
        self.curr_prompt_gen = None
        self.curr_description = None
        self.set_prompt(None)


    def get_prompt(self):
        raise NotImplementedError

    def set_prompt(self, prompt_name: str | None):
        raise NotImplementedError
        
    def get_history(self):
        raise NotImplementedError
    
    def set_history(self, messages: List[BaseMessage]):
        raise NotImplementedError

    async def set_current_prompt(self, prompt_cls):
        self.curr_prompt = prompt_cls.__name__
        await self.set_prompt(prompt_cls.__name__)
        self.curr_prompt_cls = prompt_cls
        
    async def get_current_prompt(self):        
        prompt_name = await self.get_prompt()
        if not prompt_name:
            return None
        prompt = app_manager.prompts.get(prompt_name, None)
        if not prompt:
            raise Exception(f"Prompt {self.curr_prompt} not found in app_manager.prompts")
        return prompt()

    @staticmethod
    async def get(key, **kwargs):
        return None
    
    async def set(self,  key, **kwargs):
        return None
    
    async def delete(self):
        return None