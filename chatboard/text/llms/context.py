


from datetime import datetime
from typing import Any, List, Union
import uuid

from chatboard.text.llms.views import BaseModel, Field

from chatboard.text.llms.prompt3 import ChatPrompt
from chatboard.text.llms.conversation import HumanMessage, SystemMessage, AIMessage


class HistoryMessage:

    def __init__(self, view_name, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any, date: any, view_cls=None) -> None:
        self.inputs = inputs
        self.output = output
        self.msgs = msgs
        self.view_name = view_name
        self.view_cls = view_cls
        self.date = date

    def get_input_message(self):
        return self.msgs[-1]
    
    def get_output_message(self):
        return self.output


class History:

    def __init__(self) -> None:
        self.history = []


    def add(self, view_name, view_cls, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any):
        self.history.append(HistoryMessage(
                view_name=view_name, 
                view_cls=view_cls, 
                inputs=inputs, 
                output=output,
                date=datetime.now(),
                msgs=msgs
            ))


    def get(self, view_name: List[str] | str | None=None, top_k=1):
        history = self.history        
        if view_name is None:
            return history[:top_k]
        elif isinstance(view_name, str):
            hist_msgs = [msg for msg in history if msg.view_name == view_name]
            return hist_msgs[:top_k]
        elif isinstance(view_name, list):
            hist_msgs = [msg for msg in history if msg.view_name in view_name]
            return hist_msgs[:top_k]
        else:
            raise ValueError("view_name should be a list or a string")
        

    async def get_inputs(self, view_name: List[str] | str | None=None, top_k=1):
        hist_msgs = self.get(view_name, top_k)        
        return [msg.inputs for msg in hist_msgs]

    async def get_messages(self, view_name: List[str] | str | None=None, top_k=1, msg_type=None):
        hist_msgs = self.get(view_name, top_k)
        history_msgs = []
        for msg in hist_msgs:
            history_msgs.append(msg.get_input_message())
            history_msgs.append(msg.get_output_message())

        if msg_type is not None:
            if msg_type == 'ai':
                history_msgs = [msg for msg in history_msgs if isinstance(msg, AIMessage)]
            elif msg_type == 'user':
                history_msgs = [msg for msg in history_msgs if isinstance(msg, HumanMessage)]            
        return history_msgs
        
    async def __call__(self, view_name: List[str] | str | None=None, top_k=1, msg_type=None):
        return await self.get_messages(view_name, top_k, msg_type)


class Context(BaseModel):
    key: str | int
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    curr_prompt: ChatPrompt = None
    curr_prompt_gen: Any = None
    curr_description: str = None

    message: str = None
    history: History = Field(default_factory=History)
    created_at: datetime = Field(default_factory=datetime.now)


    async def init_context(self):
        raise NotImplementedError
    
    class Config:
        arbitrary_types_allowed = True


    def get_context_view(self):

        if self.curr_prompt is not None:
            prompt_name = self.curr_prompt.__class__.__name__ if isinstance(self.curr_prompt, ChatPrompt) else self.curr_prompt._prompt_name
            return f"""
    the following is the current context of the conversation:
    <context>
        state: {prompt_name}
        description: {self.curr_description}
    </context>
    """
        return """
<context>
no conversation context
</context>
"""