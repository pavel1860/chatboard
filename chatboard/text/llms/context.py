


from datetime import datetime
from typing import Any, List, Union
import uuid


from chatboard.text.app_manager import app_manager
# from chatboard.text.llms.history import History
from chatboard.text.llms.history2 import History
from chatboard.text.llms.view_renderer import RenderOutput
from chatboard.text.llms.views import BaseModel, Field

from chatboard.text.llms.conversation import BaseMessage, HumanMessage, SystemMessage, AIMessage


# class HistoryMessage:

#     def __init__(self, view_name, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any, render_output: RenderOutput, date: any, view_cls=None) -> None:
#         self.inputs = inputs
#         self.output = output
#         self.msgs = msgs
#         self.view_name = view_name
#         self.view_cls = view_cls
#         self.date = date
#         self.render_output = render_output

#     def get_input_message(self):
#         return self.msgs[-1]
    
#     def get_output_message(self):
#         return self.output



class Context(BaseModel):
    key: str | int
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    curr_prompt: str | None = None
    curr_prompt_cls: Any = None
    curr_prompt_gen: Any = None
    curr_description: str = None

    message: str = None
    history: History = Field(default_factory=History)
    created_at: datetime = Field(default_factory=datetime.now)


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

    def set_prompt(self, prompt_name: str):
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

    # def get_current_prompt(self):
    #     if not self.curr_prompt:
    #         return None
    #     if self.curr_prompt_cls is not None:
    #         prompt = self.curr_prompt_cls
    #     else:
    #         prompt = app_manager.prompts.get(self.curr_prompt, None)
    #     if not prompt:
    #         raise Exception(f"Prompt {self.curr_prompt} not found in app_manager.prompts")
    #     return prompt()

        


#     def get_context_view(self):

#         if self.curr_prompt is not None:
#             prompt_name = self.curr_prompt.__class__.__name__ if isinstance(self.curr_prompt, ChatPrompt) else self.curr_prompt._prompt_name
#             return f"""
#     the following is the current context of the conversation:
#     <context>
#         state: {prompt_name}
#         description: {self.curr_description}
#     </context>
#     """
#         return """
# <context>
# no conversation context
# </context>
# """