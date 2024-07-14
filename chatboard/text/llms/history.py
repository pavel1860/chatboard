


from datetime import datetime
from typing import Any, List, Union
import uuid

from chatboard.text.llms.views import BaseModel, Field
from chatboard.text.llms.conversation import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

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


class HistoryMessage(BaseModel):
    message: AIMessage | HumanMessage | None = None
    prompt_name: str
    inputs: Any = None
    run_id: str
    date: datetime = Field(default_factory=datetime.now)    

    class Config:
        arbitrary_types_allowed = True
    


class History:

    def __init__(self, add_actions=False) -> None:
        self.history: List[HistoryMessage] = []
        self.add_actions = add_actions

    
    # def add(self, view_name, view_cls, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any, render_output: RenderOutput):
    #     self.history.append(HistoryMessage(
    #             view_name=view_name, 
    #             view_cls=view_cls, 
    #             inputs=inputs, 
    #             output=output,
    #             date=datetime.now(),
    #             msgs=msgs,
    #             render_output=render_output
    #         ))

    def add_many(self, messages: List[AIMessage | HumanMessage | BaseModel], prompt_name: str, run_id: str, inputs: Any = None):
        for message in messages:
            if message.is_history == False and message.is_example == False and not isinstance(message, SystemMessage):
                self.add(message=message, prompt_name=prompt_name, run_id=run_id, inputs=inputs)


    def add(self, message: AIMessage | HumanMessage | SystemMessage | BaseModel, prompt_name: str, run_id: str, inputs: Any = None):        
        if isinstance(message, BaseMessage):
            msg = message.model_copy()
            msg.is_history = True
        elif isinstance(message, BaseModel):        
            msg_str = message.__class__.__name__ + "\n" + message.json()
            msg = AIMessage(
                content=msg_str,
                is_history=True,
                is_output=True
            )
        else:
            print("message>>>>", message)
            raise ValueError("message should be a BaseMessage or BaseModel")        
        self.history.append(HistoryMessage(
            message=msg,
            prompt_name=prompt_name,
            run_id=run_id,
            inputs=inputs 
        ))


    def get_messages(self, top_k=1, prompt: List[str] | str | None=None, run_id=None, add_actions=None, message_type: str | None=None):
        messages = self.get(top_k=top_k, prompt=prompt, run_id=run_id, add_actions=add_actions, message_type=message_type)
        return [msg.message for msg in messages]

    def get_input_messages(self, input_key: str, top_k=1, prompt: List[str] | str | None=None, as_texts=False, message_type="bot"):
        hmsgs = self.get(top_k=top_k, prompt=prompt, message_type=message_type)
        history_messages = []
        for hmsg in hmsgs:
            user_message = hmsg.inputs.get(input_key, None)
            if user_message is None:
                continue
            if not as_texts:
                user_message = HumanMessage(content=user_message)
            history_messages.append(user_message)
            bot_message = hmsg.message
            if as_texts:
                bot_message = bot_message.content
            history_messages.append(bot_message)
        return history_messages


    def get(self, top_k=1, prompt: List[str] | str | None=None, run_id=None, add_actions=None, message_type: str | None=None):
        add_actions = self.add_actions if add_actions is None else add_actions
        history = self.history
        if not add_actions:
            history = [msg for msg in history if not msg.message.is_output]
        if run_id is not None:
            history = [msg for msg in history if msg.run_id == run_id]
        if prompt is not None:
            if isinstance(prompt, str):
                history = [msg for msg in history if msg.prompt_name == prompt]
            elif isinstance(prompt, list):
                history = [msg for msg in history if msg.prompt_name in prompt]
            else:
                raise ValueError("view_name should be a list or a string")
        if message_type is not None:
            if message_type == 'bot':
                history = [msg for msg in history if isinstance(msg.message, AIMessage)]
            elif message_type == 'user':
                history = [msg for msg in history if isinstance(msg.message, HumanMessage)]
            elif message_type == 'system':
                history = [msg for msg in history if isinstance(msg.message, SystemMessage)]
            else:
                raise ValueError("message_type should be 'bot', 'user' or 'system'")
        return history[-top_k:]
        # if prompt is None:
        #     return history[:top_k]
        # elif isinstance(prompt, str):
        #     hist_msgs = [msg for msg in history if msg.prompt_name == prompt]
        #     return hist_msgs[:top_k]
        # elif isinstance(prompt, list):
        #     hist_msgs = [msg for msg in history if msg.prompt_name in prompt]
        #     return hist_msgs[:top_k]
        # else:
        #     raise ValueError("view_name should be a list or a string")
        

    async def get_inputs(self, view_name: List[str] | str | None=None, top_k=1):
        hist_msgs = self.get(view_name, top_k)        
        return [msg.inputs for msg in hist_msgs]

    # async def get_messages(self, view_name: List[str] | str | None=None, top_k=1, msg_type=None):
    #     hist_msgs = self.get(view_name, top_k)
    #     history_msgs = []
    #     for msg in hist_msgs:
    #         history_msgs.append(msg.get_input_message())
    #         history_msgs.append(msg.get_output_message())

    #     if msg_type is not None:
    #         if msg_type == 'ai':
    #             history_msgs = [msg for msg in history_msgs if isinstance(msg, AIMessage)]
    #         elif msg_type == 'user':
    #             history_msgs = [msg for msg in history_msgs if isinstance(msg, HumanMessage)]            
    #     return history_msgs
        
    async def __call__(self, view_name: List[str] | str | None=None, top_k=1, msg_type=None):
        return await self.get_messages(view_name, top_k, msg_type)
