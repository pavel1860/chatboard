
import os
from typing import List, Type

import openai
from ..clients.base import BaseLlmClient
from ..messages import AIMessage, ActionCall, BaseMessage, LlmUsage, filter_action_calls, validate_msgs
from ..types import ToolChoice
from ..utils.action_manager import Actions
from pydantic import BaseModel


class OpenAiLlmClient(BaseLlmClient):


    def __init__(self, api_key=None):
        super().__init__(
            client = openai.AsyncClient(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        )
        


    async def complete(
        self, 
        messages: List[BaseMessage], 
        actions: Actions | List[Type[BaseModel]]=[], 
        model="gpt-4o",
        tool_choice: ToolChoice | BaseModel | None = openai.NOT_GIVEN,
        run_id: str | None=None, 
        **kwargs
    ):
        if not isinstance(actions, Actions):
            actions = Actions(actions=actions)
        # tools = [schema_to_function(a) for a in actions] if actions else None
        tools = actions.to_openai()
        oai_messages = [msg.to_openai() for msg in filter_action_calls(messages)]
        if isinstance(tool_choice, BaseModel):
                tool_choice =  {"type": "function", "function": {"name": tool_choice.__class__.__name__}}            
        try:
            openai_completion = None
            openai_completion = await self.client.chat.completions.create(
                messages=oai_messages,
                tools=tools,
                model=model,
                tool_choice=tool_choice,
                **kwargs
            )
            return self.parse_output(openai_completion, actions)
        except Exception as e:
            # self.serialize_messages(run_id, messages, openai_completion)
            print(oai_messages)                
            raise e
            # raise BadClientLlmRequest(str(e))
        

    
    def parse_output(self, response, actions: Actions):
        try:
            output = response.choices[0].message
            tool_calls = []
            if output.tool_calls:
                for tool_call in output.tool_calls:
                    action_instance = actions.from_openai(tool_call)
                    # tool_calls.append({"id": tool_call.id, "action": action_instance})
                    tool_calls.append(
                        ActionCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            action=action_instance
                        ))
            ai_message = AIMessage(
                id=response.id,
                model=response.model,
                content=output.content,
                action_calls=tool_calls, 
                raw=response,
                usage=LlmUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens= response.usage.completion_tokens,
                    total_tokens= response.usage.total_tokens,
                )
            )
            return ai_message
        except Exception as e:
            raise e



# class OpenAiLlmClient(BaseLlmClient):


#     def __init__(self, api_key=None):
#         super().__init__(
#             client = openai.AsyncClient(
#                 api_key=api_key or os.getenv("OPENAI_API_KEY")
#             )
#         )

#     def preprocess(self, msgs):
#         return [msg.to_openai() for msg in validate_msgs(msgs)]
#         # return [msg.to_openai() for msg in msgs if msg.is_valid()]

#     async def complete(self, msgs, tools=None, run_id: str | None=None, **kwargs):
#         msgs = self.preprocess(msgs)
#         openai_completion = await self.client.chat.completions.create(
#             messages=msgs,
#             tools=tools,
#             **kwargs
#         )
#         return openai_completion
