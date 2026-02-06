from typing import Any

from pydantic import BaseModel
from ..clients.base import BaseLlmClient
import os
import anthropic
import asyncio
from ..messages import ActionCall, BaseMessage, LlmUsage, SystemMessage, AIMessage, filter_action_calls, remove_action_calls
from ..types import ToolChoice
from ..utils.action_manager import Actions
from ...utils.model_utils import schema_to_function
from typing import List, get_args
import random


def convert_camel_to_snake(name):
    import re
    # Convert CamelCase to snake_case
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return s1.lower()


# schema = AttackAction.model_json_schema()

def actions_to_tools(actions: list[BaseModel]):
    tools = []
    for action in actions:
        schema = schema_to_function(action)['function']
        if hasattr(action,"_title"):
            name = action._title.default
        else:
            name = convert_camel_to_snake(action.__name__)
        tools.append({
            "name": name,
            "description": schema["description"],
            "input_schema": schema["parameters"],
        })
    return tools


def to_antropic_tool_choice(tool_choice: ToolChoice):
    if tool_choice is None:
        return {"type": "auto"}
    elif isinstance(tool_choice, BaseModel):
        raise ValueError("tool_choice must be a string")
        return {
            "type": "tool",
            "name": get_action_name(tool_choice)    
        }        
    elif isinstance(tool_choice, str):
        if not tool_choice in get_args(ToolChoice):
            raise ValueError(f"Invalid tool_choice string: {tool_choice}. should be one of {get_args(ToolChoice)}")
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            raise ValueError("none tool_choice is not supported by anthropic")
        elif tool_choice == "required":
            return {"type": "any"}
    else:
        raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")
    
    
    
class AnthropicLlmClient(BaseLlmClient):

    def __init__(self, api_key=None):
        super().__init__(
            # client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY")),
            client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY")),
        )
        
    async def before_complete(self, messages: List[dict], actions: any, **kwargs):
        return messages, actions

    async def after_complete(self, completion: anthropic.types.message.Message, **kwargs):
        return completion
    
    
    async def create(
            self,
            model: str,
            temperature: float,
            max_tokens: int,
            system: str,
            messages: List[dict],
            tools: Any,
            tool_choice: str,
            max_retries=10, 
            base_delay=1, 
            max_delay=32,
            betas: list[str] | None = None,
            **kwargs
        ):
        for attempt in range(max_retries):
            try:
                if betas:
                    anthropic_completion = await self.client.beta.messages.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        betas=betas,
                        **kwargs
                    )
                else:
                    anthropic_completion = await self.client.messages.create(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        **kwargs
                    )
                return anthropic_completion
            except anthropic.InternalServerError as e:
                if e.status_code == 529:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    # Add jitter to avoid thundering herd problems
                    delay = delay * random.uniform(0.5, 1.5)
                    print(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise e
        else:
            raise e
                
        

    async def complete(
        self, 
        messages: List[BaseMessage], 
        actions: Actions | List[BaseModel]=[], 
        model="claude-3-5-sonnet-20240620",
        tool_choice: ToolChoice | BaseModel | None = None,
        max_tokens=4096,
        temperature=0,
        run_id: str | None=None, 
        **kwargs
    ):
        # tools = actions_to_tools(actions) if actions else []
        if not isinstance(actions, Actions):
            actions = Actions(actions=actions)            
        tools = actions.to_anthropic()
        system_message = anthropic.NOT_GIVEN
        if isinstance(messages[0], SystemMessage):
            system_message = messages[0].content
            messages = messages[1:]
        # antropic_messages = [m.to_anthropic() for m in filter_action_calls(messages, user_first=True, check_alternation=True)]
        antropic_messages = [m.to_anthropic() for m in messages]
        # log_messages(antropic_messages)
        antropic_messages, actions = await self.before_complete(antropic_messages, actions)
        tool_choice = anthropic.NOT_GIVEN if not actions else to_antropic_tool_choice(tool_choice)
        try:
            anthropic_completion = None
            # anthropic_completion = await self.client.messages.create(
            #     model=model,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     system=system_message,
            #     messages=antropic_messages,
            #     tools=tools,
            #     tool_choice=tool_choice,
            #     **kwargs
            # )
            anthropic_completion = await self.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system_message,
                messages=antropic_messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs
            )
            anthropic_completion = await self.after_complete(anthropic_completion)
            return self.parse_output(anthropic_completion, actions)        
                
        except Exception as e:
            print(antropic_messages)
            # self.serialize_messages(run_id, messages, anthropic_completion)
            raise e
        
    
    def parse_output(self, response: anthropic.types.message.Message, actions: Actions)-> AIMessage:
        content = ''
        action_calls = []
        for content_block in response.content:
            if content_block.type == 'text':
                content = content_block.text
            elif content_block.type == 'tool_use':
                action_instance = actions.from_anthropic(content_block)
                action_calls.append(
                    ActionCall(
                        id=content_block.id,
                        name=content_block.name,
                        action=action_instance
                    ))
                # tool_calls.append({"id": content_block.id, "action": action_instance})
        ai_message = AIMessage(
            id=response.id,
            model=response.model,
            content=content, 
            action_calls=action_calls,
            raw=response,
            usage=LlmUsage(
                prompt_tokens= response.usage.input_tokens,
                completion_tokens= response.usage.output_tokens,
                total_tokens= response.usage.input_tokens + response.usage.output_tokens,
            )
        )
        return ai_message
