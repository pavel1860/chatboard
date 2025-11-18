import json
from typing import List, Type

import anthropic
import openai
from ..exceptions import LLMToolNotFound
from ...utils.model_utils import schema_to_function
from ...utils.string_utils import camel_to_snake
from pydantic import BaseModel, Field


class Actions(BaseModel):
    """handling of tool serialization and deserialization"""
    actions: List[Type[BaseModel]] = Field([], title="Tools", description="The tools the user can use")        
    snake_case: bool = Field(default=True, title="Snake Case", description="If the tools should be converted to snake case")
    
    def __init__(self, actions: List[Type[BaseModel]] = [], snake_case: bool = True):
        actions = actions or []
        super().__init__(actions=actions, snake_case=snake_case)
        self.actions = actions
    
    def __getitem__(self, index):
        return self.actions[index]
    
    def __iter__(self):
        for action in self.actions:
            yield self.get_action_name(action), action
        # return iter(self.actions)
        
    def __len__(self):
        return len(self.actions)
    
    def __bool__(self):
        return bool(self.actions)
    
    def get_action_name(self, action_class: Type[BaseModel]) -> str:
        if hasattr(action_class, "_title"):
            return action_class._title.default
        if self.snake_case:            
            return camel_to_snake(action_class.__name__)
        return action_class.__name__

    
    def add(self, action: Type[BaseModel]):
        if not issubclass(action, BaseModel):
            raise ValueError("action must be a subclass of BaseModel")
        self.actions.append(action)
        
    def extend(self, actions: List[Type[BaseModel]]):
        for action in actions:
            if not issubclass(action, BaseModel):
                raise ValueError("action must be a subclass of BaseModel")
        self.actions.extend(actions)

    def get(self, action_name)-> Type[BaseModel] | None:
        for action in self.actions:
            if self.get_action_name(action) == action_name:
                return action
        return None
    
    
    def from_anthropic(self, content_block: anthropic.types.content_block.ContentBlock)->BaseModel:
        action = self.get(content_block.name)
        if not action:
            raise LLMToolNotFound(content_block.name)
        action_instance = action(**content_block.input)
        return action_instance
    
    # ChatCompletionMessageToolCall
    def from_openai(self, tool_call)->BaseModel:
        tool_args = json.loads(tool_call.function.arguments)
        action = self.get(tool_call.function.name)
        if not action:
            raise LLMToolNotFound(tool_call.function.name)
        action_instance = action(**tool_args)
        return action_instance
            
    
    
    def to_openai_tool(self, action_class: Type[BaseModel]):
        schema = schema_to_function(action_class)
        if hasattr(action_class,"_title"):
            name = action_class._title.default
        else:
            if self.snake_case:
                name = camel_to_snake(action_class.__name__)
            else:
                name = action_class.__name__
        schema["function"]["name"] = name
        return schema
            
            
    def to_openai(self):
        if not self.actions:
            return None
        tools = [self.to_openai_tool(a) for a in self.actions]        
        return tools
    
    
    def to_anthropic_tool(self, action_class: Type[BaseModel]):
        schema = schema_to_function(action_class)['function']
        if hasattr(action_class,"_title"):
            name = action_class._title.default
        else:
            if self.snake_case:
                name = camel_to_snake(action_class.__name__)
            else:
                name = action_class.__name__
        return{
            "name": name,
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }
        
    
    @staticmethod
    def validate_actions(actions: List[Type[BaseModel]]):
        for action in actions:
            if not issubclass(action, BaseModel):
                raise ValueError(f"action must be a subclass of BaseModel, got {action}")
        return True
        

    
    def to_anthropic(self):
        tools = []
        tools = [self.to_anthropic_tool(a) for a in self.actions]            
        return tools
    
