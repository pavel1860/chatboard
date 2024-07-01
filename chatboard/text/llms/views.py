from enum import Enum
from typing import Any, List, Type
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_core.utils.function_calling import convert_to_openai_tool
import textwrap
from chatboard.text.llms.conversation import HumanMessage, SystemMessage
import inspect
from abc import ABC, abstractmethod

from chatboard.text.llms.function_utils import call_function, is_async_function







# def is_async_function(obj):
#     return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)
def describe_enum(enum_cls: Enum):
    return ", ".join([v.value for v in enum_cls])

class ToolEnum(str, Enum):

    @classmethod
    def render(self):
        return describe_enum(self)
    

def parse_properites(properties, add_type=True, add_constraints=True, tabs="\t"):
    prompt = ""
    for prop, value in properties.items():
        param_promp = f"\n{tabs}{prop}"
        if 'allOf' in value: 
            obj = value['allOf'][0]
            prompt += f"\n{tabs}{obj['title']}:"
            prompt += parse_properites(obj['properties'], tabs=tabs+"\t")
        else:
            if add_type:
                param_promp += f":({value['type']})"
            if 'description' in value:
                param_promp += f" {value['description']}"
            if add_constraints and ('minimum' in value or 'maximum' in value):
                param_promp += f". should be"
                if 'minimum' in value:
                    param_promp += f" minimum {value['minimum']}"
                if 'maximum' in value:
                    param_promp += f" maximum {value['maximum']}"
                param_promp += "."
            prompt += param_promp
    return prompt



def model_to_prompt(tool_dict, add_type=True, add_constraints=True):    
    tool_function = tool_dict['function']
    prompt = f"""{tool_function["name"]}:"""
    properties = tool_dict['function']["parameters"]['properties']
    prompt += parse_properites(properties, add_type, add_constraints)
    return prompt 




class ViewModel(BaseModel):
    _is_system: bool = False

    def render(self):
        if self._is_system:
            return self.render_system()
        return self.dict()
    
    async def render(self):
        if self._is_system:
            return self.render_system()
        return self.dict()
    
    @classmethod    
    def render_system(self):
        return model_to_prompt(convert_to_openai_tool(self))
    
    def vectorize(self):
        return None
    
    @classmethod 
    def to_tool(self):
        return convert_to_openai_tool(self)
    
    

    async def __call__(self,*args, **kwargs: Any) -> Any:
        # if is_async_function(self.render):
        #     content_out = await self.render(**kwargs)
        # else:
        #     content_out = self.render(**kwargs)
        content_out = await call_function(self.render, *args, **kwargs)
        content = textwrap.dedent(content_out).strip()
        if self._is_system:                
            return SystemMessage(content=content)
        else:
            return HumanMessage(content=content)
        

            


class Action(BaseModel):

    # @abstractmethod    
    async def handle(self, *args, **kwargs):
        return self

    @classmethod
    def render(self):
        return model_to_prompt(convert_to_openai_tool(Action))
    
    @classmethod 
    def to_tool(self):
        return convert_to_openai_tool(self)



