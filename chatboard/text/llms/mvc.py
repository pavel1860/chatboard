from enum import Enum
from typing import Any, Iterable, List, Optional, Type
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_core.utils.function_calling import convert_to_openai_tool
# from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
import textwrap
from chatboard.text.llms.conversation import HumanMessage, SystemMessage
import inspect
from abc import ABC, abstractmethod

from chatboard.text.llms.function_utils import call_function, filter_func_args, is_async_function
from chatboard.text.llms.model import iterate_class_fields







# def is_async_function(obj):
#     return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)
def describe_enum(enum_cls: Enum):
    return ", ".join([v.value for v in enum_cls])

class ToolEnum(str, Enum):

    @classmethod
    def render(self):
        return describe_enum(self)
    






            


class Action(BaseModel):

    _hide_name: bool = False

    # def _build_action(self):
    #     for field, info in iterate_class_fields(self):
    #         print(field, info)

    # @abstractmethod    
    async def handle(self, *args, **kwargs):
        return self    

    @classmethod
    def render(self):
        return None
        # return model_to_prompt(convert_to_openai_tool(self), hide_name=self._hide_name)
    
    @classmethod 
    def to_tool(self):
        return None
        # return convert_to_openai_tool(self)







class View(BaseModel):
    
    _actions: Action | List[Action] = []
    _output_model = None

    def __init__(self, **data):
        self._validate_output_format()
        super().__init__(**data)
        


    def render(self):
        return None
        # if self._is_system:
            # return self.render_system()
        # return self.json()

    
    def _validate_output_format(self):
        if self._output_model is not None:
            if not issubclass(self._output_model, BaseModel):
                raise ValueError(f"Output model should be an instance of BaseModel")
        if self._actions:
            for action in self._actions:
                if not issubclass(action, BaseModel):
                    raise ValueError(f"Actions should be instances of Action")
    
    # def render_output(self):
    #     if self._output_model is not None:
    #         if hasattr(self._output_model, 'default'):
    #             return model_to_prompt(convert_to_openai_tool(self._output_model.default), hide_output=True)        
    #         return model_to_prompt(convert_to_openai_tool(self._output_model), hide_output=True)        
    #     if self._actions:
    #         prompt = ""
    #         # for action_name, action_info in self._actions.items():
    #         for action in self._actions:
    #             # prompt += f"\n{action_name}:"
    #             prompt += model_to_prompt(convert_to_openai_tool(action), hide_output=True)
    #             prompt += "\n"
    #         return prompt

    # def render_tools(self):  
    #     if self._output_model is not None:
    #         if hasattr(self._output_model, 'default'):
    #             return convert_to_openai_tool(self._output_model.default)
    #         return convert_to_openai_tool(self._output_model)
        
    #     if self._actions:
    #         actions = []
    #         for action in self._actions:            
    #             actions.append(convert_to_openai_tool(action))
    #         return actions
    
            
    
    # async def render(self):
        # if self._is_system:
            # return self.render_system()
        # return self.dict()
    # @classmethod
    # def render_system_state(self):
    #     return model_to_prompt(convert_to_openai_tool(self), hi)
    
    @classmethod    
    def render_system(self):
        return None
        # return model_to_prompt(convert_to_openai_tool(self), hide_output=True)

    @classmethod
    def render_tool(self):
        return None
    
    
    def vectorize(self):
        return None
    
    # @classmethod 
    # def to_tool(self):
    #     return convert_to_openai_tool(self)
    
    
    # def __call__(self,*args, **kwargs: Any) -> Any:
        # return 
        # content_out = call_function(self.render, *args, **kwargs)
        # content = textwrap.dedent(content_out).strip()
        # if self._is_system:                
        #     return SystemMessage(content=content)
        # else:
        #     return HumanMessage(content=content)

    # async def __call__(self,*args, **kwargs: Any) -> Any:        
    #     content_out = await call_function(self.render, *args, **kwargs)
    #     content = textwrap.dedent(content_out).strip()
    #     if self._is_system:                
    #         return SystemMessage(content=content)
    #     else:
    #         return HumanMessage(content=content)


    # async def _recursive_render(self, **kwargs):
    #     filtered_args = filter_func_args(self.render, kwargs)    
    #     render_output = self.render(**filtered_args)
    #     if isinstance(render_output, str):
    #         return render_output
    #     elif isinstance(render_output, Iterable):
    #         prompt = ""
    #         for child_view in render_output:
    #             if isinstance(child_view, str):
    #                 prompt += "\n" + child_view 
    #             else:
    #                 prompt += "\n" + render_view(child_view, **kwargs) + "\n"
    #         return prompt
    #     else:
    #         raise ValueError(f"Invalid view type: {parent_view}")
