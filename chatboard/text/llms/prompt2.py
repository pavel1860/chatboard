
from collections.abc import Iterable
import json
from typing import Any, List, Optional, Union
from langchain_core.pydantic_v1 import BaseModel, ConfigDict, Field

from chatboard.text.llms.conversation import SystemMessage, HumanMessage
from chatboard.text.llms.function_utils import call_function, filter_func_args, flatten_list, is_async_function
from chatboard.text.llms.views import Action, ViewModel, Type
from .llm import AzureOpenAiLLM, OpenAiLLM
from .tracer import Tracer
import textwrap
import inspect
import asyncio
import itertools


# def is_async_function(obj):
#     return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)

# def get_func_args(func):
#     return list(inspect.signature(func).parameters.keys())

# def flatten_list(nested_list):
#     flat_list = []
#     for item in nested_list:
#         if isinstance(item, list):
#             flat_list.extend(flatten_list(item))
#         else:
#             flat_list.append(item)
#     return flat_list

# def filter_func_args(func, args):
#     return {k: v for k, v in args.items() if k in get_func_args(func)}


# async def call_function(func, *args, **kwargs):
#     kwargs = filter_func_args(func, kwargs)
#     if inspect.iscoroutinefunction(func):
#         return await func(*args, **kwargs)
#     return func(*args, **kwargs)







class ChatPrompt(BaseModel):
    name: Optional[str] = None
    model: str= "gpt-3.5-turbo-0125"
    llm: Union[OpenAiLLM, AzureOpenAiLLM] = Field(default_factory=OpenAiLLM)
    is_traceable: bool=True
    add_to_history: bool=True
    tools: Optional[List[Type[Action]]] = None


    def __init__(self, **data):
        super().__init__(**data)
        if self.name is None:
            self.name = self.__class__.__name__

        # tool_fields = [field for field in self.__class__.__fields__.items() if isinstance(field[1].annotation, type) and issubclass(field[1].annotation, Action)]
        # self.tools = {tool_fields[1].annotation.__name__ : {
        #     "name": tool_fields[0],
        #     "info": tool_fields[1],            
        # } for tool_fields in tool_fields}
        self._build_tool_lookup()

    class Config:
        arbitrary_types_allowed = True

    def _build_tool_lookup(self):
        tool_fields = [field for field in self.__class__.__fields__.items() if isinstance(field[1].annotation, type) and issubclass(field[1].annotation, Action)]
        self.tools = {tool_fields[1].annotation.__name__ : {
            "name": tool_fields[0],
            "info": tool_fields[1],            
        } for tool_fields in tool_fields}

    async def _call_tool_choice(self, context, **kwargs):
        tool_choice = await self.tool_choice(context=context, **kwargs)
        if tool_choice is not None:
            if not issubclass(tool_choice, ChatPrompt):
                raise ValueError("tool_choice must be an instance of Action")
            return tool_choice.to_tool()
        return None

    async def tool_choice(self, context=None, **kwargs: Any):
        return None


    async def complete(self):
        return []    
    

    async def _build_conversation2(self, context=None, **kwargs: Any):
        conversation = []
        kwargs['context'] = context
        filtered_args = filter_func_args(self.complete, kwargs)
        if is_async_function(self.complete):
            completion_views = await self.complete(**filtered_args)
        else:
            completion_views = self.complete(**filtered_args)
        
        conversation = await asyncio.gather(*completion_views)
        flat_conversation = flatten_list(conversation)
        return flat_conversation
    
    async def _build_conversation(self, context=None, **kwargs: Any):
        conversation = []
        kwargs['context'] = context


        def render_view(parent_view, **kwargs):
            filtered_args = filter_func_args(parent_view.render, kwargs)    
            render_output = parent_view.render(**filtered_args)
            if isinstance(render_output, str):
                return render_output
            elif isinstance(render_output, Iterable):
                prompt = ""
                for child_view in render_output:
                    if isinstance(child_view, str):
                        prompt += "\n" + child_view 
                    else:
                        prompt += "\n" + render_view(child_view, **kwargs) + "\n"
                return prompt
            else:
                raise ValueError(f"Invalid view type: {parent_view}")

                
        conversation = []
        kwargs['context'] = context
        filtered_args = filter_func_args(self.complete, kwargs)
        message_views = await call_function(self.complete, **filtered_args)
        for mv in message_views:
            rendered_prompt = render_view(mv, **kwargs)
            if mv._is_system:
                conversation.append(SystemMessage(content=rendered_prompt))
            else:
                conversation.append(HumanMessage(content=rendered_prompt))
        return conversation
        # return conversation


    async def preprocess(self, **kwargs: Any):
        return kwargs
    

    def convert_to_openai_tool(self):
        return [tool_cls["info"].annotation.to_tool() for tool_cls in self.tools.values()]
    

    async def __aiter__(self):
        return self
    
    async def __anext__(self, *args: Any, **kwds: Any):
        return await self.call(*args, **kwds) 
    

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        return await self.call(*args, **kwds)
    
    def bind(self, *partial_args, **partial_kwargs):
        async def call( **kwargs: Any): 
            kwargs.update(partial_kwargs)
            return await self.call(*partial_args, **kwargs)
        call._prompt_name = self.__class__.__name__
        return call

    async def call(self, prompt=None, context=None, tracer_run=None, output_conversation=False, **kwargs: Any) -> Any:
            if prompt is not None:
                kwargs['prompt'] = prompt
            if 'model' not in kwargs:
                kwargs["model"] = self.model
            log_kwargs = {}
            log_kwargs.update(kwargs)            
            
            if prompt is not None:
                log_kwargs['prompt'] = prompt
            with Tracer(
                is_traceable=self.is_traceable,
                tracer_run=tracer_run,
                name=self.name,
                run_type="prompt",
                inputs={
                    "input": log_kwargs,
                    # "messages": conversation.messages
                },
                # extra=extra,
            ) as prompt_run:
                msgs = await self._build_conversation(context=context, **kwargs)
                msgs = [m for m in msgs if m is not None]

                tools = self.convert_to_openai_tool()
                if not tools:
                    tools = None

                tool_choice = await self._call_tool_choice(context=context, **kwargs)

                completion_msg = await self.llm.complete(
                        msgs=msgs,
                        tools=tools,
                        tool_choice=tool_choice,
                        tracer_run=prompt_run, 
                        **kwargs
                    )
                
                prompt_run.end(outputs={'output': completion_msg})
                #TODO need to add history from outside
                # if context and self.add_to_history:
                #     context.history.add(
                #             view_name=self.name, 
                #             view_cls=self.__class__, 
                #             inputs=log_kwargs, 
                #             output=completion_msg, 
                #             msgs=msgs
                #         )
                if completion_msg.tool_calls is not None:
                    completion_msg = await self._handle_tool_call(context, completion_msg)
                else:  
                    completion_msg = await call_function(self.on_complete, context, completion_msg)              
                    # completion_msg = await self.on_complete(context, completion_msg)
                    # for tool_call in completion_msg.tool_calls:
                    #     tool_args = json.loads(tool_call.function.arguments)
                    #     toll_cls = self.tools['UserDetails']['info'].default.__class__
                    #     tool = toll_cls(**tool_args)
                    #     _attr = getattr(self, self.tools[tool_call.function.name]["name"])
                    #     output = await _attr.handle(tool)
                    #     completion_msg = output
                if completion_msg is None:
                    raise ValueError("Prompt did not return a completion output.")

                return completion_msg
            


    async def _handle_tool_call(self, context, completion_msg):
        for tool_call in completion_msg.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            tool_cls = self.tools[tool_call.function.name]['info'].default.__class__
            output = await self.handle(context, tool_cls, tool_args)
            return output
        

    async def handle(self, context, tool_cls, tool_args):
        tool = tool_cls(**tool_args)
        tool_args['context'] = context
        return await call_function(tool.handle, **tool_args)
    

    async def on_complete(self, context, completion_msg):
        return completion_msg
    

    
    # async def _handle_tool_call(self, context, completion_msg):
    #     for tool_call in completion_msg.tool_calls:
    #         tool_args = json.loads(tool_call.function.arguments)
    #         tool_cls = self.tools[tool_call.function.name]['info'].default.__class__
    #         # tool = tool_cls(**tool_args)
    #         # _attr = getattr(self, self.tools[tool_call.function.name]["name"])
    #         # output = await _attr.handle(tool)
    #         output = await self.handle(context, tool_cls, tool_args)
    #         return output
        
    # async def handle(self, context, tool_cls, tool_args):
    #     tool = tool_cls(**tool_args)
    #     tool_args['context'] = context
    #     kwargs = filter_func_args(tool.handle, tool_args)
    #     return await tool.handle(**kwargs)
    #     # _attr = getattr(self, self.tools[tool_call.function.name]["name"])
    #     # output = await _attr.handle(tool)
    #     # return output