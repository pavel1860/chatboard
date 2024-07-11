
from collections.abc import Iterable
import json
from typing import Any, List, Optional, Union
# from langchain_core.pydantic_v1 import BaseModel, ConfigDict, Field

from chatboard.text.llms.conversation import AIMessage, SystemMessage, HumanMessage
from chatboard.text.llms.function_utils import call_function, filter_func_args, flatten_list, is_async_function
from chatboard.text.llms.view_renderer import RenderOutput, ViewRenderer
from chatboard.text.llms.mvc import Action, View, Type, BaseModel, Field
from .llm2 import AzureOpenAiLLM, OpenAiLLM
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




class PromptResponse(BaseModel):
    view: View | BaseModel
    run_id: str
    output: AIMessage | BaseModel
    conversation: List[Union[AIMessage, HumanMessage, SystemMessage]]
    render_output: RenderOutput

    class Config:
        arbitrary_types_allowed = True








class ChatPrompt(BaseModel):
    name: Optional[str] = None
    model: str= "gpt-3.5-turbo-0125"
    llm: Union[OpenAiLLM, AzureOpenAiLLM] = Field(default_factory=OpenAiLLM)


    background: Optional[str] = None
    task: Optional[str] = None
    rules: Optional[str] = None    

    view: Optional[View | BaseModel] = None

    input_model: Optional[Type[BaseModel]] = None
    response_model: Optional[Type[BaseModel]] = None

    _rag: Optional[Type[BaseModel]] = None
    rag_top_k: int = 5

    is_traceable: bool=True
    add_to_history: bool=True
    tools: Optional[List[Type[Action]]] = None

    _view_renderer: ViewRenderer = ViewRenderer(system_indent=4, view_to_prompt=True)

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
    

    async def use_rag(self):
        return None
    
    async def add_rag(self):
        return None

    
    async def _build_conversation(
            self,             
            render_output: RenderOutput, 
            view: View | BaseModel=None,
            context=None, 
            response_model: BaseModel | None=None,
            **kwargs: Any
        ):
               
        conversation = []
        kwargs['context'] = context

        system_prompt = ""
        if self.background:
            system_prompt += f"{self.background}\n"
        if self.task:
            system_prompt += f"{self.task}\n"
        
        system_prompt += render_output.system_prompt        
        
        if render_output.is_actions:
            system_prompt += "\nyou should use one of the following actions:\n"
            system_prompt += render_output.action_prompt


        response_model = response_model or self.response_model
        if render_output.is_output:
            if response_model:
                raise ValueError("You can have either a response model or an output model, not both.")

            system_prompt += "\nyou should use the following format for the output:\n"
            system_prompt += render_output.output_model_prompt

        response_model = response_model or self.response_model
        if response_model:
            if response_model.__doc__:
                system_prompt += f"\n{response_model.__doc__}\n"
            else:
                system_prompt += "\nyou should use the following format for the response:\n"
            system_prompt += self._view_renderer.render_tool_aux(response_model)

        if self.rules:
            system_prompt += self.rules + "\n"
        
        conversation.append(SystemMessage(content=system_prompt))

        if self._rag:
            docs = await self.use_rag(render_output.view_prompt)
            for doc in docs:
                user_msg = HumanMessage(content=doc.input)
                ai_msg = AIMessage(content=doc.output)
                conversation.append(user_msg)
                conversation.append(ai_msg)

        conversation.append(HumanMessage(content=render_output.view_prompt))

        return conversation


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

    async def call(
            self, 
            view_prompt: View | BaseModel=None, 
            context=None, 
            response_model: BaseModel | None=None,
            tracer_run=None, 
            output_context=False, 
            **kwargs: Any) -> Any:
            
            if view_prompt is not None:
                kwargs['view_prompt'] = view_prompt
            if 'model' not in kwargs:
                kwargs["model"] = self.model
            log_kwargs = {}
            log_kwargs.update(kwargs)            
            
            if view_prompt is not None:
                log_kwargs['view_prompt'] = view_prompt
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
                

                view_prompt = view_prompt or self.view
                
                render_output = await self._view_renderer.render_view(view_prompt, **kwargs)

                msgs = await self._build_conversation(render_output=render_output, context=context, **kwargs)

                # tools = self.convert_to_openai_tool()
                # if not tools:
                #     tools = None

                response_model = response_model or self.response_model or render_output.output_model

                tool_choice = await self._call_tool_choice(context=context, **kwargs)

                completion_msg = await self.llm.complete(
                        msgs=msgs,
                        tools=list(render_output.actions.values()) if not response_model else None,
                        tool_choice=tool_choice if not response_model else None,
                        response_model=response_model,
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
                # if completion_msg.tool_calls is not None:
                #     completion_msg = await self._handle_tool_call(context, completion_msg)
                # else:  
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
                
                if output_context:
                    return PromptResponse(
                        run_id=str(prompt_run.id),
                        view=view_prompt,
                        render_output=render_output,
                        output=completion_msg,
                        conversation=msgs
                    )

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