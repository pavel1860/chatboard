
import asyncio
import inspect
import json
from functools import wraps
from typing import (Any, Awaitable, Callable, Dict, Generator, List, Literal,
                    Optional, Tuple, Type, Union, get_args)

from chatboard.text.llms.context import Context
from chatboard.text.llms.conversation import (AIMessage, BaseMessage,
                                              HumanMessage, SystemMessage,
                                              validate_msgs)
from chatboard.text.llms.function_utils import call_function
from chatboard.text.llms.llm2 import AzureOpenAiLLM, OpenAiLLM
from chatboard.text.llms.model_schema_prompt_parser import \
    ModelSchemaPromptParser
from chatboard.text.llms.mvc3 import (ViewNode, create_view_node, render_view,
                                      transform_list_to_view_node)
from chatboard.text.llms.tracer import Tracer
from pydantic import BaseModel, Field


def render_base_model_schema(base_model: BaseModel) -> str:
    return json.dumps(base_model.model_json_schema(), indent=4) + "\n"


def render_output_model(output_model: Type[BaseModel]) -> str:
    prompt = ''
    for field, info in output_model.model_fields.items():
        prompt += "\t"
        args = get_args(info.annotation)
        if not args:
            prompt += f"{field}: ({info.annotation.__name__}) {info.description}"
        else:
            prompt += f"{field}: (str) {info.description}"
            prompt += ' can be one of the following: '
            for arg in args:
                prompt += arg.__name__ + ', '                
        prompt += '\n'
    return prompt


    


async def render_view_arg(arg: Any, title: str, **kwargs) -> str:
    # prompt = title + ":\n" if title else ''
    prompt = ''
    if inspect.isfunction(arg):
        arg = await call_function(arg, **kwargs)
        if not arg:
            return ''
    if isinstance(arg, str) or isinstance(arg, list):
        arg = create_view_node(arg, title, title=title)
        
    
    # if isinstance(arg, str):
    #     return prompt + arg
    # elif isinstance(arg, list):
    #     render_prompt, _, _ = render_view(arg, **kwargs)
    #     return prompt + render_prompt
    # elif isinstance(arg, tuple):
    #     render_prompt, _, _ = render_view(arg, **kwargs)
    #     return prompt + render_prompt
    # elif isinstance(arg, ViewNode):
    #     render_prompt, _, _ = render_view(arg, **kwargs)
    #     return prompt + render_prompt
    if isinstance(arg, ViewNode):
        render_prompt, _, _ = render_view(arg, **kwargs)
        return render_prompt
    elif isinstance(arg, BaseModel):
        return prompt + json.dumps(arg.model_dump(), indent=2)
    else:
        raise ValueError("Invalid view arg")



class ChatPrompt(BaseModel):
    name: str | None = None
    model: str= "gpt-3.5-turbo-0125"
    llm: Union[OpenAiLLM, AzureOpenAiLLM] = Field(default_factory=OpenAiLLM)
    
    system_prompt: Optional[str] = None
    background: Optional[str] = None
    task: Optional[str] = None
    rules: Optional[str | List[str] | Callable] = None 
    examples: Optional[str | List[str] | Callable] = None 
    actions: Optional[List[Type[BaseModel]]] = []
    response_model: Optional[Type[BaseModel]] = None
    
    
    tool_choice: Literal['auto', 'required', 'none'] | BaseModel | None = None
    
    
    is_traceable: bool = True 
    
    _render_method: Optional[Callable] = None
    _output_parser_method: Optional[Callable] = None

    async def render_system_message(
        self, 
        base_models: Dict[str, BaseModel], 
        context: Context | None = None,
        response_model: Type[BaseModel] | None = None,
        actions: List[Type[BaseModel]] | None = None,
        **kwargs
        ) -> SystemMessage:
        
        if self.system_prompt:
            if type(self.system_prompt) == str:
                return SystemMessage(content=self.system_prompt)        
        
        system_prompt = ""
        
        if self.background:
            system_prompt += f"{self.background}\n"
        if self.task:
            system_prompt+= "Task:\n"
            system_prompt += f"{self.task}\n"        
                        
        for _, model in base_models.items():
            system_prompt += render_base_model_schema(model)            
            
        if actions:
            system_prompt += "\nyou should use one of the following actions:\n"
            system_prompt += "\n".join([render_base_model_schema(action) for action in actions]) + "\n"
        
        if response_model:
            system_prompt += "\nResponse format\n"
            system_prompt += f"you should return a response in the following format:\n"
            system_prompt += render_output_model(response_model)
        
        
        if self.rules is not None:            
            system_prompt += await render_view_arg(
                self.rules,
                title="Rules",
                context=context, 
                **kwargs
            )
        if self.examples is not None:
            system_prompt += await render_view_arg(
                # create_view_node(self.rules, "examples", title="Examples"),
                self.examples,
                title="Examples",
                context=context, 
                **kwargs
            )
        # if self.rules is not None:
        #     if inspect.isfunction(self.rules):  
        #         rules = await call_function(self.rules, context=context, **kwargs)
        #     else:
        #         rules = self.rules
        #     system_prompt += "\nRules:\n"                                      
        #     if isinstance(rules, list):
        #         system_prompt += "\n".join(rules) + "\n"
        #     elif isinstance(rules, str):
        #         system_prompt += rules + "\n"    
                
        # if self.examples:
        #     if inspect.isfunction(self.examples):
        #         examples = await call_function(self.examples, context=context, **kwargs)
        #     else:
        #         examples = self.examples
        #     if examples:
        #         system_prompt += "\nExamples:\n"
        #         if isinstance(examples, list):
        #             system_prompt += "\n".join(examples) + "\n"
        #         elif isinstance(examples, str):
        #             system_prompt += examples + "\n"
        #         elif isinstance(examples, ViewNode):
        #             prompt, _, _ = render_view(examples, **kwargs)
        #             system_prompt += prompt
        #         else:
        #             raise ValueError("Invalid examples format")
            
            
    
        return SystemMessage(content=system_prompt)
    
    
    def set_methods(self, render_func: Callable | None = None, output_parser: Callable | None = None) -> None:
        self._render_method = render_func
        self._output_parser_method = output_parser
        
    
    async def render(self, **kwargs: Any) -> List[ViewNode] | ViewNode:
        raise NotImplementedError("render method is not set")
    
    async def output_parser(self, response_message: AIMessage, **kwargs: Any) -> Dict[str, Any]:
        return response_message
    
    async def _render(self, context=None, **kwargs: Any) -> List[ViewNode] | ViewNode:
        views = await call_function(
                self.render if self._render_method is None else self._render_method, 
                context=context, 
                **kwargs
            )
        if isinstance(views, list):
            return views
        elif isinstance(views, tuple):
            return ViewNode(
                name=self.name or self.__class__.__name__,
                views=views,
                role='user',
            ) 
        return views
    
    
    async def _output_parser(self, response_message: AIMessage | None, **kwargs: Any) -> Dict[str, Any]:
        return await call_function(
            self.output_parser if self._output_parser_method is None else self._output_parser_method, 
            response_message,
            **kwargs
        )
    
    async def _build_conversation(
            self, 
            views: List[ViewNode] | ViewNode, 
            context: Context | None = None,
            response_model: Type[BaseModel] | None = None, 
            actions: List[Type[BaseModel]] | None= None,             
            **kwargs
        ) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:        
        if not isinstance(views, list):
            views = [views]

        total_base_models = {}

        messages = []

        for view in views:
            if isinstance(view, BaseMessage):
                messages.append(view)
                # if view.is_valid():
                #     messages.append(view)
                continue
            prompt, rendered_outputs, base_models = render_view(view, **kwargs)
            if isinstance(view, ViewNode):
                messages.append(AIMessage(content=prompt) if view.role == 'assistant' else HumanMessage(content=prompt))
            else:
                messages.append(HumanMessage(content=prompt))     
            
            total_base_models.update(base_models)
        
        # messages = validate_msgs(messages)
            
        system_message = await self.render_system_message(
            total_base_models, 
            context=context,
            response_model=response_model,
            actions=actions,            
            **kwargs)
        if system_message.content:
            messages = [system_message] + messages
        return messages

    
    async def __call__(
            self,
            views: List[ViewNode] | ViewNode | None = None, 
            context: Context | None=None, 
            response_model = None,
            actions: List[Type[BaseModel]] | None = None,
            tool_choice: Literal['auto', 'required', 'none'] | BaseModel | None = None,
            tracer_run: Tracer | None=None,
            output_messages: bool = False,
            **kwargs: Any
        ) -> AIMessage | List[BaseMessage] | str:
        
        with Tracer(
                is_traceable=self.is_traceable,
                tracer_run=tracer_run,
                name=self.name or self.__class__.__name__,
                run_type="prompt",
                inputs={
                    "input": kwargs,
                },
            ) as prompt_run:
            
            
            
            try:
            
                response_model = response_model or self.response_model
                actions = actions or self.actions
                
                if response_model and actions:
                    raise ValueError("response_model and actions cannot be used together")
                
                # views = await call_function(
                #     self._render if self._render_method is None else self._render_method, 
                #     context=context, 
                #     **kwargs
                # )
                views = views or await self._render(context=context, **kwargs)
                
                messages = await self._build_conversation(
                    views, 
                    context=context,
                    actions=actions,
                    response_model=response_model,                
                    **kwargs
                )
                
                if output_messages:
                    return messages
                
                response_message = await self.llm.complete(
                    msgs=messages,
                    tools=actions,
                    response_model=response_model,
                    tool_choice=tool_choice or self.tool_choice,
                    tracer_run=prompt_run, 
                )
                response_message = await self._output_parser(response_message, **kwargs)
                prompt_run.end(outputs={'output': response_message})
                
                return response_message
            except Exception as e:
                prompt_run.end(errors=str(e))
                raise e
        
        
        
        
        
        
        
        

def prompt(
    model: str = "gpt-3.5-turbo-0125",
    llm: Union[OpenAiLLM, AzureOpenAiLLM, None] = None,
    system_prompt: Optional[str] = None,
    background: Optional[str] = None,
    task: Optional[str] = None,
    rules: Optional[str | List[str] | Callable] = None,
    examples: Optional[str | List[str] | Callable] = None,
    actions: Optional[List[Type[BaseModel]]] = None,
    response_model: Optional[Type[BaseModel]] = None,
    parallel_actions: bool = True,
    is_traceable: bool = True,
    output_parser: Optional[Callable] = None,
    tool_choice: Literal['auto', 'required', 'none'] | BaseModel | None = None,    
):
    if llm is None:
        llm = OpenAiLLM(
            model=model, 
            parallel_tool_calls=parallel_actions
        )
    def decorator(func) -> Callable[..., Awaitable[AIMessage]]:
        
        @wraps(func)
        async def wrapper(**kwargs) -> AIMessage:
            prompt = ChatPrompt(
                name=func.__name__,
                model=model,
                llm=llm,
                system_prompt=system_prompt,
                background=background,
                task=task,
                rules=rules,
                examples=examples,
                actions=actions,
                response_model=response_model,
                tool_choice=tool_choice,
            )
            prompt.set_methods(func, output_parser)
            return await prompt(**kwargs)

        return wrapper
    
    return decorator





async def map_prompt(
    prompt: ChatPrompt | Callable,
    kwargs_list: list[Dict[str, Any]] | Generator[Dict[str, Any], Any, Any],
    metadata: dict = {},
    parallelism: int = 10,
    is_traceable: bool = True,
    name: str | None = None,
    verbose=False,   
    logger=None,
    output_run_id: bool = False,
) -> Tuple[List[Any], str] | List[Any]:
    if not name:
        if isinstance(prompt, ChatPrompt):
            name = prompt.name
        else:
            name = prompt.__name__
    name = f"{name}_map"
    with Tracer(
        is_traceable=is_traceable,
        name=name,
        run_type="chain",
        inputs={},
        extra=metadata,
        tags=["map"]
    ) as tracer_run:
        semaphore = asyncio.Semaphore(parallelism)
        async def run_prompt(index, **kwargs):
            async with semaphore:
                # output = await prompt(tracer_run=tracer_run, index=index, **kwargs)
                try:
                    output = await call_function(prompt, tracer_run=tracer_run, index=index, **kwargs)
                    if logger and verbose:
                        logger.info(f"Completed item {index}.")
                    return output
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing item {index}: {e}")
                    return None
        
        tasks = [run_prompt(index=index, **kwargs) for index, kwargs in enumerate(kwargs_list)]
        results = await asyncio.gather(*tasks)
        tracer_run.end(outputs={'map_length': len(results)})
    if output_run_id:
        return results, str(tracer_run.id)
    else:
        return results


