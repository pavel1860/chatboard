
from functools import wraps
import json
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from chatboard.text.llms.context import Context
from chatboard.text.llms.conversation import AIMessage, HumanMessage, SystemMessage
from chatboard.text.llms.function_utils import call_function
from chatboard.text.llms.llm2 import AzureOpenAiLLM, OpenAiLLM
from chatboard.text.llms.mvc2 import ViewNode, render_view
from chatboard.text.llms.tracer import Tracer



def render_base_model_schema(base_model: BaseModel) -> str:
    return json.dumps(base_model.model_json_schema(), indent=4) + "\n"



class ChatPrompt(BaseModel):
    name: str = None
    model: str= "gpt-3.5-turbo-0125"
    llm: Union[OpenAiLLM, AzureOpenAiLLM] = Field(default_factory=OpenAiLLM)
    
    background: Optional[str] = None
    task: Optional[str] = None
    rules: Optional[str | List[str]] = None 
    actions: Optional[List[BaseModel]] = []
    
    is_traceable: bool = True 
    
    _render_method: Optional[Callable] = None
  
    async def render_system_message(self, base_models: Dict[str, BaseModel]) -> SystemMessage:
        system_prompt = ""
        
        if self.background:
            system_prompt += f"{self.background}\n"
        if self.task:
            system_prompt+= "Task:\n"
            system_prompt += f"{self.task}\n"        
                        
        for _, model in base_models.items():
            system_prompt += render_base_model_schema(model)            
            
        if self.actions:
            system_prompt += "\nyou should use one of the following actions:\n"
            system_prompt += "\n".join([render_base_model_schema(action) for action in self.actions]) + "\n"
        
        if self.rules:  
            rules = self.rules
            system_prompt += "\nRules:\n"                                      
            if isinstance(rules, list):
                system_prompt += "\n".join(rules) + "\n"
            elif isinstance(rules, str):
                system_prompt += rules + "\n"    
    
        return SystemMessage(content=system_prompt)
    
    
    def set_render_method(self, render_func: Callable) -> None:
        self._render_method = render_func
    
    async def render(self, **kwargs: Any) -> List[ViewNode] | ViewNode:
        raise NotImplementedError("render method is not set")
    
    
    async def _build_conversation(self, views: List[ViewNode] | ViewNode, **kwargs) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:        
        if not isinstance(views, list):
            views = [views]

        total_base_models = {}

        messages = []

        for view in views:
            prompt, rendered_outputs, base_models = render_view(view, **kwargs)
            messages.append(AIMessage(content=prompt) if view.role == 'assistant' else HumanMessage(content=prompt))
            total_base_models.update(base_models)
            
        system_message = await self.render_system_message(total_base_models)
        messages = [system_message] + messages
        return messages

    
    async def __call__(
            self,
            views: List[ViewNode] | ViewNode | None = None, 
            context: Context | None=None, 
            response_model = None,
            tracer_run: Tracer=None,
            **kwargs: Any
        ) -> Any:
        
        with Tracer(
                is_traceable=self.is_traceable,
                tracer_run=tracer_run,
                name=self.name or self.__class__.__name__,
                run_type="prompt",
                inputs={
                    "input": kwargs,
                },
            ) as prompt_run:
            
            views = await call_function(
                self.render if self._render_method is None else self._render_method, 
                context=context, 
                **kwargs
            )
            
            messages = await self._build_conversation(views, **kwargs)
            
            response_message = await self.llm.complete(
                msgs=messages,
                tools=list(self.actions.values()) if self.actions and not response_model else None,
                tracer_run=prompt_run, 
            )            
            
            prompt_run.end(outputs={'output': response_message})
            
            return response_message
        
        
        
        
        
        
        
        

def prompt(
    model: str = "gpt-3.5-turbo-0125",
    llm: Union[OpenAiLLM, AzureOpenAiLLM, None] = None,
    background: Optional[str] = None,
    task: Optional[str] = None,
    rules: Optional[str | List[str]] = None,
    actions: Optional[List[BaseModel]] = None,
    is_traceable: bool = True
):
    if llm is None:
        llm = OpenAiLLM(model=model)
    
    def decorator(func):
        
        @wraps(func)
        async def wrapper(**kwargs):             
            prompt = ChatPrompt(
                name=func.__name__,
                model=model,
                llm=llm,
                background=background,
                task=task,
                rules=rules,
                actions=actions
            )
            prompt.set_render_method(func)            
            return await prompt(**kwargs)

        return wrapper
    
    return decorator
