from functools import wraps
from typing import (Any, Awaitable, Callable, List, Literal, Optional, Tuple,
                    Type, TypeVar)

from chatboard.llms import OpenAiLLM
from chatboard.llms.legacy.llm import LLM
from chatboard.llms.messages import AIMessage
from chatboard.prompt.legacy.chat_prompt import ChatPrompt
from pydantic import BaseModel, Field

# T = TypeVar('T')

# def prompt( 
#     self=None,   
#     model: str = "gpt-3.5-turbo-0125",
#     llm: LLM | None = None,
#     system_prompt: Optional[str] = None,
#     background: Optional[str] = None,
#     task: Optional[str] = None,
#     rules: Optional[str | List[str] | Callable] = None,
#     examples: Optional[str | List[str] | Callable] = None,
#     actions: Optional[List[Type[BaseModel]]] = None,
#     response_model: Optional[Type[BaseModel]] = None,
#     parallel_actions: bool = True,
#     is_traceable: bool = True,
#     output_parser: Callable[[AIMessage], T] | None = None,
#     tool_choice: Literal['auto', 'required', 'none'] | BaseModel | None = None,
# ):
#     if llm is None:
#         llm = OpenAiLLM(
#             model=model, 
#             parallel_tool_calls=parallel_actions
#         )
#     def decorator(func) -> Callable[..., Awaitable[T]]:
#         prompt = ChatPrompt[T](
#                 name=func.__name__,
#                 model=model,
#                 llm=llm,
#                 system_prompt=system_prompt,
#                 background=background,
#                 task=task,
#                 rules=rules,
#                 examples=examples,
#                 actions=actions,
#                 response_model=response_model,
#                 tool_choice=tool_choice,
#             )
#         prompt.set_methods(func, output_parser)
#         if self:
#             self.router_prompt = prompt
#         @wraps(func)
#         async def wrapper(**kwargs) -> T:            
#             return await prompt(**kwargs)

#         return wrapper
    
#     return decorator

from chatboard.prompt.legacy.chat_prompt import ChatPrompt

prompt = ChatPrompt.decorator_factory()