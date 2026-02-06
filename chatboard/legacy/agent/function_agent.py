import inspect
from functools import wraps
from typing import (Any, AsyncGenerator, Awaitable, Callable, Generic, List, Literal, ParamSpec, Type,
                    TypedDict, TypeVar)

from chatboard.llms.legacy.anthropic_llm import AnthropicLLM
from chatboard.llms.legacy.llm2 import LLM
from chatboard.llms.messages import AIMessage, BaseMessage, HumanMessage
from chatboard.llms.legacy.openai_llm import OpenAiLLM
from chatboard.llms.tracer import Tracer
from chatboard.llms.utils.action_manager import Actions
from chatboard.prompt.legacy.mvc import ViewBlock, create_view_block
from chatboard.state.context import Context
from chatboard.utils.function_utils import call_function, filter_func_args
from pydantic import BaseModel, Field


P = ParamSpec("P")
R = TypeVar('R')

ToolChoiceParam = Literal['auto', 'required', 'none'] | BaseModel | None




def agent(          
        is_traceable: bool = True,
    ):
    def decorator(func) -> Callable[P, Awaitable[R]]:            
        async def wrapper(context: Context | None=None, tracer_run: Tracer | None=None,  *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[R, None]:
            with Tracer(
                name=func.__name__,
                is_traceable=is_traceable,
                inputs=kwargs,
                session_id=context.session_id,
                tracer_run=tracer_run
            ) as run:
                if inspect.isasyncgenfunction(func):
                    gen_kwargs = filter_func_args(func, {"context": context, "tracer_run": run} | kwargs)
                    outputs = []
                    async for msg in func(**gen_kwargs):
                        outputs.append(msg)
                        yield msg
                    run.add_outputs({"response":outputs})                            
                else:
                    raise ValueError("Agent function must be an async generator function")
        return wrapper
    
    return decorator