
from functools import wraps
from typing import AsyncGenerator, ParamSpec, Type
from ..prompt.stream import STREAM_EVENT, STREAM_RESPONSE, StreamController
import inspect
from typing import (Any, Callable, Dict, Generic, TypeVar, ParamSpec)

from ..context import ExecutionContext



from ..block import BlockChunk
from .depends import  DependsContainer, resolve_dependency
from .context import Context
from ..utils.function_utils import filter_args_by_exclude




P = ParamSpec('P')

class PromptStream(StreamController[P, STREAM_EVENT, STREAM_RESPONSE]):
    
    
    def _filter_args_for_trace(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        from ..llms import LLM
        _args, _kwargs = filter_args_by_exclude(args, kwargs, (LLM, Context))
        return {"args": _args, "kwargs": _kwargs}
    
    
    def _sanitize_output(self, output: Any) -> Any:
        if isinstance(output, BlockChunk):
            return output.content
        return output
    
    def build_execution_ctx(self) -> Context:
        curr_ctx: Context | None = Context.get_current(False)
        if curr_ctx is not None:
            ctx = curr_ctx.build_child(self._name)
        else:
            # raise ValueError("Context is not set")
            # ctx = Context().start()
            # ctx = Context()
            ctx = ExecutionContext(self._name)
        return ctx


    async def _inject_dependencies(self, *args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
        signature = inspect.signature(self.run)
        injection_kwargs = {}
        for param_name, param in signature.parameters.items():
            default_val = param.default
            if isinstance(default_val, DependsContainer):
                dependency_func = default_val.dependency
                resolved_val = await resolve_dependency(dependency_func,  *args, **kwargs)
                injection_kwargs[param_name] = resolved_val            
                
        return injection_kwargs

    async def start(self):
        kwargs = self._kwargs.copy()
        injection_kwargs = await self._inject_dependencies(*self._args, **kwargs) 
        kwargs.update(injection_kwargs)                      
        self._agen = self.run(*self._args, **kwargs)

    
    
    # async def stream(self) -> AsyncGenerator[STREAM_EVENT, None]:        
    #     await self.init_generator()       
    #     response = None
    #     try:
    #         while event:= await self.send(response):
    #             try:
    #                 response = None
    #                 if isinstance(event, StreamController):                    
    #                     sub_gen = event
    #                     async for sub_event in sub_gen.stream():
    #                         if sub_gen.is_response_type_match(sub_event):
    #                             # if response is not None:
    #                                 # raise ValueError("Multiple responses found.\n", "current response:\n", response, "new response:\n", sub_event)
    #                             response = sub_event
    #                         yield sub_event                    
                    
    #                 else:
    #                     yield event                              
    #             except Exception as e:
    #                 await self.throw(e)
    #     except StopAsyncIteration:
    #         return      
    #     finally:
    #         await self.close()




def prompt(
    response_type: Type[STREAM_RESPONSE] | None = None,
    ) -> Callable[
    [Callable[P, AsyncGenerator[STREAM_EVENT, None]]],
    Callable[P, StreamController[P, STREAM_EVENT, STREAM_RESPONSE]]
]:    
    def decorator(func: Callable[P, AsyncGenerator[STREAM_EVENT, None]]) -> Callable[P, StreamController[P, STREAM_EVENT, STREAM_RESPONSE]]:        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamController[P, STREAM_EVENT, STREAM_RESPONSE]:
            stream_controller = PromptStream(
                func.__name__, 
                response_type, 
                *args, 
                **kwargs
            ) 
            stream_controller.run = func
            return stream_controller
        return wrapper    
    return decorator