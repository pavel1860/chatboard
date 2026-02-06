import inspect
from typing import (Any, Awaitable, Callable, Concatenate, Generic, List, Literal, Type,
                    TypedDict, TypeVar, ParamSpec)

from ..tracer import Tracer
from .controller import Controller
from ..utils.function_utils import call_function


P = ParamSpec('P')
R = TypeVar('R')


# class FunctionType(enum.Enum):
#     FUNCTION = 0
#     ASYNC_FUNCTION = 1
#     GENERATOR = 2
#     ASYNC_GENERATOR = 3
    


# def check_function_type(func):    
#     if inspect.isfunction(func):
#         return FunctionType.FUNCTION
#     elif inspect.iscoroutinefunction(func):
#         return FunctionType.ASYNC_FUNCTION
#     elif inspect.isgeneratorfunction(func):
#         return FunctionType.GENERATOR
#     elif inspect.isasyncgenfunction(func):
#         return FunctionType.ASYNC_GENERATOR
        



# class FunctionDescription:
#     """
#     container for function type and parameters
#     """
#     def __init__(self, func) -> None:
#         self.type = check_function_type(func)
#         self.params = [a for a in inspect.signature(func).parameters.values() if a.name != 'self']

#     def merge_args_kwargs(self, args, kwargs):
#         log_args = {}
#         for i, arg in enumerate(args):
#             log_args[self.params[i].name] = arg
#         log_args.update(kwargs)
#         return log_args
    
#     # def filter_args(self, args, kwargs):
        
        # return log_args, log_kwargs




    
    
    
    



class Prompt(Controller[P, R]):
    
    def __init__(self, tracable: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.tracable = tracable
    
    
    async def _call_with_dependencies(self, *args, **kwargs):
        injection_kwargs = await self._inject_dependencies(*args, **kwargs)                      
        res = await call_function(self._complete, *args, **kwargs, **injection_kwargs)
        return res
    
    
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        execution_ctx = self.build_execution_ctx()
        async with execution_ctx.start_tracer(self._name, "prompt", inputs={}) as ctx:
            inspect.signature(self._complete).bind(*args, **kwargs)
            injection_kwargs = await self._inject_dependencies(*args, **kwargs)               
            kwargs.update(injection_kwargs)
            try:
                res = await call_function(self._complete, *args, **kwargs)
                ctx.tracer.add_outputs({"response": self._sanitize_output(res)})
                return res
            except Exception as e:
                ctx.tracer.end(errors=str(e))
                raise e
        


        
def prompt(
    tracable: bool = False,
    **kwargs: Any
)-> Callable[[Callable[P, Awaitable[R]]], Prompt[P, R]]:
    
    def decorator(func: Callable[P, Awaitable[R]]) -> Prompt[P,R]:
        prompt = Prompt(
                # model=model, #type: ignore 
                tracable=tracable,               
                **kwargs
            )
        prompt._name=func.__name__
        prompt._complete = func
        return prompt        
    return decorator

