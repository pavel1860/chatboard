from functools import wraps
import inspect
from typing import Awaitable, Callable, Generic, ParamSpec
from typing_extensions import TypeVar


def is_async_function(obj):
    return inspect.iscoroutinefunction(obj) or inspect.isasyncgenfunction(obj)

def get_func_args(func):
    return list(inspect.signature(func).parameters.keys())

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def filter_func_args(func, args):
    return {k: v for k, v in args.items() if k in get_func_args(func)}


# async def call_function(func, *args, **kwargs):
#     func_args = get_func_args(func)
#     if 'kwargs' not in func_args:
#         kwargs = filter_func_args(func, kwargs)
#     if inspect.iscoroutinefunction(func):
#         return await func(*args, **kwargs)
#     return func(*args, **kwargs)

async def call_function(func, *args, **kwargs):
    sig = inspect.signature(func)
    func_params = list(sig.parameters.values())
    
    # Filter positional args to match the number of non-keyword-only parameters
    positional_params = [p for p in func_params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    filtered_args = args[:len(positional_params)]
    
    # Filter kwargs if there's no **kwargs parameter
    if not any(p.kind == p.VAR_KEYWORD for p in func_params):
        kwargs = filter_func_args(func, kwargs)
    
    if inspect.iscoroutinefunction(func):
        return await func(*filtered_args, **kwargs)
    return func(*filtered_args, **kwargs)



def filter_args_by_exclude(args: tuple, kwargs: dict, exclude_classes: tuple[type, ...]) -> tuple[tuple, dict]:
    """
    Filter args and kwargs by excluding values of specified classes.
    
    Args:
        args: Tuple of positional arguments
        kwargs: Dictionary of keyword arguments
        exclude_classes: Tuple of classes to exclude
        
    Returns:
        Tuple of (filtered_args, filtered_kwargs)
    """
    filtered_args = tuple(arg for arg in args if not isinstance(arg, exclude_classes))
    filtered_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, exclude_classes)}
    return filtered_args, filtered_kwargs







P = ParamSpec("P")
R = TypeVar("R")

class ContextFuture(Generic[R]):
    """
    A context manager for an awaitable that returns a value
    can be used in an async with statement or as regular awaitable
    """
    def __init__(self, future: Awaitable[R]):
        self.future = future
        self._output = None
        self._has_context = False
                
    def __await__(self):
        return self.future.__await__()
        
    async def __aenter__(self) -> R:
        self._output = await self.future
        if hasattr(self._output, "__aenter__"):
            self._has_context = True
            return await self._output.__aenter__()
        elif hasattr(self._output, "__enter__"):
            self._has_context = True
            return self._output.__enter__()
        return self._output
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._has_context:
            if hasattr(self._output, "__aexit__"):
                return await self._output.__aexit__(exc_type, exc_value, traceback)
            elif hasattr(self._output, "__exit__"):
                return self._output.__exit__(exc_type, exc_value, traceback)
            else:
                raise ValueError(f"No exit method found for {self._output}")
        return False


def contextcallable(func: Callable[P, Awaitable[R]]) -> Callable[P, ContextFuture[R]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ContextFuture[R]:
        return ContextFuture(func(*args, **kwargs))
    return wrapper








def is_overridden(cls, method_name, base_cls):
    return getattr(cls, method_name) is not getattr(base_cls, method_name)


def get_if_overridden(cls, method_name: str, base_cls: type) -> Callable | None:
    method = getattr(cls, method_name)
    if method is not getattr(base_cls, method_name):
        return method
    return None