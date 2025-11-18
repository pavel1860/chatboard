



import inspect
from typing import Annotated, Any, Callable, Generic, Optional, ParamSpec, TypeVar
from typing_extensions import Doc

# from aiohttp_retry import Any, Optional
from ..utils.function_utils import call_function


# P = ParamSpec('P')
# R = TypeVar('R')

# class Depends(Generic[P, R]):
#     def __init__(self, dependency: Callable[P, R]):
#         self.dependency = dependency

# class Depends:
#     def __init__(
#         self, dependency: Optional[Callable[..., Any]] = None, *, use_cache: bool = True
#     ):
#         self.dependency = dependency
#         self.use_cache = use_cache

#     def __repr__(self) -> str:
#         attr = getattr(self.dependency, "__name__", type(self.dependency).__name__)
#         cache = "" if self.use_cache else ", use_cache=False"
#         return f"{self.__class__.__name__}({attr}{cache})"

class DependencyInjectionError(Exception):
    pass


class DependsContainer:
    def __init__(
        self, dependency: Optional[Callable[..., Any]] = None, *, use_cache: bool = True
    ):
        self.dependency = dependency
        self.use_cache = use_cache

    def __repr__(self) -> str:
        attr = getattr(self.dependency, "__name__", type(self.dependency).__name__)
        cache = "" if self.use_cache else ", use_cache=False"
        return f"{self.__class__.__name__}({attr}{cache})"


def Depends(  # noqa: N802
    dependency: Annotated[
        Optional[Callable[..., Any]],
        Doc(
            """
            A "dependable" callable (like a function).

            Don't call it directly, Chatboard will call it for you, just pass the object
            directly.
            """
        ),
    ] = None,
    *,
    use_cache: Annotated[
        bool,
        Doc(
            """
            By default, after a dependency is called the first time in a request, if
            the dependency is declared again for the rest of the request (for example
            if the dependency is needed by several dependencies), the value will be
            re-used for the rest of the request.

            Set `use_cache` to `False` to disable this behavior and ensure the
            dependency is called again (if declared more than once) in the same request.
            """
        ),
    ] = True,
) -> Any:
    return DependsContainer(dependency=dependency, use_cache=use_cache)




async def resolve_dependency(dependency_func, *args, **kwargs):
    """
    Recursively resolve the dependencies of `dependency_func` by inspecting
    its signature. Calls sub-dependencies first, then calls `dependency_func`.
    """
    signature = inspect.signature(dependency_func)
    dep_kwargs = {}
    
    for param_name, param in signature.parameters.items():
        default_val = param.default
        
        # If the default is a Depends(...) object, recursively resolve it
        if isinstance(default_val, DependsContainer):
            sub_dependency_func = default_val.dependency
            dep_kwargs[param_name] = await resolve_dependency(sub_dependency_func)
    
    
    # Once all sub-dependencies are resolved, call the function
    try:
        # return await call_function(dependency_func, *args, **kwargs, **dep_kwargs)
        return await call_function(dependency_func, *args, **(kwargs | dep_kwargs))
    except TypeError as e:
        raise DependencyInjectionError(f"Dependency injection error for {dependency_func.__name__}:\n" + ",".join(e.args))
        # raise DependencyInjectionError(e)
    except Exception as e:
        raise e
