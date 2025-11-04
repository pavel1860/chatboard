import inspect
from typing import (Any, Callable, Dict, Generic, TypeVar, ParamSpec)
from ..context import ExecutionContext


from ..block import BlockChunk
from .depends import  DependsContainer, resolve_dependency
from .context import Context
from ..utils.function_utils import filter_args_by_exclude


P = ParamSpec('P')
R = TypeVar('R')


class Controller(Generic[P, R]):
    _name: str
    _complete: Callable[P, R]
    
    def _filter_args_for_trace(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        from ..llms import LLM
        _args, _kwargs = filter_args_by_exclude(args, kwargs, (LLM, Context))
        return {"args": _args, "kwargs": _kwargs}
    
    
    def _sanitize_output(self, output: Any) -> Any:
        if isinstance(output, BlockChunk):
            return output.content
        return output
    
    def build_execution_ctx(self) -> ExecutionContext:
        curr_ctx: ExecutionContext | None = ExecutionContext.current_or_none()
        if curr_ctx is not None:
            ctx = curr_ctx.build_child(self._name)
        else:
            ctx = ExecutionContext(self._name)
        return ctx    
        

    async def _inject_dependencies(self, *args: P.args, **kwargs: P.kwargs) -> Dict[str, Any]:
        signature = inspect.signature(self._complete)
        injection_kwargs = {}
        for param_name, param in signature.parameters.items():
            default_val = param.default
            if isinstance(default_val, DependsContainer):
                dependency_func = default_val.dependency
                resolved_val = await resolve_dependency(dependency_func,  *args, **kwargs)
                injection_kwargs[param_name] = resolved_val            
                
        return injection_kwargs