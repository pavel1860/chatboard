import inspect
from ..prompt.depends import DependsContainer, resolve_dependency






class InjectableFunction:
    
    def __init__(self, func, args, kwargs):
        self._func = func                
        self._args = args
        self._kwargs = kwargs
        self.__name__ = func.__name__
        
        
    async def _resolve_dependencies(self, args, kwargs):
        signature = inspect.signature(self._func)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        dep_kwargs = {}
        for name, param in signature.parameters.items():
            val = bound.arguments.get(name, param.default)
            if isinstance(val, DependsContainer):
                dep_kwargs[name] = await resolve_dependency(val.dependency)
            elif name not in bound.arguments:
                dep_kwargs[name] = val

        return inspect.BoundArguments(signature, bound.arguments | dep_kwargs)
    
    
    async def __call__(self):
        bound = await self._resolve_dependencies(self._args, self._kwargs)
        return self._func(*bound.args, **bound.kwargs)
    
    
    
    
async def resolve_dependencies(func, args, kwargs):
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    dep_kwargs = {}
    for name, param in signature.parameters.items():
        val = bound.arguments.get(name, param.default)
        if isinstance(val, DependsContainer):
            dep_kwargs[name] = await resolve_dependency(val.dependency)
        elif name not in bound.arguments:
            dep_kwargs[name] = val

    return inspect.BoundArguments(signature, bound.arguments | dep_kwargs)





async def resolve_dependencies_kwargs(func, args, kwargs):
    signature = inspect.signature(func)
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()

    dep_kwargs = {}
    param_dict = {}
    for name, param in signature.parameters.items():
        val = bound.arguments.get(name, param.default)
        if isinstance(val, DependsContainer):
            resolved = await resolve_dependency(val.dependency)
            dep_kwargs[name] = resolved
            param_dict[name] = resolved
        elif name not in bound.arguments:
            dep_kwargs[name] = val
        elif name in bound.arguments and name not in ('self', 'cls'):
            param_dict[name] = val
        elif name in kwargs:
            param_dict[name] = kwargs[name]
        # elif 
    return inspect.BoundArguments(signature, bound.arguments | dep_kwargs), param_dict

