
import inspect


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


async def call_function(func, *args, **kwargs):
    func_args = get_func_args(func)
    if 'kwargs' not in func_args:
        kwargs = filter_func_args(func, kwargs)
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)

