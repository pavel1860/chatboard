
from typing import Awaitable, Callable, ParamSpec, TypeVar
from fastapi import FastAPI


P = ParamSpec('P')
R = TypeVar('R')


def chat(app: FastAPI):
        
    def decorator(func: Callable[P, Awaitable[R]]):
        @app.post("/chat")
        async def chat_endpoint(*args: P.args, **kwargs: P.kwargs):
            return await func(*args, **kwargs)
        return chat_endpoint
    return decorator
