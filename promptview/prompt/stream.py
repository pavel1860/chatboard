from functools import wraps
import inspect
from typing import (Any, AsyncGenerator, Callable, Dict, Generic, Type, TypeVar, ParamSpec)

from ..prompt.events import StreamEvent





    


STREAM_EVENT = TypeVar("STREAM_EVENT")
STREAM_RESPONSE = TypeVar("STREAM_RESPONSE")
P = ParamSpec("P")

class StreamController(Generic[P, STREAM_EVENT, STREAM_RESPONSE]):
    """
    Base class to control an async generator stream:
    - automatically exhausts it if desired
    - allows .send() and .throw() into it
    - override `run()` to implement your logic
    """
    
    _name: str
    _did_prime: bool = False
    _args: Any = ()
    _kwargs: Any = {}
    
    def __init__(
        self,                 
        # args: tuple,
        # kwargs: dict,
        name: str,
        response_type: Type[STREAM_RESPONSE] | None = None,
        *args: P.args,                 
        **kwargs: P.kwargs,        
    ):
        self._args = args
        self._kwargs = kwargs
        self._name = name
        self._agen: AsyncGenerator[STREAM_EVENT, None] | None = None
        self._closed = False
        self.response_type = response_type
        self._did_stop = False
        self.index = 0
        
    @property
    def agen(self) -> AsyncGenerator[STREAM_EVENT, None]:
        if self._agen is None:
            raise ValueError("Generator not started")
        return self._agen
    
    def get_and_increment_index(self) -> int:
        index = self.index
        self.index += 1
        return index

    async def start(self):
        """Starts the generator by calling the user-defined async generator."""
        self._agen = self.run(*self._args, **self._kwargs)
        # await self.agen.asend(None)  # prime it

    async def next(self):
        """Get next value from generator."""
        return await self.agen.__anext__()

    async def send(self, value):
        """Send value into the generator."""
        return await self.agen.asend(value)

    async def throw(self, exc):
        """Throw exception into the generator."""
        return await self.agen.athrow(exc)
    
    def _stop(self):
        self._is_streaming = False

    async def close(self):
        """Close the generator."""
        if self.agen and not self._closed:
            await self.agen.aclose()
            self._closed = True
            

    async def exhaust(self):
        """Iterate over all values until exhausted."""
        results = []
        async for item in self.agen:
            results.append(item)
        return results

    async def __aenter__(self):
        await self.start()
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[STREAM_EVENT, None]:
        """
        Override this method in your subclass to implement your custom logic.
        This must be an async generator.
        """
        raise NotImplementedError("You must override the `run()` method.")
        yield

    def is_response_type_match(self, event: STREAM_EVENT) -> bool:
        if self.response_type is None:
            return False
        return isinstance(event, self.response_type)  
    

    
    def emit_event(self, payload: STREAM_EVENT | StreamEvent, index: int) -> StreamEvent:        
        if index == 0:           
            event = StreamEvent(type="stream_start", payload=payload, index=index)
        elif self._did_stop:
            event = StreamEvent(type="stream_end", payload=payload, index=index)
        else:
            event = StreamEvent(type="message_delta", payload=payload, index=index)
        return event
    
    
    # async def stream(self) -> AsyncGenerator[STREAM_EVENT, None]:
    
    async def stream(self) -> AsyncGenerator[StreamEvent, None]:        
        await self.start()        
        response = None
        prev_event = None
        try:
            while event:= await self.send(response):
                try:
                    response = None
                    if isinstance(event, StreamController):                    
                        sub_gen = event
                        async for sub_event in sub_gen.stream():
                            if sub_gen.is_response_type_match(sub_event):
                                # if response is not None:
                                    # raise ValueError("Multiple responses found.\n", "current response:\n", response, "new response:\n", sub_event)
                                response = sub_event
                            yield sub_event                    
                    
                    else:  
                        if prev_event is not None:                                                                 
                            yield self.emit_event(prev_event, self.get_and_increment_index())
                        prev_event = event
                                           
                except Exception as e:
                    await self.throw(e)
        except StopAsyncIteration:            
            self._did_stop = True
            yield self.emit_event(event, self.get_and_increment_index())          
        finally:
            await self.close()

        
        
#     @classmethod
#     def build_decorator(cls):
#         def decorator() -> Callable[
#             [Callable[P, AsyncGenerator[STREAM_EVENT, None]]],
#             Callable[P, StreamController[STREAM_EVENT, P]]
#         ]:    
#             def decorator_inner(func: Callable[P, AsyncGenerator[STREAM_EVENT, None]]) -> Callable[P, StreamController[STREAM_EVENT, P]]:        
#                 @wraps(func)
#                 def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamController[STREAM_EVENT, P]:
#                     stream_controller = cls(*args, **kwargs)
#                     stream_controller._name = func.__name__
#                     stream_controller.run = func
#                     return stream_controller
#                 return wrapper    
#             return decorator_inner
#         return decorator
            
            
            
# stream = StreamController.build_decorator()           
            
            
def stream(
    response_type: Type[STREAM_RESPONSE] | None = None,
    ) -> Callable[
    [Callable[P, AsyncGenerator[STREAM_EVENT, None]]],
    Callable[P, StreamController[P, STREAM_EVENT, STREAM_RESPONSE]]
]:    
    def decorator(func: Callable[P, AsyncGenerator[STREAM_EVENT, None]]) -> Callable[P, StreamController[P, STREAM_EVENT, STREAM_RESPONSE]]:        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamController[P, STREAM_EVENT, STREAM_RESPONSE]:
            stream_controller = StreamController(
                func.__name__, 
                response_type, 
                *args, 
                **kwargs
            ) 
            stream_controller.run = func
            return stream_controller
        return wrapper    
    return decorator