from typing import Any, Callable, Optional, ParamSpec, Union, AsyncGenerator
from functools import wraps
import time
import inspect
from typing_extensions import TypeVar
from pydantic import BaseModel


from ..prompt.depends import DependsContainer, resolve_dependency
from ..prompt.events import StreamEvent
from ..prompt.stream2 import GeneratorFrame, StreamController, SupportsExtend, CHUNK
from uuid import uuid4
# Assuming AsyncStreamWrapper, DependsContainer, and resolve_dependency are available


class Span:
    def __init__(self, name: str, inputs: Any, id: str | None = None):
        self.name = name
        self.id = id or str(uuid4())
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.error = None
        self.inputs = inputs
        self.output = None
        self.events = []
        
    def start(self):
        self.start_time = time.time()
        self.events.append(StreamEvent(type="span_start", payload=self.dump_start(), index=0))
        
    def end(self, output: Any = None):
        self.end_time = time.time()
        if self.start_time is None:
            raise ValueError("Span not started")
        self.duration = self.end_time - self.start_time
        self.output = output
        self.events.append(StreamEvent(type="span_end", payload=self.dump_end(), index=0))
        
    def error(self, error: Exception):
        self.error = error
        
    def dump_start(self):
        return {
            "type": "span_start",
            "name": self.name,
            "id": self.id,
            "start_time": self.start_time,
            "inputs": self.inputs,
        }
                
    def dump_end(self):
        return {
            "type": "span_end",
            "name": self.name,
            "id": self.id,
            "end_time": self.end_time,
            "duration": self.duration,
            "error": self.error,
            "output": self.output,
        }
        

class Component(StreamController):
    def __init__(
        self,
        name: str | None = None,
        span_func: Callable[..., AsyncGenerator] | None = None,
        args: tuple = (),
        accumulator: Optional[Union[Any, Callable[[], Any]]] = None,
        kwargs: dict = {},
    ):
        super().__init__(name=name, agen=span_func or self.run, accumulator=accumulator)
        # self._name = name
        self.span_func = span_func or self.run
        # self.span_func = span_func or self.run
        self.span_args = args
        self.span_kwargs = kwargs
        self.resolved_args = None
        self.accumulator_factory = accumulator
        self._start_time = None
        self._end_time = None
        self._trace_id = id(self)
        # self._stream: Optional[StreamController] = None
        self._span = None
    
    async def run(self, *args, **kwargs):
        """
        Override this method in your subclass to implement your custom logic.
        This must be an async generator.
        """
        pass
        yield
    
    
    @property
    def span(self):
        if self._span is None:
            raise ValueError("Span not started")
        return self._span
    
    # @property
    # def stream(self):
    #     if self._stream is None:
    #         raise ValueError(f"Stream not started for span {self._name}")
    #     return self._stream

    async def _resolve_dependencies(self):
        signature = inspect.signature(self.span_func)
        bound = signature.bind_partial(*self.span_args, **self.span_kwargs)
        bound.apply_defaults()

        dep_kwargs = {}
        for name, param in signature.parameters.items():
            val = bound.arguments.get(name, param.default)
            if isinstance(val, DependsContainer):
                dep_kwargs[name] = await resolve_dependency(val.dependency)
            elif name not in bound.arguments:
                dep_kwargs[name] = val

        return inspect.BoundArguments(signature, bound.arguments | dep_kwargs)
    
    async def build_frame(self) -> GeneratorFrame:        
        name = self._name or self.span_func.__name__
        self._span = Span(name, self._trace_id, self.resolved_args)
        self._span.start()
        self.resolved_args = await self._resolve_dependencies()
        gen = self.span_func(*self.resolved_args.args, **self.resolved_args.kwargs)        
        return GeneratorFrame(self, gen, self.accumulator_factory)
    
    
    def to_event(self, ctx: GeneratorFrame, value: Any) -> StreamEvent:
        if not ctx.emitted_start:
            ctx.emitted_start = True
            return StreamEvent(type="span_start", name=self._name, payload=None, index=0)
        elif ctx.exhausted:
            return StreamEvent(type="span_end", name=self._name, payload=ctx.accumulator, index=ctx.index)
        else:
            return StreamEvent(type="span_delta", name=self._name, payload=value, index=ctx.index)


    # async def _init_stream(self):
    #     name = self._name or self.span_func.__name__
    #     self._span = Span(name, self._trace_id, self.resolved_args)
    #     self._span.start()
    #     self.resolved_args = await self._resolve_dependencies()
    #     gen = self.span_func(*self.resolved_args.args, **self.resolved_args.kwargs)        
    #     self._stream = StreamController(name=name, agen=gen, accumulator=self.accumulator_factory)
    #     self._stream.__aiter__()

    # def _start_span(self):
    #     self._start_time = time.time()
    #     print(f"[Span Start] {self.span_func.__name__} (trace_id={self._trace_id})")

    # def _end_span(self, error: Optional[Exception] = None):
    #     if self._end_time is None:
    #         self._end_time = time.time()
    #         duration = self._end_time - self._start_time
    #         if error:
    #             print(f"[Span Error] {self.span_func.__name__}: {error}")
    #         print(f"[Span End] {self.span_func.__name__} (duration={duration:.2f}s)")

    # def __aiter__(self):
    #     return self

    # async def __anext__(self):
    #     if not self._stream:
    #         await self._init_stream()
    #     try:
    #         return await self.stream.__anext__()
    #     except StopAsyncIteration:
    #         self.span.end()
    #         raise
    #     except Exception as e:
    #         self.span.error(e)
    #         raise


    # def stream_events(self):
    #     async def event_stream():
    #         if not self._stream:
    #             await self._init_stream()
    #         event = Event(type="span_start", span=self._name, payload=self.span.dump_start(), index=0)
    #         yield event
    #         async for event in self.stream.stream_events():
    #             yield event
    #         self.span.end(self.stream.current.accumulator)
    #         event = Event(type="span_end", span=self._name, payload=self.span.dump_end(), index=1)
    #         yield event
    #     return event_stream()
    
    

    # def __await__(self):
    #     async def await_result():
    #         if not self.stream:
    #             await self._init_stream()
    #         try:
    #             result = await self.stream
    #             self.span.end(result)
    #             return result
    #         except Exception as e:
    #             self.span.error(e)
    #             raise
    #     return await_result().__await__()
    @classmethod
    def decorator_factory(cls):
        def component(
            accumulator: Optional[Union[Any, Callable[[], Any]]] = None
        ) -> Callable[[Callable[..., AsyncGenerator]], Callable[..., cls]]:
            def decorator(func: Callable[..., AsyncGenerator]) -> Callable[..., cls]:
                @wraps(func)
                def wrapper(*args, **kwargs) -> cls:
                    return cls(name=func.__name__, span_func=func, args=args, kwargs=kwargs, accumulator=accumulator)
                return wrapper
            return decorator
        return component


P = ParamSpec("P")



def component(
    accumulator: SupportsExtend[CHUNK] | Callable[[], SupportsExtend[CHUNK]]
) -> Callable[[Callable[P, AsyncGenerator[CHUNK | StreamController, None]]], Callable[P, Component]]:
    def decorator(func: Callable[P, AsyncGenerator[CHUNK | StreamController, None]]) -> Callable[P, Component]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Component:
            return Component(name=func.__name__, span_func=func, args=args, kwargs=kwargs, accumulator=accumulator)
        return wrapper
    return decorator

# Decorator for span-enabled async generators using composition
