from functools import wraps
import inspect
from typing import Any, AsyncGenerator, Callable, Iterable, Literal, ParamSpec, Protocol, Union, AsyncIterator, Optional, Generic
from typing_extensions import TypeVar

from ..prompt.events import StreamEvent



class StreamResponse:
    
    def __init__(self, value: Any):
        self.value = value
        
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"StreamResponse({self.value})"







CHUNK = TypeVar("CHUNK")

class SupportsExtend(Protocol[CHUNK]):
    def extend(self, iterable: Iterable[CHUNK], /) -> None: ...
    def append(self, item: CHUNK, /) -> None: ...
    def __iter__(self) -> Iterable[CHUNK]: ...

StreamFilter = Literal["pass_events", "all", "self"]
RESPONSE_ACC = TypeVar("RESPONSE_ACC", bound="SupportsExtend[Any]")


class GeneratorFrame(Generic[CHUNK]):
    def __init__(self, controller: "StreamController", agen: AsyncGenerator[CHUNK, None], accumulator: SupportsExtend[CHUNK]):
        self.controller = controller
        self.agen = agen
        self.accumulator = self._init_accumulator(accumulator)
        self.emitted_start = False
        self.started = False
        self.index = 0
        self.exhausted = False

    async def advance(self, value: CHUNK | None = None):
        try:
            self.index += 1
            if not self.started:
                self.started = True
                return await self.agen.__anext__()
            else:
                # return await self.agen.asend(self.accumulator)
                return await self.agen.asend(value or self.accumulator)
        except StopAsyncIteration as e:
            self.exhausted = True
            return self.accumulator
            
        
    @property
    def output_mode(self) -> Literal["events", "chunks"]:
        return self.controller._output_mode
    
    @property
    def filter_mode(self) -> StreamFilter:
        return self.controller._filter_mode
    
    def to_event(self, value: CHUNK) -> StreamEvent:
        return self.controller.to_event(self, value)
        
    def _init_accumulator(self, acc) -> Any:
        if acc is None:
            return None
        elif callable(acc):
            return acc()
        else:
            return acc
        
    def try_append(self, value: Any):
        # Try using append or += for accumulation
        if self.accumulator is None:
            return None
        try:
            self.accumulator.append(value)
        except AttributeError:
            try:
                self.accumulator += value
            except Exception:
                pass  # Optionally: raise or log a warning

P = ParamSpec("P")



class StreamController(Generic[P, CHUNK]):
    def __init__(
        self,
        name: str | None = None,
        agen: Union[AsyncGenerator[CHUNK , None], Callable[[], AsyncGenerator[CHUNK, None]]] | None = None,
        accumulator: SupportsExtend[CHUNK] | Callable[[], SupportsExtend[CHUNK]] | None = None
    ):
        self._name = name
        self._stack = []
        self._initial_gen = agen or self.stream
        self._accumulator_factory = accumulator
        self._raise_on_next = False
        self._output_mode: Literal["events", "chunks"] = "chunks"
        self._filter_mode: StreamFilter = "pass_events"
        
        
    @property
    def response(self) -> SupportsExtend[CHUNK]:
        return self.current.accumulator
        
    async def stream(self, *args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any, None]:
        """
        Override this method in your subclass to implement your custom logic.
        This must be an async generator.
        """
        pass
        yield
        
        
    @property
    def current(self) -> Any:
        if not self._stack:
            raise ValueError("No current generator")
        return self._stack[-1]

        
    def __await__(self):
        async def _consume():
            async for _ in self:
                pass
            return self.current.accumulator
        return _consume().__await__()


    def __aiter__(self) -> AsyncIterator[Any]:
        # self._stack = [self.build_frame()]
        return self

    # async def __anext__(self) -> Any:
    #     frame_response = None
    #     if self._raise_on_next:
    #         self._raise_on_next = False
    #         raise StopAsyncIteration(self.current.accumulator)
    #     while self._stack:
    #         try:
    #             value = await self.current.advance(frame_response)
    #             frame_response = None
                
    #             if isinstance(value, StreamController):
    #                 if self._filter_mode == "pass":                        
    #                     return value
    #                 else:
    #                     value._output_mode = self._output_mode
    #                     self._stack.append(value.build_frame())
    #                     continue                    

    #             # Attempt to append to the accumulator
    #             self.current.try_append(value)
                
    #             if self.current.output_mode == "events":
    #                 return self.current.to_event(value)
    #             return value

    #         except StopAsyncIteration as e:
    #             if len(self._stack) == 1:
    #                 raise e
    #                 # self._raise_on_next = True                    
    #                 # return self.current.accumulator
    #             frame = self._stack.pop()
    #             frame_response = frame.accumulator
                
    #             if hasattr(e, "value") and e.value is not None:
    #                 self._accumulator = e.value

    #     raise StopAsyncIteration
    @property
    def is_self_stream(self):
        return len(self._stack) == 1
    
    async def __anext__(self) -> Any:
        if not self._stack:
            self._stack.append(await self.build_frame())
        
        while self._stack:
            if not self.current.emitted_start and self.current.output_mode == "events":
                return self.current.to_event(None)
            if self.current.exhausted: 
                if len(self._stack) == 1:
                    raise StopAsyncIteration()                                   
                frame = self._stack.pop()
                value = await self.current.advance(frame.accumulator)
            else:    
                value = await self.current.advance()
            if isinstance(value, StreamController):
                if self._filter_mode == "pass":                        
                    return value
                else:
                    value._output_mode = self._output_mode                    
                    frame = await value.build_frame()
                    self._stack.append(frame)
                    continue                    

            # Attempt to append to the accumulator
            if not self.current.exhausted:
                self.current.try_append(value)
            
            # if (
            #     not self.is_self_stream or \
            #     self._output_mode == "events" or \
            #     self._filter_mode in ["pass_events", "all"] \
            # ):
            #     return self._pack_value(value)
            # else:
            #     continue
            if self._should_output(value):
                return self._pack_value(value)
            else:
                continue

            
                
                    
            
        raise StopAsyncIteration
    
    def _should_output(self, value: Any):
        if not self.is_self_stream:
            if self._filter_mode == "all":
                return True
            elif self._filter_mode == "pass_events" and self._output_mode == "events":
                return True
            else:
                return False
        return True
                        

    def _pack_value(self, value: Any):
        if self.current.output_mode == "events":
            value = self.current.to_event(value)
        return value

    def _wrap(self, value: Any) -> GeneratorFrame:
        if inspect.isasyncgenfunction(value):
            value = value()
        if not inspect.isasyncgen(value):
            raise TypeError(f"{value} is not an async generator")
        return GeneratorFrame(self,value, self._accumulator_factory)
    
    async def build_frame(self) -> GeneratorFrame:
        if inspect.ismethod(self._initial_gen):
            return GeneratorFrame(self, self._initial_gen(), self._accumulator_factory)
        else:
            return GeneratorFrame(self, self._initial_gen, self._accumulator_factory)

    
    def config(self, filter_mode: StreamFilter = "all", output_mode: Literal["events", "chunks"] = "chunks"):
        self._filter_mode = filter_mode
        self._output_mode = output_mode
        return self
    
    def to_event(self, ctx: GeneratorFrame, value: CHUNK) -> StreamEvent:
        if not ctx.emitted_start:
            ctx.emitted_start = True
            return StreamEvent(type="stream_start", name=self._name, payload=None, index=0)
        elif ctx.exhausted:
            return StreamEvent(type="stream_end", name=self._name, payload=ctx.accumulator, index=ctx.index)
        else:
            return StreamEvent(type="message_delta", name=self._name, payload=value, index=ctx.index)
        
        

    async def stream_events(self, filter_mode: StreamFilter = "all"):
        self._filter_mode = filter_mode
        self._output_mode = "events"
        event = StreamEvent(type="stream_start", name=self._name, payload=None, index=0)
        yield event

        async for event in self:
            if not isinstance(event, StreamEvent):
                event = StreamEvent(type="message_delta", name=self._name, payload=event, index=self.current.index)
            yield event

        event = StreamEvent(type="stream_end", name=self._name, payload=self.current.accumulator, index=self.current.index)
        yield event



P = ParamSpec("P")
T = TypeVar("T")



def stream(
    accumulator: SupportsExtend[CHUNK] | Callable[[], SupportsExtend[CHUNK]]
) -> Callable[[Callable[P, AsyncGenerator[CHUNK, None]]], Callable[P, StreamController[P, CHUNK]]]:
    def decorator(
        func: Callable[P, AsyncGenerator[Any, Any]]
    ) -> Callable[P, StreamController]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamController:
            agen = func(*args, **kwargs)
            return StreamController(name=func.__name__, agen=agen, accumulator=accumulator)
        return wrapper
    return decorator