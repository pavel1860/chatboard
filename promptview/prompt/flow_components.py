import asyncio
import copy
from enum import Enum
from functools import wraps
import json
from queue import SimpleQueue
import datetime as dt

from typing import Any, AsyncGenerator, Callable, Iterable, Literal, ParamSpec, Protocol, Self, TypeVar, TYPE_CHECKING, runtime_checkable
import xml
from ..block import BlockChunk, BlockSchema
from ..block.block9.block_schema import BlockBuilderContext, BlockBuilderError
from ..prompt.injector import resolve_dependencies, resolve_dependencies_kwargs
from ..prompt.events import StreamEvent
from lxml import etree

from ..block import BlockSchema, Block
if TYPE_CHECKING:
    from ..model.versioning.models import ExecutionSpan, SpanValue, Log, SpanTypeEnum




class EventLogLevel(Enum):
    chunk = 0
    span = 1
    turn = 2
    
    


class BaseFbpComponent:
    
    def __init__(self, gen=None):
        self._gen = gen
        self._did_start = False
        self._did_yield = False
        self._did_end = False
        self._did_error = False
        self._last_value = None
        self._span = None
        
        
    @property
    def gen(self):
        if self._gen is None:
            raise ValueError(f"Flow generator ({self.__class__.__name__}) is not initialized")
        return self._gen
    
    @property
    def name(self):
        return self.gen.__name__
    

    
    

    
    def get_response(self):
        return self._last_value
    
    async def on_start_event(self, payload: Any = None, attrs: dict[str, Any] | None = None):
        raise NotImplementedError(f"Flow generator ({self.__class__.__name__}) does not implement on_start_event")
    
    async def on_value_event(self, payload: Any = None):
        raise NotImplementedError(f"Flow generator ({self.__class__.__name__}) does not implement on_value_event")
    
    async def on_stop_event(self, payload: Any = None):
        raise NotImplementedError(f"Flow generator ({self.__class__.__name__}) does not implement on_stop_event")
    
    async def on_error_event(self, error: Exception):
        raise NotImplementedError(f"Flow generator ({self.__class__.__name__}) does not implement on_error_event")
    
    async def on_start(self, value: Any = None):
        pass
    
    async def on_stop(self):
        pass
    
    async def on_error(self, error: Exception):
        pass
    
    
    async def pre_next(self):
        """
        This method is called before the next value is yielded.        
        """
        pass
    
    async def post_next(self, value: Any = None):
        """
        This method is called after the next value is yielded.
        """
        pass
    
    def __call__(self, _=None): return self
    
    def __or__(self, other: "BaseFbpComponent"):
        return self.pipe(other)
        # if isinstance(other, type):
        #     # `Parser | Accumulator` returns a callable
        #     return lambda x: other(self(x))
        # else:
        #     return other.__class__(self(other._gen))
    def __ior__(self, other: "BaseFbpComponent"):
        return self.pipe(other)
        
    def pipe(self, other: 'BaseFbpComponent'):
        other._gen = self
        return other
    
    
    async def start_generator(self):
        if self._did_start:
            raise RuntimeError(f"Generator ({self.__class__.__name__}) already started")
        await self.on_start()
        self._did_start = True
    
    def __aiter__(self):
        return self
    
    
    async def __anext__(self):
        try:
            if not self._did_start:
                await self.start_generator()
            value = await self.gen.__anext__()
            self._last_value = value
            if not self._did_yield:
                self._did_yield = True
            await self.post_next(value)
            return value
        except StopAsyncIteration:
            await self.on_stop()
            raise StopAsyncIteration
        except Exception as e:
            await self.on_error(e)
            raise e
    
    async def asend(self, value: Any = None):
        # print(f"Intercepted asend({value})")
        # Optionally modify the value before sending
        if not self._did_start:
            await self.on_start(value)
            self._did_start = True
        value = await self.gen.asend(value)
        self._last_value = value
        await self.post_next(value)
        return value

    async def athrow(self, typ, val=None, tb=None):
        return await self.gen.athrow(typ, val, tb)

    async def aclose(self):
        return await self.gen.aclose()



class Stream(BaseFbpComponent):
    _index = 0
    
    
    def __init__(self, gen: AsyncGenerator, name: str = "stream"):
        super().__init__(gen)
        self._name = name
        self._save_stream_dir: str | None = None
    

    def __aiter__(self):
        return self
    
    def save_stream(self, filepath: str):
        self._save_stream_dir = filepath        
    
    
    async def pre_next(self):
        self._index += 1
        
    async def post_next(self, value: Any = None):
        if self._save_stream_dir:
            with open(self._save_stream_dir, "a") as f:
                f.write(json.dumps(value.model_dump()) + "\n")
        return value
        
    async def on_stop(self):
        self._index -= 1
        
    # async def __anext__(self):
    #     try:
    #         self._index += 1
    #         return await self.gen.__anext__()
    #     except StopAsyncIteration:
    #         self._index -= 1
    #         raise StopAsyncIteration


            
    

class Parser(BaseFbpComponent):
      
    def __init__(self, response_schema: "Block", gen=None) -> None:
        super().__init__(gen)
        self.start_tag = "tag_start"
        self.end_tag = "tag_end"
        self.text_tag = "chunk"                
        self._safety_tag = "stream_start"        
        self.res_ctx = BlockBuilderContext(response_schema.copy())
        self.parser = etree.XMLPullParser(events=("start", "end"))
        self.block_buffer = []
        self._stream_started = False
        self._detected_tag = False
        self._total_chunks = 0
        self._chunks_from_last_tag = 0
        self._tag_stack = []
        self._full_content = ""
        
        

    def __aiter__(self):
        return self
    
    def _push_tag(self, tag: str, is_list: bool):
        self._tag_stack.append({
            "tag": tag,
            "is_list": is_list
        })
    
    def _pop_tag(self):
        return self._tag_stack.pop()
    
    @property
    def current_tag(self):
        if not self._tag_stack:
            return None
        return self._tag_stack[-1]["tag"]
    
    @property
    def current_tag_is_list(self):
        if not self._tag_stack:
            return False
        return self._tag_stack[-1]["is_list"]
    
    
    
    def _read_buffer(self, start_from: str | None = None, flush=True):
        buffer = []
        start_appending = start_from is None
        for block in self.block_buffer:
            if start_from and start_from in block.content:
                start_appending = True
            if start_appending:
                buffer.append(block)
        if flush:
            self.block_buffer = []
        return buffer
    
    def _write_to_buffer(self, value: Any):
        self._total_chunks += 1
        self._chunks_from_last_tag += 1
        self.block_buffer.append(value)
        
        
    def _buffer_size(self):
        return len(self.block_buffer)
    

        
    def _try_set_tag_lock(self, value: Any):
        if "<" in value.content:
            self._detected_tag = True
            
    def _release_tag_lock(self):
        self._chunks_from_last_tag = 0
        self._detected_tag = False
        
    def _should_output_chunk(self):
        if self.current_tag and not self._detected_tag:
            if self._chunks_from_last_tag < 2:
                return False
            return True
        return False
    
    
    def _feed_parser(self, content: str):
        # try:
        self._full_content += content
        self.parser.feed(content)
        # except Exception as e:
        #     print(self._full_content)
        #     print(f"Error happened on content: {content}")
        #     raise e

        
    async def asend(self, value: Any = None):
        for i in range(20):            
            try:
                if not self._stream_started:
                    self._feed_parser(f'<{self._safety_tag}>')
                    self._stream_started = True                
                if self.res_ctx.has_events():
                    v = self.res_ctx.get_event()
                    # print("1", type(v), v)
                    return v
                value = await self.gen.asend(value) 
                # print(value.content)
                self._write_to_buffer(value)
                try:
                    self._feed_parser(value.content)
                except Exception as e:
                    raise e
                
                self._try_set_tag_lock(value)
                
                # in the middle of the stream, adding chunks to the current field
                if self._should_output_chunk():               
                    for c in self._read_buffer(flush=True):
                        chunk = self.res_ctx.append(
                            self.current_tag,
                            c,
                        )
                
                # start or end of a field, adding the whole field to the queue
                for event, element in self.parser.read_events():
                    if element.tag == self._safety_tag:
                        continue                
                    if event == 'start':
                        # start of a field
                        if self.current_tag_is_list:
                            view, schema = self.res_ctx.instantiate_list_item(
                                self.current_tag,
                                element.tag,
                                self._read_buffer(),
                                attrs=dict(element.attrib),
                            )
                        else:
                            view, schema = self.res_ctx.instantiate(
                                element.tag,
                                self._read_buffer(),
                                attrs=dict(element.attrib),
                            )                    
                        
                        self._push_tag(element.tag, schema.is_list)
                        self._release_tag_lock()
                    elif event == 'end':
                        # end of a field
                        self.res_ctx.set_view_attr(
                            element.tag,
                            postfix=self._read_buffer(start_from="</"), 
                        )
                        # self.res_ctx.commit_view(
                        #     element.tag,
                        # )
                        self._pop_tag()
                        # field = self.response_schema.commit_field(
                        #     element.tag, 
                        #     postfix=self._read_buffer(start_from="</"), 
                        #     tags=[self.end_tag]
                        # )
                        # if field.postfix is not None:
                            # self._push_to_output(field.postfix)
                        
                        self._release_tag_lock()
                    
                        
                if self.res_ctx.has_events():
                    v = self.res_ctx.get_event()
                    return v
            except StopAsyncIteration:
                raise 
            except Exception as e:
                print(self._full_content)
                print(f"Parser Error happened on content: {value.content}")
                raise e
        else:
            raise StopAsyncIteration
        

    

class Accumulator(BaseFbpComponent):
    def __init__(self, accumulator) -> None:
        super().__init__(None)
        self._accumulator = accumulator
        
    @property
    def result(self):
        return self._accumulator
        
    def __aiter__(self):
        return self
    
    
    async def post_next(self, value: Any = None):
        return self._accumulator.append(value)
    
    # async def __anext__(self):
    #     try:
    #         res = await self._gen.__anext__()
    #         # print(res)
    #         self._accumulator.append(res)
    #         return res
    #     except StopAsyncIteration:
    #         raise StopAsyncIteration
  
@runtime_checkable
class Spanable(Protocol):
    index: int
    async def span_start(self, index: int, parent_span: "ExecutionSpan | None" = None):
        pass
    async def span_end(self):
        pass
    async def span_error(self, error: Exception):
        pass
    

    
    
CHUNK = TypeVar("CHUNK")

class SupportsExtend(Protocol[CHUNK]):
    def extend(self, iterable: Iterable[CHUNK], /) -> None: ...
    def append(self, item: CHUNK, /) -> None: ...
    def __iter__(self) -> Iterable[CHUNK]: ...

RESPONSE_ACC = TypeVar("RESPONSE_ACC", bound="SupportsExtend[Any]")

P = ParamSpec("P")
T = TypeVar("T")


class StreamController(BaseFbpComponent):
    
    def __init__(
        self, 
        gen: AsyncGenerator, 
        name:str,
        tags: list[str] | None = None,
        span_type: "SpanTypeEnum" = "stream",
        response_schema=None, 
        acc_factory=None
    ):
        super().__init__(None)
        self._name = name
        self._tags = tags
        self._stream = Stream(gen)
        self._gen = self._stream
        # self._acc = Accumulator(BlockList())
        # self._gen |= self._acc
        self._response_schema = response_schema
        # self._acc_factory = acc_factory or (lambda: BlockList(style="stream"))
        self._parser = None
        self.index = 0
        self._span_type = span_type
        self.parent: "PipeController | None" = None
        self._span: "ExecutionSpan | None" = None
        self.span_value: "SpanValue | None" = None
        
        
    async def build_span(self, parent_span_id: str | None = None):
        from ..model import ExecutionSpan
        self._span = await ExecutionSpan(
            span_type=self._span_type,
            name=self._name,
            index=self.index,
            tags=self._tags,
            start_time=dt.datetime.now(),
            parent_span_id=parent_span_id,
        ).save()
        return self._span
    
    @property
    def acc(self):
        if self._acc is None:
            raise ValueError("StreamController is not initialized")
        return self._acc.result
    
    @property
    def span(self):
        if self._span is None:
            raise ValueError(f"Span is not started for {self.__class__.__name__}")
        return self._span
    
    @property
    def span_id(self):
        return self.span.id

    def get_execution_path(self) -> list[int]:
        """Build execution path using existing index tracking"""
        path = []
        current = self
        while current:
            path.insert(0, current.index)  # Prepend to build path from root
            current = current.parent
        return path
    
    def get_response(self):
        if self._parser:
            return self._parser.res_ctx.instance
        # return self.acc
        
    def parse(self, block_schema: Block) -> Self:
        if self._parser is not None:
            raise ValueError("Parser already initialized")
        if self._gen is None:
            raise ValueError("StreamController is not initialized")
        self._parser = Parser(response_schema=block_schema)
        self._gen |= self._parser        
        return self
    
    def save(self, name: str, dir: str | None = None):
        import os
        path = f"{dir}/{name}.jsonl" if dir else f"{name}.jsonl"
        self._stream.save_stream(path)
        if os.path.exists(path):
            os.remove(path)
        return self
    
    def load(self, name: str, dir: str | None = None, delay: float = 0.07):
        import asyncio
        import random
        path = f"{dir}/{name}.jsonl" if dir else f"{name}.jsonl"
        
        async def load_stream():
            with open(path, "r") as f:
                for line in f:
                    # Add random delay between 0.5x and 1.5x of base delay
                    random_delay = delay * (max(-0.5 + random.random(), 0))
                    if random_delay > 0:
                        await asyncio.sleep(random_delay)
                    j = json.loads(line)
                    block = BlockChunk.model_validate(j)
                    yield block
                    
        self._gen = Stream(load_stream())
        return self
    
                
    
    @property
    def name(self):
        return self._name


    # async def on_start(self, value: Any = None):
    #     from ..model.versioning.models import ExecutionSpan
    #     self._span = await ExecutionSpan(
    #         span_type=self._span_type,
    #         name=self._name,
    #         index=self.index,
    #         start_time=dt.datetime.now(),
    #         parent_span_id=self.parent.span_id if self.parent else None,
    #     ).save()
    #     if self.parent:
    #         await self.parent.span.add_span(self.span, self.parent.index)
    
    async def on_start(self, value: Any = None):
        if not self._span:
            from ..model.versioning.models import ExecutionSpan
            self._span = await ExecutionSpan(
                span_type=self._span_type,
                name=self._name,
                index=self.index,
                tags=self._tags,
                start_time=dt.datetime.now(),
                parent_span_id=self.parent.span_id if self.parent else None,
            ).save()

            
    async def on_stop(self):
        self.span.end_time = dt.datetime.now()
        self.span.status = "completed"
        self._span = await self.span.save()
        
    async def on_error(self, error: Exception):
        from ..model.versioning.models import Log
        log = await Log(
            message=str(error),
            level="error"
        ).save()
        self.span.end_time = dt.datetime.now()
        self.span.status = "failed"
        self.span.metadata = {"error": str(error)}
        self._span = await self.span.save()
        self.span_value = None
        
        
    
        
        
    
    
    async def on_start_event(self, payload: Any = None, attrs: dict[str, Any] | None = None):
        value = await self.span.log_value(self.span, io_kind="output")
        self.span_value = value
        StreamEvent(
            type="stream_start", 
            name=self._name, 
            payload=self.span, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            value=self.span_value,
            parent_value_id=self.span_value.id if self.span_value else None,
        )
    
    async def on_value_event(self, payload: Any = None):
        
        return StreamEvent(
            type="stream_delta", 
            name=self._name, 
            payload=payload, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            parent_value_id=self.span_value.id if self.span_value else None,
        )
    
    async def on_stop_event(self, payload: Any = None):
        print("STREAM CONTROLLER on_stop_event", self.get_response())
        response = self.get_response()
        if response is not None:
            await self.span.log_value(response)
        return StreamEvent(
            type="stream_end", 
            name=self._name, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            parent_value_id=self.span_value.id if self.span_value else None,
        )
    
    async def on_error_event(self, error: Exception):
        from ..model.versioning.models import Log
        log = await Log(
            message=str(error),
            level="error"
        ).save()
        value = await self.span.log_value(log)
        return StreamEvent(
            type="stream_error", 
            name=self._name, 
            payload=error, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            value=value,
            parent_value_id=self.span_value.id if self.span_value else None,
        )
        
    def __aiter__(self):
        return FlowRunner(self)
    
    async def stream_events(self, event_level: EventLogLevel = EventLogLevel.chunk):
        return FlowRunner(self, event_level).stream_events()
    


    async def stream(self, *args, **kwargs):
        raise NotImplementedError("StreamController is not streamable")
        yield
        
    @classmethod
    def decorator_factory(cls) -> Callable[[], Callable[[Callable[P, AsyncGenerator[CHUNK, None]]], Callable[P, Self]]]:
        def decorator_wrapper(
            # accumulator: SupportsExtend[CHUNK] | Callable[[], SupportsExtend[CHUNK]]
        ) -> Callable[[Callable[P, AsyncGenerator[CHUNK, None]]], Callable[P, Self]]:
            def decorator(
                func: Callable[P, AsyncGenerator[Any, Any]]
            ) -> Callable[P, Self]:
                @wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> Self:            
                    return cls(gen=func(*args, **kwargs), name=func.__name__)
                return wrapper
            return decorator
        return decorator_wrapper



# def streamable(func: Callable[P, AsyncGenerator[CHUNK, None]]) -> Callable[P, StreamController]:
#     @wraps(func)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> StreamController:
#         return StreamController(gen_func=func, args=args, kwargs=kwargs)
#     return wrapper



class PipeController(BaseFbpComponent):
    
    
    def __init__(
        self, 
        gen_func, 
        name: str, 
        span_type: "SpanTypeEnum",
        tags: list[str] | None = None,
        args = (),
        kwargs = {}
    ):
        super().__init__(None)
        self._gen_func = gen_func
        self._tags = tags
        self._args = args
        self._kwargs = kwargs
        self._name = name
        self._span_type = span_type
        self.parent: "PipeController | None" = None
        self._span: "ExecutionSpan | None" = None
        self.index = 0
        self.event: "SpanValue | None" = None
    
    @property
    def span(self):
        if self._span is None:
            raise ValueError(f"Span is not started for {self.__class__.__name__}")
        return self._span

    @property
    def span_id(self):
        return self.span.id

    def get_execution_path(self) -> list[int]:
        """Build execution path using existing index tracking"""
        path = []
        current = self
        while current:
            path.insert(0, current.index)  # Prepend to build path from root
            current = current.parent
        return path
        
    async def on_start_event(self, payload: Any = None, attrs: dict[str, Any] | None = None):
        return StreamEvent(
            type="span_start", 
            name=self._gen_func.__name__, 
            payload=self.span, 
            attrs=attrs, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            parent_value_id=self.event.id if self.event else None,            
        )
    
    async def on_value_event(self, payload: Any = None):
        if isinstance(payload, StreamController):
            span = await payload.build_span(str(self.span_id))
            event = await self.span.log_value(span)
            payload.span_value = event
            return StreamEvent(
                type="span_event", 
                name=self._gen_func.__name__, 
                payload=span, 
                span_id=str(self.span_id), 
                path=self.get_execution_path(), 
                value=event,
                parent_value_id=self.event.id if self.event else None,
            )
            
        elif isinstance(payload, PipeController):
            span = await payload.build_span(str(self.span_id))
            event = await self.span.log_value(span)
            payload.event = event
            return StreamEvent(
                type="span_event", 
                name=self._gen_func.__name__, 
                payload=span, 
                span_id=str(self.span_id), 
                path=self.get_execution_path(), 
                value=event,
                parent_value_id=self.event.id if self.event else None,
            )
        elif isinstance(payload, Block):
            event = await self.span.log_value(payload)
            return StreamEvent(
                type="span_event", 
                name=self._gen_func.__name__, 
                payload=payload, 
                span_id=str(self.span_id), 
                path=self.get_execution_path(), 
                value=event,
                parent_value_id=self.event.id if self.event else None,
            )
        else:
            # raise ValueError(f"Invalid payload type: {type(payload)}")
            return StreamEvent(type="span_event", name=self._gen_func.__name__, payload=payload, span_id=str(self.span_id), path=self.get_execution_path())
    
    async def on_stop_event(self, payload: Any = None):
        return StreamEvent(
            type="span_end", 
            name=self._gen_func.__name__, 
            payload=payload, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            parent_value_id=self.event.id if self.event else None,
        )
    
    async def on_error_event(self, error: Exception):
        from ..model.versioning.models import Log
        log = await Log(
            message=str(error),
            level="error"
        ).save()
        event = await self.span.log_value(log)
        return StreamEvent(
            type="span_error", 
            name=self._gen_func.__name__, 
            payload=error, 
            span_id=str(self.span_id), 
            path=self.get_execution_path(), 
            value=event
        )
    
    async def post_next(self, value: Any = None):
        self.index += 1
        return value
    
    # async def post_next(self, value: Any = None):
    #     ..model.versioning.models import Turn
    #     if isinstance(value, Block):
    #         turn = Turn.current()
    #         if turn:
    #             await turn.add_block(value, self.index, span_id=self.span_id)
    #             self.index += 1
    #     return value
    
    # async def post_next(self, value: Any = None):
    #     from ..model.versioning.models import ExecutionSpan
    #     if isinstance(value, Block):
    #         await self.span.add_block(value, self.index)
    #         self.index += 1
    #     return value
    def get_response(self):
        if self._last_value and isinstance(self._last_value, BaseFbpComponent):
            return self._last_value.get_response()
        return self._last_value
    
    @property
    def response(self):
        return self._last_value
    
    async def build_span(self, parent_span_id: str | None = None):
        from ..model.versioning.models import ExecutionSpan
        self._span = await ExecutionSpan(
            span_type=self._span_type,
            name=self._name,
            index=self.index,
            tags=self._tags,
            start_time=dt.datetime.now(),
            parent_span_id=parent_span_id,
        ).save()
        return self._span
    
    async def add_span(self, span: "ExecutionSpan"):
        event = await self.span.log_value(span)
        return event
    
    
    async def add_event(self, gen: "PipeController | StreamController"):
        if isinstance(gen, StreamController):
            span = await gen.build_span(str(self.span_id))
            event = await self.span.log_value(span)
            gen.span_value = event
            return event
        elif isinstance(gen, PipeController):
            span = await gen.build_span(str(self.span_id))
            event = await self.span.log_value(span)        
            gen.event = event
            return event
        else:
            raise ValueError(f"Invalid generator type: {type(gen)}")

    async def on_start(self, value: Any = None):
        from ..model.versioning.models import ExecutionSpan
        bound, kwargs = await resolve_dependencies_kwargs(self._gen_func, self._args, self._kwargs)
        self._gen = self._gen_func(*bound.args, **bound.kwargs)
        if not self._span:
            self._span = await ExecutionSpan(
                span_type=self._span_type,
                name=self._name,
                tags=self._tags,
                index=self.index,
                parent_span_id=self.parent.span_id if self.parent else None,
            ).save()
        
        
    async def on_stop(self):
        self.span.end_time = dt.datetime.now()
        self.span.status = "completed"
        self._span = await self.span.save()
        
    async def on_error(self, error: Exception):
        from ..model.versioning.models import Log
        log = await Log(
            message=str(error),
            level="error"
        ).save()
        await self.span.log_value(log)
        self.span.end_time = dt.datetime.now()
        self.span.status = "failed"
        self.span.metadata = {"error": str(error)}
        self._span = await self.span.save()
    
    def __aiter__(self):
        return FlowRunner(self)
    
    def stream_events(self, event_level: EventLogLevel = EventLogLevel.chunk):
        return FlowRunner(self, event_level).stream_events()
            
    @classmethod
    def decorator_factory(cls) -> Callable[[], Callable[[Callable[P, AsyncGenerator[CHUNK, None]]], Callable[P, Self]]]:
        def component_decorator(
            # accumulator: SupportsExtend[CHUNK] | Callable[[], SupportsExtend[CHUNK]]
        ) -> Callable[[Callable[P, AsyncGenerator[CHUNK | StreamController, None]]], Callable[P, Self]]:
            def decorator(func: Callable[P, AsyncGenerator[CHUNK | StreamController, None]]) -> Callable[P, Self]:
                @wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> Self:
                    return cls(gen_func=func, args=args, kwargs=kwargs)
                return wrapper
            return decorator
        return component_decorator



    


class FlowRunner:
    def __init__(self, gen: BaseFbpComponent, event_level: EventLogLevel = EventLogLevel.chunk):
        self.stack: list[BaseFbpComponent] = [gen]
        self.last_value: Any = None
        self._output_events = False
        self._error_to_raise = None
        self._last_gen = None
        self._event_level = event_level
        
        
    @property
    def current(self):
        if not self.stack:
            raise StopAsyncIteration
            # raise ValueError("Stack is empty")
        return self.stack[-1]
    
    @property
    def should_output_events(self):
        if self._output_events:
            return True
        return False
    
    def push(self, value: BaseFbpComponent):
        self.stack.append(value)
        
    def pop(self):
        gen = self.stack.pop()
        self._last_gen = gen
        return gen
    
    
    def __aiter__(self):
        return self
    
    def _get_response(self):
        if self._last_gen:
            res = self._last_gen.get_response()
            self._last_gen = None
            return res
        else:
            last_yield = self.current._last_value
            if last_yield:
                return last_yield    
        return None
    
    
    def _try_to_gen_event(self, func: Callable[[], Any], value: Any):
        if isinstance(value, StreamEvent):
            return value
        return func(value)
    
    
    async def try_build_start_event(self, gen: BaseFbpComponent, value: Any):
        event = await gen.on_start_event(value)
        if event is None:
            return None
        if self._event_level == EventLogLevel.span:
            return None
        elif self._event_level == EventLogLevel.turn:
            return None
        elif self._event_level == EventLogLevel.chunk:
            return event
        return None
    
    
    async def try_build_value_event(self, gen: BaseFbpComponent, value: Any):
        event = await gen.on_value_event(value)
        if event is None:
            return None
        if self._event_level == EventLogLevel.span:
            if isinstance(value, PipeController):
                return event
            return None
        elif self._event_level == EventLogLevel.turn:
            return None
        elif self._event_level == EventLogLevel.chunk:
            return event
        return None
    
    async def try_build_stop_event(self, gen: BaseFbpComponent, value: Any):
        event = await gen.on_stop_event(value)
        if event is None:
            return None
        if self._event_level == EventLogLevel.span:
            return gen.get_response()
        elif self._event_level == EventLogLevel.turn:
            return None
        elif self._event_level == EventLogLevel.chunk:
            return event
        return None
    
    async def try_build_error_event(self, gen: BaseFbpComponent, value: Any):
        event = await gen.on_error_event(value)
        if event is None:
            return None
        if self._event_level == EventLogLevel.span:
            return event
        elif self._event_level == EventLogLevel.turn:
            return None
        elif self._event_level == EventLogLevel.chunk:
            return event
        return None
    
    

    


    async def __anext__(self):        
        
        value = None
        while self.stack:
            
            try:                
                gen = self.current            
                if self._error_to_raise:
                    # raise self._error_to_raise
                    await gen.athrow(self._error_to_raise)
                if not gen._did_start:
                    await gen.start_generator()                    
                    payload = gen.span if isinstance(gen, PipeController) and len(self.stack) == 1 else None
                    # event = await gen.on_start_event(payload)
                    # if not self.should_output_events:
                    #     continue
                    # return event
                    if event:=await self.try_build_start_event(gen, payload):
                        return event
                    else:
                        continue
                    

                response = self._get_response()
                value = await gen.asend(response)
                
                if isinstance(value, StreamController):
                    self.push(value)
                elif isinstance(value, PipeController):
                    self.push(value)
                
                self.last_value = value
                # event = await gen.on_value_event(value)
                if event:=await self.try_build_value_event(gen, value):
                    return event
                # if not self.should_output_events:
                #     return value
                # return event
                
            except StopAsyncIteration:
                gen = self.pop()
                await gen.on_stop()
                # event = await gen.on_stop_event(None)
                if event:=await self.try_build_stop_event(gen, value):
                    return event
                # if not self.should_output_events:
                #     continue
                # return event                
            except Exception as e:
                gen = self.pop()
                event = await gen.on_error_event(e)
                await gen.on_error(e)                
                if not self.should_output_events or not self.stack:
                    raise e
                self._error_to_raise = e
                return event
        else:
            raise StopAsyncIteration

    
    def stream_events(self, event_level: EventLogLevel = EventLogLevel.chunk):
        self._event_level = event_level
        self._output_events = True
        return self
    
    
    
    

