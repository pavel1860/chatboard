"""
FBP Process - Flow-Based Programming Core Components

This module implements Flow-Based Programming (FBP) primitives for LLM streaming.

## The Telegram Problem & FBP

When information is split into parts (like streaming LLM tokens), each part must
preserve its context and boundaries. FBP solves this by treating data as continuous
flows of discrete packets (Information Packets - IPs), each carrying metadata,
flowing through independent, connected processes.

## Architecture

1. **Information Packets (IPs)**: Discrete data units with boundaries
   - In LLM context: BlockChunk, StreamEvent, nested Controllers

2. **Processes**: Independent components that transform IPs
   - All processes share same interface: receive IPs, transform, send IPs
   - Types: Stateless (Parser), Stateful (StreamController), Composite (PipeController)

3. **Connections**: Bounded buffers between processes (async generators)

4. **Ports**: Input/output interfaces (asend/yield)

## Replay Capability

Critical requirement: Process network must support starting execution from any
subcomponent by reconstructing state from SpanTree inputs/outputs.

Like telegram retransmission: if message fails at station 5, restart from station 5
by replaying messages from stations 1-4.

## Modes

- **Normal Mode**: Process receives IPs from upstream, executes, logs I/O to SpanTree
- **Replay Mode**: Process reconstructs from SpanTree, yields saved outputs

---

PRD: Flow-Based Programming Process Implementation
===================================================

## Goals

1. Create FBP process architecture for LLM streaming pipelines
2. Enable process composition via pipes (|)
3. Support observability through SpanTree integration
4. Enable replay/resumption from any point in process network
5. Maintain backward compatibility with existing flow_components.py

## Non-Goals

- Replacing flow_components.py immediately (gradual migration)
- Distributed execution across machines (single-process for now)
- Visual flow editor (code-first approach)

## Requirements

### Functional

1. **Process Base Class**
   - Async iteration protocol (__aiter__, __anext__)
   - Pipe composition operator (|)
   - Lifecycle hooks (on_start, on_stop, on_error)
   - Dual mode: normal execution vs replay from SpanTree
   - I/O logging to SpanTree for observability

2. **Stream Process**
   - Wraps async generator as FBP source
   - Optional persistence (save stream to file)
   - Optional replay (load stream from file)

3. **Accumulator Process**
   - Sink process that collects all IPs
   - Returns accumulated result

4. **StreamController Process**
   - Composite process (contains stream + optional parser)
   - Span tracking integration
   - Event emission (start, delta, end, error)
   - Replay from saved span

5. **PipeController Process**
   - Dynamic composite (yields other processes)
   - Child process management
   - Replay with subnetwork reconstruction

6. **Parser Process** (later phase)
   - XML stream parsing into structured blocks
   - Stateful transformation

7. **FlowRunner**
   - Orchestrates process network execution
   - Manages process stack (when processes yield other processes)
   - Emits events at different granularities (chunk/span/turn)

### Non-Functional

1. **Performance**: Minimal overhead vs raw async generators
2. **Testability**: Each component independently testable
3. **Type Safety**: Full type hints
4. **Documentation**: Inline examples and docstrings

"""

# Implementation starts here

from typing import Any, AsyncGenerator, Callable, Type, TYPE_CHECKING

import json
import asyncio
import hashlib
import os


# from ..versioning import DataFlowNode, ExecutionSpan, SpanType

if TYPE_CHECKING:
    from ..block import Block
    from ..versioning import DataFlowNode, ExecutionSpan, SpanType, LlmCall


def _serialize_for_hash(value: Any) -> Any:
    """
    Serialize a value for hashing. Handles Pydantic models, dicts, lists, and primitives.

    For Block objects, uses the text content for deterministic hashing.
    """
    from ..block import Block

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # Handle Block specially - use content hash for deterministic hashing
    if isinstance(value, Block):
        from ..versioning.block_storage import compute_block_hash
        return compute_block_hash(value)
    if hasattr(value, 'model_dump'):
        # Pydantic model - try with mode='json', fall back to without
        try:
            return value.model_dump(mode='json')
        except TypeError:
            try:
                return value.model_dump()
            except Exception:
                return str(value)
    if isinstance(value, dict):
        return {k: _serialize_for_hash(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_hash(v) for v in value]
    # For other types, use string representation
    return str(value)


def compute_stream_cache_key(name: str, bound_args: dict[str, Any]) -> str:
    """
    Compute a cache key hash from stream name and bound arguments.

    Args:
        name: The stream name (used as prefix)
        bound_args: Dictionary of bound arguments from inspect.signature.bind()

    Returns:
        A hash string suitable for use in filenames
    """
    # Filter out non-serializable types (Context, LLM instances, etc.)
    from .context import Context
    from ..llms import LLM, LlmConfig
    from ..evaluation.decorators import EvalCtx

    filtered_args = {}
    for key, value in bound_args.items():
        if value is None:
            continue
        # Skip 'self' argument (method bound to instance)
        if key == 'self':
            continue
        if isinstance(value, (Context, LLM, LlmConfig, EvalCtx)):
            continue
        # Skip private arguments
        if key.startswith('_'):
            continue
        # Skip objects that look like LLM instances (have 'stream' method)
        if hasattr(value, 'stream') and hasattr(value, 'complete'):
            continue
        filtered_args[key] = _serialize_for_hash(value)

    # Create deterministic JSON string
    json_str = json.dumps(filtered_args, sort_keys=True, default=str)

    # Debug: print what's being hashed
    print(f"[CACHE DEBUG] name={name}, json_str={json_str[:500]}")

    # Compute hash
    hash_value = hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]
    print(f"[CACHE DEBUG] hash_value={hash_value}")
    return f"{name}_{hash_value}.json"


if TYPE_CHECKING:
    from .span_tree import SpanTree, DataFlow
    from ..block import Block, BlockChunk, BlockSchema, BaseBlock
    from .context import Context


class FlowException(Exception):
    """Exception raised by FBP processes"""
    pass


class Process:
    """
    Base class for all FBP processes.

    Processes receive information packets (IPs), transform them,
    and send them downstream.

    In LLM streaming context:
    - IPs are BlockChunks, StreamEvents, or nested Controllers
    - Processes maintain boundaries and context
    - Connections are async generators (bounded buffers)

    Example:
        >>> async def gen():
        ...     for i in range(3):
        ...         yield i
        >>>
        >>> class MockUpstream:
        ...     def __aiter__(self):
        ...         return gen().__aiter__()
        ...     async def __anext__(self):
        ...         return await gen().__anext__()
        >>>
        >>> proc = Process(upstream=MockUpstream())
        >>> results = []
        >>> async for ip in proc:
        ...     results.append(ip)
        >>> print(results)  # [0, 1, 2]
    """

    def __init__(self, upstream: "Process | None" = None):
        """
        Initialize a process.

        Args:
            upstream: The upstream process (connection) to receive IPs from.
                     None for source processes (like Stream).
        """
        self._upstream = upstream
        self._did_start = False
        self._did_yield = False
        self._did_end = False
        self._last_ip: Any = None
        self._span_tree: "SpanTree | None" = None

    @property
    def upstream(self):
        """
        The upstream connection to receive IPs from.

        Raises:
            FlowException: If no upstream is set (e.g., process not connected)
        """
        if self._upstream is None:
            raise FlowException(
                f"Process {self.__class__.__name__} has no upstream connection"
            )
        return self._upstream
    
    def __call__(self, _=None): return self

    async def on_start(self, value: Any = None):
        """
        Called when process starts (FBP process initialization).

        Override this method to perform initialization tasks like:
        - Opening files/connections
        - Building subnetworks
        - Registering with context

        Args:
            value: Optional initial value
        """
        pass

    async def on_stop(self):
        """
        Called when process completes (FBP process termination).

        Override this method to perform cleanup tasks like:
        - Closing files/connections
        - Saving state
        - Unregistering from context
        """
        pass

    async def on_error(self, error: Exception):
        """
        Called when process encounters an error.

        Override this method to handle errors gracefully:
        - Log errors
        - Update span status
        - Cleanup resources

        Args:
            error: The exception that occurred
        """
        pass

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self):
        """
        Receive next IP from upstream.

        This implements the core FBP flow:
        1. On first call, initialize the process
        2. Receive IP from upstream
        3. Store the IP
        4. Log the IP as input (for observability)
        5. Return the IP downstream

        Returns:
            The next information packet from upstream

        Raises:
            StopAsyncIteration: When upstream is exhausted
        """
        try:
            # Initialize on first iteration
            if not self._did_start:
                await self.on_start()
                self._did_start = True

            # Receive IP from upstream
            ip = await self.upstream.__anext__()
            self._last_ip = ip

            # Log input to span tree for observability
            if self._span_tree:
                value = await self._span_tree.log_value(ip, io_kind="input")

            # Track that we've yielded at least once
            if not self._did_yield:
                self._did_yield = True

            return ip

        except StopAsyncIteration as e:
            # Upstream exhausted - cleanup and propagate
            await self.on_stop()
            self._did_end = True
            raise e

        except Exception as e:
            # Error occurred - handle and propagate
            await self.on_error(e)
            raise e

    async def asend(self, value: Any = None):
        """
        Send a value into the process.

        This enables replay mode where saved outputs can be sent back into
        the process network. Also logs the sent value as output for observability.

        Args:
            value: The value to send into the process

        Returns:
            The next information packet from upstream

        Raises:
            StopAsyncIteration: When upstream is exhausted
        """
        # Log output to span tree for observability
        if self._span_tree and value is not None:
            await self._span_tree.log_value(value, io_kind="output")

        # If upstream supports asend, use it; otherwise just call __anext__
        if hasattr(self._upstream, 'asend'):
            return await self._upstream.asend(value)
        else:
            return await self.__anext__()

    def __or__(self, downstream: "Process") -> "Process":
        """
        Pipe operator for process composition.

        Connects this process to a downstream process, creating a pipeline.
        This is the core FBP composition operator, similar to Unix pipes.

        Example:
            >>> stream | parser | accumulator

        Args:
            downstream: The process to connect downstream

        Returns:
            The downstream process (for chaining)
        """
        return self.connect(downstream)

    def connect(self, downstream: "Process") -> "Process":
        """
        Establish connection to downstream process.

        Sets this process as the upstream of the downstream process,
        creating a data flow connection.

        Args:
            downstream: The process to connect downstream

        Returns:
            The downstream process (for chaining)
        """
        downstream._upstream = self
        return downstream
    
    async def athrow(self, typ, val=None, tb=None):
        return await self.upstream.athrow(typ, val, tb)

    async def aclose(self):
        return await self.upstream.aclose()
    
    def did_end(self):
        return self._did_end
    
    def did_upstream_end(self):
        if self._upstream is None:
            raise ValueError("Upstream is not set")
        return self._upstream.did_end()



class Stream(Process):
    """
    Stream process - FBP source that wraps an async generator.

    Stream is a SOURCE process in FBP terminology - it generates IPs
    from an external source (async generator) rather than receiving
    them from an upstream process.

    Stream wraps a generator INSTANCE (not a function). The inputs that
    created this generator should be logged by the controller that created it.
    Stream itself has no inputs - it just yields what the generator produces.

    Example:
        >>> async def number_gen():
        ...     for i in range(5):
        ...         yield i
        >>>
        >>> stream = Stream(number_gen())
        >>> results = []
        >>> async for ip in stream:
        ...     results.append(ip)
        >>> print(results)  # [0, 1, 2, 3, 4]
    """

    def __init__(self, gen: AsyncGenerator, name: str = "stream"):
        """
        Initialize a Stream process.

        Args:
            gen: The async generator instance to wrap as a source
            name: Name for this stream (used in logging/debugging)
        """
        # Create a wrapper that makes the generator behave like an upstream process
        # This allows base Process.__anext__() to work correctly
        class GeneratorWrapper:
            """Wraps async generator to provide upstream interface"""
            def __init__(self, generator):
                self._gen = generator

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self._gen.__anext__()

            async def athrow(self, typ, val=None, tb=None):
                return await self._gen.athrow(typ, val, tb)

            async def aclose(self):
                return await self._gen.aclose()

        super().__init__(upstream=GeneratorWrapper(gen))
        self._name = name
        self._save_stream_path: str | None = None
        self._collected_chunks: list[Any] = []  # Collect chunks for JSONL save

    def save_stream(self, filepath: str):
        """
        Enable stream persistence - saves all IPs to a JSONL file on successful completion.

        Args:
            filepath: Path to JSONL file to save stream to
        """
        self._save_stream_path = filepath
        self._collected_chunks = []

    @classmethod
    def load(cls, filepath: str, delay: float = 0.0):
        """
        Load a stream from a saved JSONL file.

        Args:
            filepath: Path to JSONL file to load stream from
            delay: Optional delay between IPs (for simulating streaming)

        Returns:
            New Stream instance that replays from file
        """
        async def load_stream():
            from ..block import BlockChunk
            with open(filepath, "r") as f:
                for line in f:
                    if line.strip():
                        if delay > 0:
                            await asyncio.sleep(delay)
                        data = json.loads(line)
                        block = BlockChunk.model_load(data)
                        yield block

        return cls(load_stream(), name=f"stream_from_{filepath}")


    @classmethod
    def from_list(cls, chunks: list[str], name: str = "stream_from_list"):
        async def gen():
            from ..block import BlockChunk
            for chunk in chunks:
                yield BlockChunk(content=chunk, logprob=1.0)
        return cls(gen(), name=name)

    async def __anext__(self):
        """
        Receive next IP from wrapped generator.

        If persistence is enabled, collects IP for later save.
        Only saves to JSONL on successful stream completion (not on errors).
        """
        ip = await super().__anext__()

        # Collect for save if persistence enabled
        if self._save_stream_path is not None:
            if hasattr(ip, "model_dump"):
                data = ip.model_dump()
            else:
                data = ip
            self._collected_chunks.append(data)

        return ip

    async def on_stop(self):
        """Save collected chunks to JSONL file on successful completion."""
        if self._save_stream_path is not None and self._collected_chunks:
            with open(self._save_stream_path, "w") as f:
                for data in self._collected_chunks:
                    f.write(json.dumps(data) + "\n")
        await super().on_stop()
    




class Accumulator(Process):
    """
    Accumulator process - FBP sink that collects all IPs.

    Accumulator is a SINK process in FBP terminology - it consumes IPs
    and stores them in a buffer rather than transforming and passing them on.

    The accumulated result can be accessed via the `result` property.

    Example:
        >>> async def gen():
        ...     for i in range(5):
        ...         yield i
        >>>
        >>> stream = Stream(gen())
        >>> acc = Accumulator()
        >>> pipeline = stream | acc
        >>>
        >>> async for _ in pipeline:
        ...     pass
        >>>
        >>> print(acc.result)  # [0, 1, 2, 3, 4]
    """

    def __init__(self, accumulator: list | None = None, upstream: Process | None = None):
        """
        Initialize an Accumulator process.

        Args:
            accumulator: The buffer to accumulate IPs into. Defaults to empty list.
            upstream: The upstream process to receive IPs from
        """
        super().__init__(upstream)
        self._accumulator = accumulator if accumulator is not None else []

    @property
    def result(self):
        """
        Get the accumulated result.

        Returns:
            The accumulator buffer containing all collected IPs
        """
        return self._accumulator

    async def __anext__(self):
        """
        Receive next IP and add to accumulator.

        Accumulator passes IPs through so they can be further processed
        downstream, while also storing them in the buffer.

        Returns:
            The IP (passed through)

        Raises:
            StopAsyncIteration: When upstream is exhausted
        """
        ip = await super().__anext__()

        # Accumulate the IP
        self._accumulator.append(ip)

        return ip


# ============================================================================
# Phase 3: Observable Process - Base class for span-tracked, event-emitting processes
# ============================================================================


class ObservableProcess(Process):
    """
    Base class for processes that emit events and track execution spans.

    ObservableProcess extends Process with:
    - Context integration (span tracking)
    - Event emission (start, value, stop, error events)
    - Execution path tracking (parent/child relationships)
    - Dependency injection support
    - Event streaming via FlowRunner

    This is the base class for StreamController and PipeController.
    """

    def __init__(
        self,
        gen_func: Callable[..., AsyncGenerator],
        name: str,
        span_type: "SpanType" = "component",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None,
        should_log_inputs: bool = True
    ):
        super().__init__(upstream)
        """
        Initialize ObservableProcess.

        Args:
            gen_func: The generator function to wrap
            name: Name for the span
            span_type: Type of span (e.g., "stream", "component")
            tags: Optional tags for the span
            args: Positional arguments to pass to gen_func
            kwargs: Keyword arguments to pass to gen_func (dependencies auto-resolved)
            upstream: Optional upstream process
        """
        super().__init__(upstream)
        self._gen_func = gen_func
        self._name = name
        self._span_type = span_type
        self._tags = tags or []
        self._args = args
        self._kwargs = kwargs or {}
        # self._span: "ExecutionSpan | None" = None
        self._data_flow: "DataFlowNode | None" = None
        self.resolved_kwargs: dict[str, Any] = {}
        self.index: int = 0  # Set by parent PipeController
        self.parent: "ObservableProcess | None" = None  # Set by parent PipeController
        self._replay_inputs: list | None = None  # Saved inputs for replay mode
        self._replay_outputs: list | None = None  # Saved outputs for replay mode
        self._replay_index: int = 0  # Current position in replay buffer
        
        self._start_event_type: str = f"{self._span_type}_start"
        self._value_event_type: str = f"{self._span_type}_value"
        self._stop_event_type: str = f"{self._span_type}_stop"
        self._error_event_type: str = f"{self._span_type}_error"
        self._should_log_inputs = should_log_inputs
        self._ctx: "Context | None" = None
        self._load_eval_args: bool = False
        self._input_data_flows: dict[str, "DataFlowNode"] = {}
        
        
    @property
    def ctx(self):
        """Get current context."""
        from .context import Context
        if self._ctx:
            return self._ctx
        ctx = Context.current_or_none()
        if ctx is None:
            raise ValueError("StreamController requires Context. Use 'async with Context():'")
        return ctx
    
    @property
    def name(self):
        """Get the name."""
        return self._name

    @property
    def span(self):
        """Get the execution span."""
        if self._data_flow is None:
            raise ValueError("Span is not initialized")
        return self._data_flow.value
    
    
    @property
    def path(self):
        """Get the path."""
        if self._data_flow is None:
            raise ValueError("Path is not initialized")
        return self._data_flow.path
    
    @property
    def path_or_none(self):
        """Get the path or None."""
        if self._data_flow is None:
            return None
        return self._data_flow.path

    @property
    def span_id(self):
        """Get the span ID."""
        return self.span.id if self.span else None
    
    
    def get_output(self, idx: int | None = None) -> Any:
        raise NotImplementedError("get_output is not implemented")

    async def _resolve_dependencies(self):
        """
        Resolve dependencies using injector.py and log as inputs.

        Returns:
            Tuple of (bound_arguments, resolved_kwargs)
        """
        from .injector import resolve_dependencies_kwargs
        from ..llms import LLM, LlmConfig
        from ..evaluation.decorators import EvalCtx
        
        
        self._data_flow = await self.ctx.start_span(
            component=self,
            name=self._name,
            span_type=self._span_type,
            tags=self._tags
        )
        if self._load_eval_args:
            self._replay_inputs = self.ctx.eval_ctx.get_eval_turn_inputs()
            bound, kwargs = await resolve_dependencies_kwargs(
                self._gen_func,
                args=self._replay_inputs,
                kwargs={}
            )
            self.resolved_kwargs = kwargs
        elif self.span.inputs:            
            self._replay_inputs = [v.value for v in self.span.inputs]
            bound, kwargs = await resolve_dependencies_kwargs(
                self._gen_func,
                args=self._replay_inputs,
                kwargs={}
            )
            self.resolved_kwargs = kwargs
        else:   
            bound, kwargs = await resolve_dependencies_kwargs(
                self._gen_func,
                args=self._args,
                kwargs=self._kwargs
            )
            self.resolved_kwargs = kwargs

            # Log resolved kwargs as inputs
            if self.span and self._should_log_inputs:
                for key, value in kwargs.items():
                    if value is not None and type(value) not in [LLM, LlmConfig, EvalCtx]:
                        data_flow = await self.span.log_value(value, io_kind="input", name=key)
                        self._input_data_flows[data_flow.path] = data_flow                    
        if self.span.outputs and not self.span.need_to_replay:
            self._replay_outputs = [v.value for v in self.span.outputs]

        return bound, kwargs

    async def on_stop(self):
        """
        Mark span as completed when process exhausts.
        """
        if self.span:
            self.span.status = "completed"
            self.span.end_time = __import__('datetime').datetime.now()
            await self.span.save()

        # Pop from context execution stack
        if self.ctx:
            await self.ctx.end_span()

    async def on_error(self, error: Exception):
        """
        Mark span as failed on error.
        """
        if self.span:
            self.span.status = "failed"
            self.span.end_time = __import__('datetime').datetime.now()
            await self.span.save()

        # Pop from context execution stack
        if self.ctx:
            await self.ctx.end_span()

    # Event emission methods

    async def on_start_event(self, payload: Any = None, attrs: dict[str, Any] | None = None):
        """
        Generate start event.

        Returns:
            StreamEvent with type="stream_start" or "span_start"
        """
        from .events import StreamEvent

        # Build attrs from resolved_kwargs (excluding Context and other non-serializable items)
        value_attrs = {}
        for key, value in self.resolved_kwargs.items():
            if value is not None and not key.startswith('_'):
                # Skip Context instances
                if self.ctx and isinstance(value, type(self.ctx)):
                    continue
                value_attrs[key] = value

        
        return self.ctx.build_event(
            path=self.path,
            kind=self._start_event_type,
            payload={
                self.path: self._data_flow,
                **self._input_data_flows,
            },
            name=self._name
        )
        return StreamEvent(
            type=self._start_event_type,
            name=self._name,
            attrs=value_attrs,
            payload=self.span,
            span_id=str(self.span_id) if self.span_id else None,
            path=self.get_execution_path(),
            value=self.span.parent_value
        )

    async def on_value_event(self, payload: Any = None):
        """
        Generate value event.

        Returns:
            StreamEvent with type="stream_delta" or "span_event"
        """
        from .events import StreamEvent
        
        
        return self.ctx.build_event(
            path=self.path,
            kind=self._value_event_type,
            payload=payload,
            name=self._name
        )

        return StreamEvent(
            type=self._value_event_type,
            name=self._name,
            payload=payload,
            # span_id=str(self.span_id) if self.span_id else None,
            # path=self.get_execution_path(),
        )

    async def on_stop_event(self, payload: Any = None):
        """
        Generate stop event.

        Returns:
            StreamEvent with type="stream_end" or "span_end"
        """
        from .events import StreamEvent
        
        
        return self.ctx.build_event(
            path=self.path,
            kind=self._stop_event_type,
            payload=payload,
            name=self._name
        )

        return StreamEvent(
            type=self._stop_event_type,
            name=self._name,
            span_id=str(self.span_id) if self.span_id else None,
            path=self.get_execution_path(),
        )

    async def on_error_event(self, error: Exception):
        """
        Generate error event.

        Args:
            error: The exception that occurred

        Returns:
            StreamEvent with type="stream_error" or "span_error"
        """
        from .events import StreamEvent
        
        
        return self.ctx.build_event(
            path=self.path_or_none,
            kind=self._error_event_type,
            payload=error,
            name=self._name,
            error=str(error)
        )

        return StreamEvent(
            type=self._error_event_type,
            name=self._name,
            payload=error,
            error=str(error),
            span_id=str(self.span_id) if self.span_id else None,
            path=self.get_execution_path(),
        )

    def get_execution_path(self) -> list[int]:
        """
        Build execution path from root to this component.

        Returns:
            list[int]: Path of indices from root
        """
        path = []
        current = self
        while current:
            if hasattr(current, 'index'):
                path.insert(0, current.index)
            current = current.parent if hasattr(current, 'parent') else None
        return path

    def stream(self, event_level=None, ctx: "Context | None" = None, load_eval_args: bool = False) -> "FlowRunner":
        """
        Return a FlowRunner that streams events from this process.

        This enables event streaming directly from decorated functions:
            @stream()
            async def my_stream():
                yield "hello"

            async for event in my_stream().stream():
                print(event)

        Args:
            event_level: EventLogLevel.chunk, .span, or .turn

        Returns:
            FlowRunner configured to emit events
        """
        from .flow_components import EventLogLevel
        if event_level is None:
            event_level = EventLogLevel.chunk
        self._ctx = ctx
        self._load_eval_args = load_eval_args
        return FlowRunner(self, ctx=ctx, event_level=event_level).stream_events()
    
    
    def get_response(self):
        """
        Get the response/result from this process.

        For most processes, this is just the last value.
        Subclasses can override to provide more sophisticated response handling.
        """
        if isinstance(self._last_ip, ObservableProcess):
            return self._last_ip.get_response()
        return self._last_ip
    
    

    




# ============================================================================
# Phase 3: Observable Composite - StreamController
# ============================================================================


class StreamController(ObservableProcess):
    """
    StreamController wraps a generator function and manages its execution span.

    Unlike Stream which wraps a generator instance, StreamController wraps
    the generator FUNCTION. This allows logging the function's kwargs as inputs
    and enables dependency injection from Context.

    Uses the existing dependency injection system from injector.py to resolve
    function arguments based on type annotations and Context state.

    Example:
        >>> async def my_gen(count: int):
        ...     for i in range(count):
        ...         yield i
        >>>
        >>> ctx = Context()
        >>> async with ctx:
        ...     controller = StreamController(
        ...         gen_func=my_gen,
        ...         name="counter",
        ...         span_type="stream",
        ...         kwargs={"count": 5}
        ...     )
        ...     async for value in controller:
        ...         print(value)
    """

    def __init__(
        self,
        gen_func: Callable[..., AsyncGenerator],
        name: str,
        span_type: "SpanType" = "stream",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None
    ):
        """
        Initialize StreamController.

        Args:
            gen_func: The generator function to wrap (NOT an instance)
            name: Name for the span
            span_type: Type of span (typically "stream")
            tags: Optional tags for the span
            args: Positional arguments to pass to gen_func
            kwargs: Keyword arguments to pass to gen_func (dependencies auto-resolved)
            upstream: Optional upstream process (usually None for StreamController)
        """
        from ..block import XmlParser
        super().__init__(gen_func, name, span_type, tags, args, kwargs, upstream)
        self._stream: Process | None = None
        self._accumulator: Accumulator | None = None
        self._parser: XmlParser | None = None
        self._gen: AsyncGenerator | None = None
        self._save_filepath: str | None = None
        self._load_filepath: str | None = None
        self._stream_value: DataFlow | None = None
        self._value_event_type: str = f"{self._span_type}_delta"
        self._load_delay: float | None = None
        self._temp_data_flow: "DataFlowNode | None" = None        


    def get_output(self, idx: int | None = None, include_fence: bool = False) -> Any:
        """
        Get the output from this stream controller.

        Args:
            idx: Not used, kept for interface compatibility.
            include_fence: If True, include markdown code fences in result.
                          If False (default), return just the XML content.

        Returns:
            The parsed result or accumulated result.
        """
        if self._parser is not None:
            return self._parser.get_result(include_fence=include_fence)
        if self._accumulator is not None:
            return self._accumulator.result
        return None
    
    def _init_stream(self, args: tuple, kwargs: dict):
        gen_instance = self._gen_func(*args, **kwargs)
        stream = Stream(gen_instance, name=f"{self._name}_stream")
        return stream
    
    
    def _load_cache(self, bound):
        load_filepath = self._load_filepath
        if self.ctx is not None:
            if load_filepath is None:
                load_filepath = self.ctx.load_cache.get(self._name)
        if load_filepath is None and self._save_filepath is None:
            cache_dir = self.ctx.cache_dir
            if cache_dir is not None:
                # Compute cache key from bound arguments
                cache_filename = compute_stream_cache_key(self._name, dict(bound.arguments))
                cache_path = os.path.join(cache_dir, cache_filename)

                # Check if cache exists
                if os.path.exists(cache_path):
                    # Load from cache
                    load_filepath = cache_path
                else:
                    # Save to cache
                    self._save_filepath = cache_path
        return load_filepath
    
    def _handle_save_cache(self, stream: Stream):
        if self._save_filepath is not None:
            os.makedirs(os.path.dirname(self._save_filepath), exist_ok=True)
            if os.path.exists(self._save_filepath):
                os.remove(self._save_filepath)
            stream.save_stream(self._save_filepath)
        

    
    async def on_start(self):
        """
        Build subnetwork and register with context.

        Called automatically on first __anext__() call.
        Creates span, resolves dependencies, logs kwargs as inputs, and builds the internal stream.

        In replay mode (when span has saved outputs), sets up replay buffer instead of
        executing the generator function.

        Auto-cache behavior:
        - If ctx.cache_dir is set and no explicit load/save path is configured:
          - Computes cache key from stream name + bound arguments
          - If cache file exists: loads from cache (no re-execution)
          - If cache file doesn't exist: executes and saves to cache
        """
        from ..block import Block
        bound, kwargs = await self._resolve_dependencies()
        
        load_filepath = self._load_cache(bound)
        if load_filepath is not None:
            stream = Stream.load(load_filepath, delay=self._load_delay or 0.05)
        else:
            gen_instance = self._gen_func(*bound.args, **bound.kwargs)
            stream = Stream(gen_instance, name=f"{self._name}_stream")

        self._handle_save_cache(stream)

        self._stream = stream
        self._gen = stream
        if self._parser is not None:
            self._gen |= self._parser
        else:
            self._accumulator = Accumulator(Block())
            self._gen |= self._accumulator
        
        
    async def on_value_event(self, payload: Any = None):
        """
        Generate value event.

        Returns:
            StreamEvent with type="stream_delta" or "span_event"
        """
        from .events import StreamEvent
        from ..versioning.artifact_log import ArtifactLog
        from ..block.block12.parsers import ParserEvent

        
        if type(payload) == ParserEvent: 
            print(payload.type, payload.path)    
            # if payload.type == "block_stream":
            # if payload.type == "block_init":
            #     print(payload.type, payload.value.path)
            if self._temp_data_flow is None:                    
                if self._parser is None:
                    if self._accumulator is None:
                        raise FlowException("Accumulator is not initialized")
                    payload = self._accumulator.result

                data_flow = ArtifactLog.build_data_flow_node(self.span, payload.value, io_kind="output")
                self._temp_data_flow = data_flow
                return self.ctx.build_event(
                    path=self._temp_data_flow.path,
                    kind=f"{self._span_type}_stream",
                    payload=data_flow,
                    name=self._name,
                    
                )            
        
            return self.ctx.build_event(
                path=self._temp_data_flow.path,
                kind=self._value_event_type,
                payload=payload,
                name=self._name
            )
            
        elif self._temp_data_flow is None:
            if self._parser is None:
                if self._accumulator is None:
                    raise FlowException("Accumulator is not initialized")
                payload = self._accumulator.result

            data_flow = ArtifactLog.build_data_flow_node(self.span, payload, io_kind="output")
            self._temp_data_flow = data_flow
            return self.ctx.build_event(
                path=self._temp_data_flow.path,
                kind=f"{self._span_type}_value",
                payload=data_flow,
                name=self._name,
                
                
            )


        
        return self.ctx.build_event(
            path=self._temp_data_flow.path,
            kind=self._value_event_type,
            payload=payload,
            name=self._name
        )

        
    def parse(self, block_schema: "Block"):
        from ..block import XmlParser
        from .context import Context
        if self._parser is not None:
            raise FlowException("Parser already initialized")
        if self._gen_func is None:
            raise FlowException("StreamController is not initialized")
        ctx = Context.current_or_none()
        verbose = False
        if ctx is not None:
            verbose = ctx.get_verbosity("parser")
        self._parser = XmlParser(block_schema, verbose=verbose)      
        return self
    
    # def name(self, name: str):
    #     self._name = name
    #     return self


    async def __anext__(self):
        """
        Delegate to internal stream process or replay from saved outputs.

        In replay mode, yields saved outputs from the span instead of executing
        the generator function.
        """
        from ..block.block12.parsers import ParserEvent
        if not self._did_start:
            await self.on_start()
            self._did_start = True
            return ParserEvent(path="0", type="block_stream", value=[])

        try:
            # Check if we're in replay mode
            if self._replay_outputs is not None:
                # Replay mode: yield from saved outputs
                if self._replay_index >= len(self._replay_outputs):
                    await self.on_stop()
                    raise StopAsyncIteration

                ip = self._replay_outputs[self._replay_index]
                self._replay_index += 1
                self._last_ip = ip

                if not self._did_yield:
                    self._did_yield = True

                return ip
            else:
                # Normal mode: get next IP from stream (which passes through Accumulator)
                ip = await self._gen.__anext__()
                self._last_ip = ip

                # Don't log individual values - they're stream deltas
                # We'll log the final accumulated result in on_stop()

                if not self._did_yield:
                    self._did_yield = True

                return ip
        except StopAsyncIteration:
            # Log the final accumulated result as output
            # if self._span_tree and self._accumulator:
                # await self._span_tree.log_value(self._accumulator.result, io_kind="output")
            if self.span:
                value = None
                if self._parser and self._parser.result is not None:
                    value = await self.span.log_value(self._parser.result, io_kind="output")
                elif self._accumulator:
                    # raise FlowException("Accumulator is not supported for StreamController")
                    value = await self.span.log_value(self._accumulator.result, io_kind="output")
                self._stream_value = value

            await self.on_stop()
            raise StopAsyncIteration
        except Exception as e:
            await self.on_error(e)
            raise e

    def get_response(self):
        """
        Get the accumulated response from the stream.

        In replay mode, returns all replay outputs.
        In normal mode, returns accumulated values.

        Returns:
            List of all values yielded by the stream
        """
        if self._replay_outputs is not None:
            # Replay mode: return all replay outputs
            return self._replay_outputs
        # elif self._accumulator:
            # Normal mode: return accumulated values
            # return self._accumulator.result
        # if self._stream_value is not None:
        #     if self._stream_value.kind == "block":
        #         return self._stream_value.value._block
        #     return self._stream_value.value
        if self._parser:
            return self._parser.result
        if self._accumulator:
            return self._accumulator.result
        # return self.acc
        return None
    
    def save(self, filename: str):
        if self._load_filepath is not None:
            raise FlowException("StreamController is already loaded")
        self._save_filepath = filename
        return self
    
    def load(self, filename: str, delay: float = 0.0):
        if self._save_filepath is not None:
            raise FlowException("StreamController is already saved")
        self._load_filepath = filename
        self._load_delay = delay
        return self
    
    async def athrow(self, typ, val=None, tb=None):
        if self._stream is None:
            raise FlowException("StreamController is not initialized")
        return await self._stream.athrow(typ, val, tb)

    async def aclose(self):
        if self._stream is None:
            raise FlowException("StreamController is not initialized")
        return await self._stream.aclose()




# ============================================================================
# Phase 4: Dynamic Composite - PipeController
# ============================================================================


class PipeController(ObservableProcess):
    """
    PipeController wraps a generator function that yields other processes.

    This is a COMPOSITE process in FBP terminology - it coordinates multiple
    child processes. Unlike StreamController which yields data IPs, PipeController
    yields other Process instances (StreamController, other PipeControllers, etc.).

    Key responsibilities:
    - Create span via Context.start_span()
    - Resolve dependencies from Context
    - Track index for each child process
    - Set parent reference on child processes
    - Emit span events for orchestration

    Example:
        >>> async def my_pipe():
        ...     stream1 = StreamController(gen1, "stream1", "stream")
        ...     yield stream1
        ...
        ...     stream2 = StreamController(gen2, "stream2", "stream")
        ...     yield stream2
        >>>
        >>> ctx = Context()
        >>> async with ctx:
        ...     pipe = PipeController(
        ...         gen_func=my_pipe,
        ...         name="my_pipe",
        ...         span_type="component"
        ...     )
        ...     async for child in pipe:
        ...         print(f"Child: {child._name}")
    """

    def __init__(
        self,
        gen_func: Callable[..., AsyncGenerator],
        name: str,
        span_type: "SpanType" = "component",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None,
        need_ctx: bool = True
    ):
        """
        Initialize PipeController.

        Args:
            gen_func: The generator function that yields processes
            name: Name for the span
            span_type: Type of span (typically "component")
            tags: Optional tags for the span
            args: Positional arguments to pass to gen_func
            kwargs: Keyword arguments to pass to gen_func (dependencies auto-resolved)
            upstream: Optional upstream process (usually None for PipeController)
        """
        super().__init__(gen_func, name, span_type, tags, args, kwargs, upstream)
        self._gen: AsyncGenerator | None = None
        self._last_value: DataFlow | None = None
        self._output_data_flows: list[DataFlow] = []
        
        
    def get_output(self, idx: int | None = None) -> Any:
        if idx is None:
            return self._output_data_flows[-1]
        else:
            return self._output_data_flows[idx]
        

    async def on_start(self):
        """
        Build subnetwork and register with context.

        Called automatically on first __anext__() call.
        Creates span, resolves dependencies, and builds the internal generator.

        In replay mode (when span has saved outputs), sets up replay buffer instead of
        executing the generator function.
        """
        
        bound, kwargs = await self._resolve_dependencies()
        # Call generator function with resolved kwargs
        self._gen = self._gen_func(*bound.args, **bound.kwargs)
        
        
    async def on_value_event(self, value: Any):
        if isinstance(value, StreamController):
            return None
        elif isinstance(value, PipeController):
            return None
        else:
            return await super().on_value_event(value)

    async def asend(self, value: Any = None):
        """
        Send a value into the pipe's internal generator or replay from saved outputs.

        This enables yield-based communication where parent can receive
        responses from completed child processes:
            response = yield child_process

        In replay mode, yields saved child processes from the span instead of
        executing the generator function.

        Args:
            value: The value to send (typically response from completed child)

        Returns:
            Next child process from the generator or replay buffer
        """
        from ..versioning import ExecutionSpan
        if not self._did_start:
            await self.on_start()
            self._did_start = True

        try:
            # Check if we're in replay mode
            if self._replay_outputs is not None:
                # Replay mode: yield from saved outputs
                if self._replay_index >= len(self._replay_outputs):
                    await self.on_stop()
                    raise StopAsyncIteration

                child = self._replay_outputs[self._replay_index]
                self._replay_index += 1
                self._last_ip = child
                self._output_data_flows.append(child)
                if not self._did_yield:
                    self._did_yield = True
                    
                # if isinstance(child, ExecutionSpan):
                while isinstance(child, ExecutionSpan):
                    # child_span = self.ctx.
                    print(f"child is ExecutionSpan {child.id}")
                    path = [int(i) for i in child.path.split(".")]
                    child_span = self.ctx.get_span(path[1:])
                    if child_span is None:
                        raise ValueError(f"Child span not found for {child.path}")
                    child = child_span.outputs[-1].value

                return child
            else:
                # Normal mode: send value into the internal generator using asend()
                child = await self._gen.asend(value)
                self._last_ip = child
                self._output_data_flows.append(child)

                # Set parent reference and index on child
                if isinstance(child, (StreamController, PipeController)):
                    child.parent = self
                    child.index = self.index
                    self.index += 1  # Increment for next child
                elif self.span:
                    value = await self.span.log_value(child, io_kind="output")
                    self._last_value = value

                # Log child as output
                if self.span and isinstance(child, (StreamController, PipeController)):
                    # Log the child's span tree once it's created
                    # Note: child span is created when child.on_start() is called by FlowRunner
                    pass

                if not self._did_yield:
                    self._did_yield = True

                return child
        except StopAsyncIteration:
            await self.on_stop()
            raise StopAsyncIteration
        except Exception as e:
            await self.on_error(e)
            raise e

    async def __anext__(self):
        """
        Get next child process from the pipe.

        Returns child processes and tracks their index.
        Sets parent reference on children.
        """
        # Delegate to asend with None (equivalent to __anext__)
        return await self.asend(None)
    
    
    async def athrow(self, typ, val=None, tb=None):
        if self._gen is None:
            raise FlowException("Process is not initialized")
        return await self._gen.athrow(typ, val, tb)

    async def aclose(self):
        if self._gen is None:
            raise FlowException("Process is not initialized")
        return await self._gen.aclose()






class EvaluatorController(ObservableProcess):
    """
    EvaluationController is a process that evaluates values based on evaluation context.
    """
    def __init__(
        self, 
        gen_func: Callable[..., AsyncGenerator],
        name: str,
        span_type: "SpanType" = "evaluator",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None
    ):
        super().__init__(gen_func, name, span_type, tags, args, kwargs, upstream, should_log_inputs=False)
        self._gen: AsyncGenerator | None = None
        self._did_start = False
        self._did_yield = False
        self._last_ip = None
        self._last_value = None
        self._parent = upstream
        self._index = 0 if upstream else None
    
    
    async def on_start(self):
        bound, kwargs = await self._resolve_dependencies()
        # Call generator function with resolved kwargs
        # self._gen = self._gen_func(*bound.args, **bound.kwargs)
    
    
    async def on_stop(self):
        return None
    
    async def on_error(self, error: Exception):
        return None

    async def asend(self, value: Any = None):
        return await self._gen_func(**self.resolved_kwargs)
    
    async def __anext__(self):
        return await self._gen_func(**self.resolved_kwargs)
    
    
    

class TurnController(Process):
    
    
    def __init__(self, agent_component: Callable[..., AsyncGenerator], ctx: "Context", message: "Block", name: str):
        async def gen():
            async with ctx:
                yield agent_component(message)
            
        super().__init__(upstream=agent_component(message))
        self.name = name
        self._ctx = ctx
        
    
    

    # async def on_start(self, value: Any = None):
    #     await self._ctx.__aenter__()
    #     return None


    # async def on_stop(self):
    #     await self._ctx.__aexit__(None, None, None)
    #     # return None

    # async def on_error(self, error: Exception):
    #     await self._ctx.__aexit__(type(error), error, None)
    #     return None
    
    def stream(self, event_level=None, ctx: "Context | None" = None, load_eval_args: bool = False) -> "FlowRunner":
        """
        Return a FlowRunner that streams events from this process.

        This enables event streaming directly from decorated functions:
            @stream()
            async def my_stream():
                yield "hello"

            async for event in my_stream().stream():
                print(event)

        Args:
            event_level: EventLogLevel.chunk, .span, or .turn

        Returns:
            FlowRunner configured to emit events
        """
        from .flow_components import EventLogLevel
        if event_level is None:
            event_level = EventLogLevel.chunk
        if ctx is not None:
            self._ctx = ctx
        # self._load_eval_args = load_eval_args
        return FlowRunner(self, ctx=self._ctx, event_level=event_level).stream_events()
        # return FlowRunner(self, event_level=event_level, need_ctx=False).stream_events()

# ============================================================================
# FlowRunner - Orchestrates nested process execution
# ============================================================================


class FlowRunner:
    """
    FlowRunner orchestrates execution of nested FBP processes.

    Manages a stack of processes and handles:
    - Pushing child processes to stack when yielded by PipeController
    - Emitting events based on event_level
    - Propagating errors through the stack
    - Coordinating lifecycle hooks

    Example:
        >>> pipe = PipeController(my_pipe, "main", "component")
        >>> runner = FlowRunner(pipe, event_level=EventLogLevel.chunk)
        >>>
        >>> async for event in runner.stream_events():
        ...     print(f"Event: {event.type} - {event.name}")
    """

    def __init__(self, root_process: Process, ctx: "Context | None" = None, event_level=None, need_ctx: bool = True):
        """
        Initialize FlowRunner.

        Args:
            root_process: The root process to execute (usually PipeController)
            event_level: Level of events to emit (chunk, span, turn)
        """
        from .context import Context
        self.stack: list[Process] = [root_process]
        self.last_value: Any = None
        self._output_events = False
        self._error_to_raise: Exception | None = None
        self._last_process: Process | None = None
        self._event_level = event_level
        self._pending_child: Process | None = None  # Child process waiting to be pushed
        self._response_to_send: Any = None  # Response from child to send to parent
        self._exited_processes: list[ObservableProcess] = []
        self.ctx = ctx or Context.current_or_none()
        self._need_ctx = need_ctx

    @property
    def current(self) -> Process:
        """Get current process from stack."""
        if not self.stack:
            raise StopAsyncIteration
        return self.stack[-1]
    
    @property
    def result(self) -> Any:
        """Get result from root process."""
        return self.get_output()

    @property
    def should_output_events(self) -> bool:
        """Check if events should be emitted."""
        return self._output_events

    def push(self, process: Process):
        """Push process onto stack."""
        self.stack.append(process)

    def pop(self) -> Process:
        """Pop process from stack."""
        process = self.stack.pop()
        self._last_process = process
        self._exited_processes.append(process)
        return process
    
    async def revert(self, reason: str | None = None):
        """Revert the stack to the last process."""
        if not self.ctx.turn:
            raise ValueError("Context turn is not set")
        await self.ctx.turn.revert(reason)
    
    
    def get_output(self, name: str | None = None) -> Any:
        if name is None:
            return self._exited_processes[-1].get_response()
        else:
            for process in self._exited_processes:
                if process.name == name:
                    return process.get_output()
            return None

    def _get_response(self) -> Any:
        """
        Get the response to send to the next process.

        When a child process completes, we send its response to the parent.
        This mimics the behavior of: response = yield child_process
        """
        if self._response_to_send is not None:
            response = self._response_to_send
            self._response_to_send = None
            return response
        return None
    
    def get_span(self, name: str) -> "ExecutionSpan":
        for process in self._exited_processes:
            if process.name == name:
                return process.span
        raise ValueError(f"ExecutionSpan {name} not found")

    def get_llm_call(self, name: str, index: int | None = None) -> "LlmCall":
        span = self.get_span(name)
        index = index or -1
        if span is None:
            raise ValueError(f"Span {name} not found")
        if span.span_type != "llm":
            raise ValueError(f"Span {name} is not an llm")
        if index >= 0 and index >= len(span.llm_calls):
            raise ValueError(f"Index {index} out of range for span {name}")
        return span.llm_calls[index]

    def __aiter__(self):
        """Make FlowRunner async iterable."""
        return self
    
    
    async def enter_context(self):
        if not self.ctx:
            raise ValueError("Context is not set")
        if self.ctx.is_set():
            return self
        await self.ctx.__aenter__()
        return self

    async def exit_context(self, error: Exception | None = None):
        if not self.ctx:
            raise ValueError("Context is not set")
        if not self.ctx.is_set():
            return self
        if error:
            await self.ctx.__aexit__(type(error), error, None)
        else:
            await self.ctx.__aexit__(None, None, None)
        return self

    async def __anext__(self):
        """
        Execute next step in process network.

        Returns:
            StreamEvent if emitting events, otherwise the value from current process
        """
        
        await self.enter_context()
        while self.stack:
            try:
                # First, check if we have a pending child to push
                if self._pending_child is not None:
                    child = self._pending_child
                    self._pending_child = None
                    self.push(child)
                    # Continue to start the child process
                    continue

                process = self.current

                # Handle error propagation
                if self._error_to_raise:
                    error = self._error_to_raise
                    self._error_to_raise = None
                    raise error

                # Check if process is starting - emit start event before first value
                if not process._did_start:
                    # Start the process and get first value
                    await process.on_start()
                    process._did_start = True

                    # Now emit start event if needed (before getting first value)
                    if self.should_output_events:
                        if event := await self.try_build_start_event(process, None):
                            return event

                # Get next value using asend (to support yield-based communication)
                response = self._get_response()
                value = await process.asend(response)
                self.last_value = value

                # Trigger evaluation if context has evaluation enabled
                

                # If value is a Process (from PipeController), push to stack
                if isinstance(value, (StreamController, PipeController)):
                    if isinstance(value, StreamController):
                        value._name = process._name + "_" + value._name
                    self.push(value)
                    # Emit event for child process if needed
                    if self.should_output_events:
                        if event := await self.try_build_value_event(process, value):
                            return event
                    continue
                
                # if not isinstance(process, EvaluatorController):
                #     await self._try_evaluate_value(process)

                # Emit value event if needed
                if self.should_output_events:
                    if event := await self.try_build_value_event(process, value):
                        return event

                # If not emitting events, return the value
                if not self.should_output_events:
                    return value

            except StopAsyncIteration:
                # Process exhausted, pop from stack
                process = self.pop()
                                
                

                # Get the response from the completed process
                if hasattr(process, 'get_response'):
                    response = process.get_response()
                else:
                    response = self.last_value
                    
                await self._try_evaluate_value(process)

                # If there's a parent waiting, save the response to send
                if self.stack:
                    self._response_to_send = response

                # Emit stop event if needed
                if self.should_output_events:
                    if event := await self.try_build_stop_event(process, self.last_value):
                        return event

            except Exception as e:
                # Error occurred, pop from stack
                
                try:
                    await self.current.athrow(e)
                except Exception as sub_ex:
                    pass
                
                
                process = self.pop()

                # Emit error event if needed
                if self.should_output_events:
                    event = await self.try_build_error_event(process, e)
                    if not self.stack:
                        await self.exit_context(e)
                        raise e
                    self._error_to_raise = e
                    return event
                else:
                    raise e
                
                
        # if not self.stack:
            # result = await self._commit_evaluation()
        
        await self.exit_context()
        raise StopAsyncIteration
    
    # async def _try_evaluate_value(self, process: Process):
    #     """Try to evaluate value based on evaluation context."""
    #     eval_ctx = self.ctx._evaluation_context
    #     value = process._last_value if hasattr(process, '_last_value') else None
    #     if eval_ctx is None or value is None:
    #         return
        
    #     evaluator_controllers = eval_ctx.build_evaluator_controllers(value)
    #     for evaluator_controller in reversed(evaluator_controllers):
    #         self.push(evaluator_controller)        
    #     return value
    
    async def _commit_evaluation(self):
        eval_ctx = self.ctx._evaluation_context
        if eval_ctx is None:
            return
        return await eval_ctx.commit()
    
    async def _try_evaluate_value(self, process: Process):
        """Try to evaluate value based on evaluation context."""
        eval_ctx = self.ctx._evaluation_context
        value = process._stream_value if hasattr(process, '_stream_value') else None
        # value = process._last_value if hasattr(process, '_last_value') else None
        if eval_ctx is None or value is None:
            return
        
        await eval_ctx.build_turn_eval_from_current()        
        for gen_func, ctx in eval_ctx.get_evaluator_handlers(value):
            ref_value = eval_ctx.get_ref_value(value.path)
            score, metadata = await gen_func(ctx, ref_value, value)
            value_eval = await eval_ctx.log_eval(value, score, metadata, ctx.config)
        



    async def try_build_start_event(self, process: Process, value: Any):
        """Try to build start event based on event_level."""
        if not hasattr(process, 'on_start_event'):
            return None

        event = await process.on_start_event(value)
        if event is None:
            return None

        from .flow_components import EventLogLevel

        if self._event_level == EventLogLevel.chunk:
            return event
        elif self._event_level == EventLogLevel.span:
            return event
        elif self._event_level == EventLogLevel.turn:
            return None

        return None

    async def try_build_value_event(self, process: Process, value: Any):
        """Try to build value event based on event_level."""
        if not hasattr(process, 'on_value_event'):
            return None

        event = await process.on_value_event(value)
        if event is None:
            return None

        from .flow_components import EventLogLevel

        if self._event_level == EventLogLevel.chunk:
            return event
        elif self._event_level == EventLogLevel.span:
            if isinstance(value, (StreamController, PipeController)):
                return event
            return None

        return None

    async def try_build_stop_event(self, process: Process, value: Any):
        """Try to build stop event based on event_level."""
        if not hasattr(process, 'on_stop_event'):
            return None

        event = await process.on_stop_event(value)
        if event is None:
            return None

        from .flow_components import EventLogLevel

        if self._event_level == EventLogLevel.chunk:
            return event
        elif self._event_level == EventLogLevel.span:
            return event

        return None

    async def try_build_error_event(self, process: Process, error: Exception):
        """Try to build error event based on event_level."""
        if not hasattr(process, 'on_error_event'):
            return None

        event = await process.on_error_event(error)
        return event if self._event_level else None

    def stream_events(self, event_level=None):
        """Enable event emission mode."""
        if event_level is not None:
            self._event_level = event_level
        self._output_events = True
        return self

    # def print(self):

# # ============================================================================
# # Phase 5: Parser Integration
# # ============================================================================

# class ParserError(Exception):
#     pass


# class Parser(Process):
#     def __init__(self, schema: "BlockSchema"):
#         from xml.parsers import expat
#         from ..block import BlockChunk
#         super().__init__()
#         self.schema = schema
#         self.build_ctx = StreamingBlockBuilder(schema)
#         self.parser = expat.ParserCreate()
#         self.parser.buffer_text = False
#         self.parser.StartElementHandler = self._on_start
#         self.parser.EndElementHandler = self._on_end
#         self.parser.CharacterDataHandler = self._on_chardata
        
#         self.chunks = []  # (start_byte, end_byte, chunk)
#         self.total_bytes = 0
#         self.pending = None  # (event_type, event_data, start_byte)
#         self.chunk_queue = []
#         self._tag_path = []
#         self._has_synthetic_root_tag = False
#         if self.schema.is_wrapper:
#             self.feed(BlockChunk(content=f"<{self.schema.name}>"))
#             self._has_synthetic_root_tag = True
#         # self._root_tag_in_schema = not self.schema.is_wrapper
#         # self._synthetic_root_tag: str | None = "root" if self.schema.is_wrapper else None
        
#     @property
#     def result(self):
#         return self.build_ctx.result
    
#     def feed(self, chunk: "BlockChunk", isfinal=False):
#         from xml.parsers.expat import ExpatError
#         # print(chunk.content)
#         # data = chunk.content.encode() if isinstance(chunk.data, str) else chunk.data
#         data = chunk.content.encode("utf-8")
#         start = self.total_bytes
#         end = start + len(data)
#         self.chunks.append((start, end, chunk))
#         self.total_bytes = end
#         try:
#             self.parser.Parse(data, isfinal)
#         except ExpatError as e:
#             if e.code == 4:
#                 raise ParserError(f"Invalid XML token: {data}")
#             else:
#                 raise e

        
        
#     def _push_block(self, block: "BaseBlock"):
#         self.chunk_queue.append(block)
        
#     def _push_block_list(self, blocks: list["BaseBlock"]):
#         self.chunk_queue.extend(blocks)
        
#     def _pop_block(self):
#         return self.chunk_queue.pop(0)
    
#     def _has_outputs(self):
#         return len(self.chunk_queue) > 0
    
#     def close(self):
#         from ..block import BlockChunk
#         if self._has_synthetic_root_tag:
#             self.feed(BlockChunk(content=f"</{self.schema.name}>"))        
#         self.parser.Parse(b'', True)
#         # Flush any pending event
#         self._flush_pending(self.total_bytes)
    
#     def _get_chunks_in_range(self, start, end):
#         """
#         Return all chunks overlapping [start, end), splitting chunks at boundaries.

#         When a chunk partially overlaps the range, it is split so that only the
#         portion within [start, end) is returned. This handles cases where the LLM
#         returns mixed content like "Hello<tag>" in a single chunk.

#         The original chunks in self.chunks remain intact - only the returned
#         list contains the split portions.

#         Args:
#             start: Start byte position (inclusive)
#             end: End byte position (exclusive)

#         Returns:
#             List of BlockChunk objects, potentially split at byte boundaries
#         """
#         from ..block import BlockChunk

#         result = []
#         for chunk_start, chunk_end, chunk in self.chunks:
#             # Check if chunk overlaps with [start, end)
#             if chunk_start < end and chunk_end > start:
#                 # Calculate the overlap
#                 overlap_start = max(chunk_start, start)
#                 overlap_end = min(chunk_end, end)

#                 # Check if we need to split (chunk extends beyond the range)
#                 if overlap_start == chunk_start and overlap_end == chunk_end:
#                     # Full chunk is within range, no split needed
#                     result.append(chunk)
#                 else:
#                     # Need to split the chunk - extract only the overlapping portion
#                     content_bytes = chunk.content.encode("utf-8")

#                     # Calculate byte offsets relative to chunk start
#                     slice_start = overlap_start - chunk_start
#                     slice_end = overlap_end - chunk_start

#                     # Adjust slice boundaries to respect UTF-8 character boundaries
#                     # UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
#                     # Move slice_start forward to skip continuation bytes
#                     while slice_start < len(content_bytes) and (content_bytes[slice_start] & 0xC0) == 0x80:
#                         slice_start += 1

#                     # Move slice_end forward to include full character
#                     while slice_end < len(content_bytes) and (content_bytes[slice_end] & 0xC0) == 0x80:
#                         slice_end += 1

#                     # Extract the slice and decode back to string
#                     sliced_bytes = content_bytes[slice_start:slice_end]

#                     # Skip if slice is empty after boundary adjustment
#                     if not sliced_bytes:
#                         continue

#                     sliced_content = sliced_bytes.decode("utf-8")

#                     # Create a new chunk with the sliced content
#                     # Preserve logprob from original chunk
#                     split_chunk = BlockChunk(
#                         content=sliced_content,
#                         logprob=chunk.logprob,
#                         # prefix=chunk.prefix if slice_start == 0 else "",
#                         # postfix=chunk.postfix if slice_end == len(content_bytes) else "",
#                     )
#                     result.append(split_chunk)
#         return result
    
#     def _flush_pending(self, end_byte):
#         if self.pending is None:
#             return
        
#         event_type, event_data, start_byte = self.pending
#         chunks = self._get_chunks_in_range(start_byte, end_byte)
#         metas = [c.logprob for c in chunks]
#         # print(event_type)
#         if event_type == 'start':
#             name, attrs = event_data
#             block = self.build_ctx.open_view(name, chunks, attrs=attrs, ignore_style=True)
#             self._push_block(block)
#             # self._push_block_list(blocks)
#             # print(f"StartElement '{name}' {attrs or ''} from chunks: {metas}")
#         elif event_type == 'end':
#             view = self.build_ctx.close_view(chunks)
#             self._push_block(view)
#             # self._push_block(view.postfix)
#             # self.build_ctx.commit_view()
#             # print(f"EndElement '{event_data}' from chunks: {metas}")
#         elif event_type == 'chardata':
#             for chunk in chunks:
#                 cb = self.build_ctx.append(chunk)
#                 self._push_block(cb)
#             # print(f"CharData {repr(event_data)} from chunks: {metas}")
        
#         self.pending = None
    
#     def _on_start(self, name, attrs):
#         current_pos = self.parser.CurrentByteIndex
#         self._flush_pending(current_pos)
#         self._tag_path.append(name)
#         self.pending = ('start', (name, attrs), current_pos)
    
#     def _on_end(self, name):
#         current_pos = self.parser.CurrentByteIndex
#         self._flush_pending(current_pos)
#         self._tag_path.pop()
#         self.pending = ('end', name, current_pos)
    
#     def _on_chardata(self, data):
#         # print(f"chardata: '{data}'")
#         current_pos = self.parser.CurrentByteIndex
#         self._flush_pending(current_pos)
#         # For chardata we could compute end directly, but for consistency
#         # we'll use the deferred approach too
#         self.pending = ('chardata', data, current_pos)
        
        
#     async def on_stop(self):
#         self.close()
        
    
#     # async def __anext__(self):
#     #     while not self._has_outputs() and not self.did_upstream_end():
#     #         try:
#     #             value = await super().__anext__()
#     #             # print("anext", value.content)
#     #             self.feed(value)        
#     #         except Exception as e:
#     #             pass
#     #     if not self._has_outputs() and self.did_upstream_end():
#     #         raise StopAsyncIteration
#     #     block = self._pop_block()
#     #     return block
    
#     async def __anext__(self):
#         while not self._has_outputs():
#             value = await super().__anext__()
#             # print("anext", value.content)
#             self.feed(value)        
#         block = self._pop_block()
#         return block

# class Parser2(Process):
#     """
#     Parser process - transforms XML-tagged chunks into Block structures.
    
#     This is a STATEFUL TRANSFORMER in FBP - maintains parsing state,
#     buffers chunks, and yields Block events as XML tags are parsed.
    
#     The Parser uses lxml's XMLPullParser to incrementally parse streaming
#     XML content and build Block structures via BlockBuilderContext.
    
#     Example:
#         >>> response_schema = Block(text=str)
#         >>> parser = Parser(response_schema)
#         >>> 
#         >>> stream = Stream(chunk_gen())
#         >>> pipeline = stream | parser | acc
#         >>> async for block_event in pipeline:
#         ...     print(block_event)
#     """
    
#     def __init__(self, response_schema: "Block", upstream: Process | None = None):
#         """
#         Initialize Parser.
        
#         Args:
#             response_schema: Block schema defining expected structure
#             upstream: Upstream process providing BlockChunk IPs
#         """
#         super().__init__(upstream)
#         from lxml import etree
#         from ..block import BlockBuilderContext
        
#         self.start_tag = "tag_start"
#         self.end_tag = "tag_end"
#         self.text_tag = "chunk"
#         self._safety_tag = "stream_start"
#         self.res_ctx = BlockBuilderContext(response_schema.copy())
#         self.parser = etree.XMLPullParser(events=("start", "end"))
#         self.block_buffer = []
#         self._stream_started = False
#         self._detected_tag = False
#         self._total_chunks = 0
#         self._chunks_from_last_tag = 0
#         self._tag_stack = []
#         self._full_content = ""
#         self._event_queue = []  # Queue events to yield across multiple __anext__ calls
        
#     def _push_tag(self, tag: str, is_list: bool):
#         self._tag_stack.append({"tag": tag, "is_list": is_list})
    
#     def _pop_tag(self):
#         return self._tag_stack.pop()
    
#     @property
#     def current_tag(self):
#         if not self._tag_stack:
#             return None
#         return self._tag_stack[-1]["tag"]
    
#     @property
#     def current_tag_is_list(self):
#         if not self._tag_stack:
#             return False
#         return self._tag_stack[-1]["is_list"]
    
#     def _read_buffer(self, start_from: str | None = None, flush=True):
#         buffer = []
#         start_appending = start_from is None
#         for block in self.block_buffer:
#             if start_from and start_from in block.content:
#                 start_appending = True
#             if start_appending:
#                 buffer.append(block)
#         if flush:
#             self.block_buffer = []
#         return buffer
    
#     def _write_to_buffer(self, value: Any):
#         self._total_chunks += 1
#         self._chunks_from_last_tag += 1
#         self.block_buffer.append(value)
    
#     def _try_set_tag_lock(self, value: Any):
#         if "<" in value.content:
#             self._detected_tag = True
    
#     def _release_tag_lock(self):
#         self._chunks_from_last_tag = 0
#         self._detected_tag = False
    
#     def _should_output_chunk(self):
#         if self.current_tag and not self._detected_tag:
#             if self._chunks_from_last_tag < 2:
#                 return False
#             return True
#         return False
    
#     def _feed_parser(self, content: str):
#         self._full_content += content
#         self.parser.feed(content)
    
#     def _process_chunk(self, chunk):
#         """Process a chunk and add events to queue."""
#         if not self._stream_started:
#             self._feed_parser(f'<{self._safety_tag}>')
#             self._stream_started = True
        
#         # Check if res_ctx has queued events first
#         if self.res_ctx.has_events():
#             event = self.res_ctx.get_event()
#             self._event_queue.append(event)
        
#         # Process the chunk
#         self._write_to_buffer(chunk)
#         try:
#             self._feed_parser(chunk.content)
#         except Exception as e:
#             print(self._full_content)
#             print(f"Parser Error on content: {chunk.content}")
#             raise e
        
#         self._try_set_tag_lock(chunk)
        
#         # Output chunks in middle of stream
#         if self._should_output_chunk():
#             for c in self._read_buffer(flush=True):
#                 self.res_ctx.append(self.current_tag, c)
        
#         # Process XML events
#         for event, element in self.parser.read_events():
#             if element.tag == self._safety_tag:
#                 continue
            
#             if event == 'start':
#                 if self.current_tag_is_list:
#                     view, schema = self.res_ctx.instantiate_list_item(
#                         self.current_tag,
#                         element.tag,
#                         self._read_buffer(),
#                         attrs=dict(element.attrib),
#                     )
#                 else:
#                     view, schema = self.res_ctx.instantiate(
#                         element.tag,
#                         self._read_buffer(),
#                         attrs=dict(element.attrib),
#                     )
#                 self._push_tag(element.tag, schema.is_list)
#                 self._release_tag_lock()
            
#             elif event == 'end':
#                 self.res_ctx.set_view_attr(
#                     element.tag,
#                     postfix=self._read_buffer(start_from="</"),
#                 )
#                 self._pop_tag()
#                 self._release_tag_lock()
        
#         # Collect any new events
#         while self.res_ctx.has_events():
#             event = self.res_ctx.get_event()
#             self._event_queue.append(event)
    
#     async def __anext__(self):
#         """
#         Get next block event.
        
#         Parser may need to consume multiple chunks to produce one event,
#         or produce multiple events from one chunk.
#         """
#         # Process queued events first
#         if self._event_queue:
#             return self._event_queue.pop(0)
        
#         # Try to get more chunks and process them
#         max_iterations = 20000
#         for _ in range(max_iterations):
#             try:
#                 # Get next chunk from upstream
#                 chunk = await super().__anext__()
                
#                 # Process the chunk
#                 self._process_chunk(chunk)
                
#                 # If we now have events, return one
#                 if self._event_queue:
#                     return self._event_queue.pop(0)
            
#             except StopAsyncIteration:
#                 # Upstream exhausted
#                 # Return any remaining queued events first
#                 if self._event_queue:
#                     return self._event_queue.pop(0)
#                 # No more events, we're done
#                 raise StopAsyncIteration
#         else:
#             print("no more chunks")
        
#         # Hit max iterations without producing event
#         raise StopAsyncIteration


# # %%
