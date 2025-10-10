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

## Implementation Phases

### Phase 1: Core Process (MVP)
- [ ] Process base class with normal execution mode
- [ ] Pipe operator (|)
- [ ] Lifecycle hooks
- [ ] Basic tests

### Phase 2: Source & Sink
- [ ] Stream process (source)
- [ ] Accumulator process (sink)
- [ ] Pipe composition tests
- [ ] Stream persistence (save/load)

### Phase 3: Observable Composite
- [ ] StreamController with span tracking
- [ ] Context integration
- [ ] Event emission
- [ ] Integration tests

### Phase 4: Dynamic Composite & Orchestration
- [ ] PipeController
- [ ] FlowRunner (stack management)
- [ ] Nested process tests

### Phase 5: Parser Integration
- [ ] Parser process
- [ ] End-to-end pipeline tests
- [ ] Performance benchmarks

### Phase 6: Replay Foundation
- [ ] SpanTree I/O logging
- [ ] Replay mode in Process base
- [ ] Full replay from root span
- [ ] Replay tests

### Phase 7: Advanced Replay
- [ ] Partial replay (resume from specific span)
- [ ] Function registry for reconstruction
- [ ] Resume tests

## Success Criteria

1. Can compose simple pipeline: `stream | parser | accumulator`
2. Can execute nested pipelines with PipeController + StreamController
3. Can save execution to SpanTree and replay from start
4. All tests pass with 80%+ coverage
5. Performance within 10% of current flow_components.py

---

TASK BREAKDOWN
==============

## Phase 1: Core Process (MVP)

### Task 1.1: Process Base Class - Structure
**File**: fbp_process.py
**Playground**: research/fbp/fbp_process_playground.py (Cell 1)
**Estimate**: 30 min

**Requirements**:
- Create `Process` base class
- Constructor: `__init__(upstream: Process | None = None)`
- Properties: `upstream`, `_did_start`, `_last_ip`
- Method stubs: `on_start()`, `on_stop()`, `on_error()`

**Test in Playground**:
```python
# %%
# Cell 1: Basic Process instantiation
from promptview.prompt.fbp_process import Process

proc = Process()
assert proc._did_start == False
assert proc._last_ip is None
```

### Task 1.2: Process Base Class - Async Iteration
**File**: fbp_process.py
**Playground**: Cell 2
**Estimate**: 45 min

**Requirements**:
- Implement `__aiter__()` and `__anext__()`
- Call `on_start()` on first iteration
- Receive IP from upstream
- Store last IP
- Raise `StopAsyncIteration` when upstream exhausted

**Test in Playground**:
```python
# %%
# Cell 2: Async iteration protocol
async def test_iteration():
    # Create mock upstream that yields 3 values
    async def mock_gen():
        for i in range(3):
            yield i

    class MockUpstream:
        def __init__(self, gen):
            self.gen = gen
        def __aiter__(self):
            return self
        async def __anext__(self):
            return await self.gen.__anext__()

    upstream = MockUpstream(mock_gen())
    proc = Process(upstream=upstream)

    results = []
    async for ip in proc:
        results.append(ip)

    assert results == [0, 1, 2]
    assert proc._did_start == True

await test_iteration()
```

### Task 1.3: Process Base Class - Pipe Operator
**File**: fbp_process.py
**Playground**: Cell 3
**Estimate**: 20 min

**Requirements**:
- Implement `__or__()` operator
- Implement `connect()` method
- Set downstream's upstream to self
- Return downstream for chaining

**Test in Playground**:
```python
# %%
# Cell 3: Pipe composition
proc1 = Process()
proc2 = Process()
proc3 = Process()

result = proc1 | proc2 | proc3

assert proc2._upstream is proc1
assert proc3._upstream is proc2
assert result is proc3
```

### Task 1.4: Process Base Class - Lifecycle Hooks
**File**: fbp_process.py
**Playground**: Cell 4
**Estimate**: 30 min

**Requirements**:
- Implement `on_start()`, `on_stop()`, `on_error()`
- Call hooks at appropriate times in `__anext__()`
- Handle exceptions and call `on_error()`

**Test in Playground**:
```python
# %%
# Cell 4: Lifecycle hooks
class LifecycleTestProcess(Process):
    def __init__(self, upstream=None):
        super().__init__(upstream)
        self.started = False
        self.stopped = False
        self.errored = False

    async def on_start(self, value=None):
        self.started = True

    async def on_stop(self):
        self.stopped = True

    async def on_error(self, error):
        self.errored = True

# Test successful execution
async def test_lifecycle():
    async def gen():
        yield 1

    class MockUpstream:
        def __aiter__(self):
            return gen().__aiter__()
        async def __anext__(self):
            return await gen().__anext__()

    proc = LifecycleTestProcess(upstream=MockUpstream())

    async for _ in proc:
        pass

    assert proc.started == True

await test_lifecycle()
```

---

## Phase 2: Source & Sink Processes

### Task 2.1: Stream Process - Basic
**File**: fbp_process.py
**Playground**: Cell 5
**Estimate**: 30 min

**Requirements**:
- Create `Stream(Process)` class
- Constructor: `__init__(gen: AsyncGenerator)`
- Override `upstream` property to return generator
- No actual upstream (Stream is a source)

**Test in Playground**:
```python
# %%
# Cell 5: Stream as source process
from promptview.prompt.fbp_process import Stream

async def number_gen():
    for i in range(5):
        yield i

stream = Stream(number_gen())

results = []
async for ip in stream:
    results.append(ip)

assert results == [0, 1, 2, 3, 4]
```

### Task 2.2: Stream Process - Persistence
**File**: fbp_process.py
**Playground**: Cell 6
**Estimate**: 45 min

**Requirements**:
- Add `save_stream(filepath: str)` method
- Override `__anext__()` to write to file
- Add `load(filepath: str)` class method
- Return new Stream that reads from file

**Test in Playground**:
```python
# %%
# Cell 6: Stream persistence
import json
import os
from pydantic import BaseModel

class TestChunk(BaseModel):
    content: str
    index: int

async def chunk_gen():
    for i in range(3):
        yield TestChunk(content=f"chunk_{i}", index=i)

# Save stream
stream = Stream(chunk_gen())
stream.save_stream("/tmp/test_stream.jsonl")

results_save = []
async for chunk in stream:
    results_save.append(chunk)

# Load stream
stream_loaded = Stream.load("/tmp/test_stream.jsonl", model=TestChunk)

results_load = []
async for chunk in stream_loaded:
    results_load.append(chunk)

assert len(results_save) == len(results_load) == 3
assert results_save[0].content == results_load[0].content

os.remove("/tmp/test_stream.jsonl")
```

### Task 2.3: Accumulator Process
**File**: fbp_process.py
**Playground**: Cell 7
**Estimate**: 30 min

**Requirements**:
- Create `Accumulator(Process)` class
- Constructor: `__init__(accumulator=None, upstream=None)`
- Default accumulator is list
- Append each IP to accumulator
- Property `result` returns accumulator

**Test in Playground**:
```python
# %%
# Cell 7: Accumulator as sink
from promptview.prompt.fbp_process import Accumulator

async def test_accumulator():
    async def gen():
        for i in range(5):
            yield i

    stream = Stream(gen())
    acc = Accumulator(accumulator=[])

    pipeline = stream | acc

    async for _ in pipeline:
        pass

    assert acc.result == [0, 1, 2, 3, 4]

await test_accumulator()
```

### Task 2.4: Pipe Composition Integration Test
**File**: fbp_process.py
**Playground**: Cell 8
**Estimate**: 20 min

**Test in Playground**:
```python
# %%
# Cell 8: Full pipeline test
async def test_full_pipeline():
    async def source_gen():
        for i in range(10):
            yield i * 2

    stream = Stream(source_gen())
    acc = Accumulator()

    pipeline = stream | acc

    async for _ in pipeline:
        pass

    assert acc.result == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

await test_full_pipeline()
```

---

## Phase 3: Observable Composite (StreamController)

### Task 3.1: Context Integration - ContextVar
**File**: context.py
**Playground**: Cell 9
**Estimate**: 30 min

**Requirements**:
- Add `contextvars.ContextVar` to context.py
- Implement `Context.current()` class method
- Set context in `__aenter__`, clear in `__aexit__`

**Test in Playground**:
```python
# %%
# Cell 9: Context with ContextVar
from promptview.prompt.context import Context

async def test_context_var():
    ctx = Context()

    assert Context.current() is None

    async with ctx:
        assert Context.current() is ctx

    assert Context.current() is None

await test_context_var()
```

### Task 3.2: Context - Execution Stack
**File**: context.py
**Playground**: Cell 10
**Estimate**: 45 min

**Requirements**:
- Add `_execution_stack: list[Process]` to Context
- Add `start_component(component, name, span_type, tags) -> SpanTree`
- Add `complete_component() -> Process`
- Properties: `current_component`, `current_span_tree`

**Test in Playground**:
```python
# %%
# Cell 10: Context execution stack
from promptview.prompt.context import Context
from promptview.prompt.fbp_process import Process

async def test_execution_stack():
    ctx = Context()

    async with ctx:
        proc1 = Process()
        span1 = await ctx.start_component(proc1, "proc1", "component", [])

        assert ctx.current_component is proc1
        assert ctx.current_span_tree is span1

        proc2 = Process()
        span2 = await ctx.start_component(proc2, "proc2", "component", [])

        assert ctx.current_component is proc2
        assert len(ctx._execution_stack) == 2

        await ctx.complete_component()
        assert ctx.current_component is proc1

await test_execution_stack()
```

### Task 3.3: StreamController - Basic Structure
**File**: fbp_process.py
**Playground**: Cell 11
**Estimate**: 60 min

**Requirements**:
- Create `StreamController(Process)` class
- Constructor: `__init__(gen_func, name, span_type, tags, ...)`
- Properties: `ctx`, `span`, `span_id`
- Implement `on_start()` to build subnetwork and register with context

**Test in Playground**:
```python
# %%
# Cell 11: StreamController basic
from promptview.prompt.fbp_process import StreamController
from promptview.prompt.context import Context

async def my_stream_gen():
    for i in range(3):
        yield {"value": i}

async def test_stream_controller():
    ctx = Context()

    async with ctx:
        controller = StreamController(
            gen_func=my_stream_gen,
            name="test_stream",
            span_type="stream",
            tags=["test"]
        )

        results = []
        async for ip in controller:
            results.append(ip)

        assert len(results) == 3
        assert controller.span.name == "test_stream"
        assert controller.span.status == "completed"

await test_stream_controller()
```

### Task 3.4: StreamController - Event Emission
**File**: fbp_process.py
**Playground**: Cell 12
**Estimate**: 45 min

**Requirements**:
- Implement event methods: `on_start_event`, `on_value_event`, `on_stop_event`, `on_error_event`
- Events return `StreamEvent` objects
- Events include span_id, path, payload

**Test in Playground**:
```python
# %%
# Cell 12: StreamController events
from promptview.prompt.events import StreamEvent

async def test_stream_events():
    ctx = Context()

    async with ctx:
        controller = StreamController(
            gen_func=my_stream_gen,
            name="test_stream",
            span_type="stream"
        )

        # Manually test event generation
        await controller.on_start()

        start_event = await controller.on_start_event()
        assert isinstance(start_event, StreamEvent)
        assert start_event.type == "stream_start"
        assert start_event.name == "test_stream"

await test_stream_events()
```

---

## Phase 4: Dynamic Composite & Orchestration

### Task 4.1: PipeController - Basic Structure
**File**: fbp_process.py
**Playground**: Cell 13
**Estimate**: 60 min

**Requirements**:
- Create `PipeController(Process)` class
- Similar to StreamController but for generators that yield other processes
- Track `index` for each IP sent

**Test in Playground**:
```python
# %%
# Cell 13: PipeController basic
from promptview.prompt.fbp_process import PipeController

async def my_pipe():
    stream1 = StreamController(my_stream_gen, "stream1", "stream")
    yield stream1

    stream2 = StreamController(my_stream_gen, "stream2", "stream")
    yield stream2

async def test_pipe_controller():
    ctx = Context()

    async with ctx:
        pipe = PipeController(
            gen_func=my_pipe,
            name="test_pipe",
            span_type="component"
        )

        children = []
        async for child in pipe:
            children.append(child)

        assert len(children) == 2
        assert isinstance(children[0], StreamController)

await test_pipe_controller()
```

### Task 4.2: FlowRunner - Stack Management
**File**: fbp_process.py
**Playground**: Cell 14
**Estimate**: 90 min

**Requirements**:
- Create `FlowRunner` class
- Manage stack of processes
- When process yields another process, push to stack
- Emit events based on `event_level`

**Test in Playground**:
```python
# %%
# Cell 14: FlowRunner with nested processes
from promptview.prompt.fbp_process import FlowRunner
from promptview.prompt.flow_components import EventLogLevel

async def test_flow_runner():
    ctx = Context()

    async with ctx:
        pipe = PipeController(my_pipe, "test_pipe", "component")
        runner = FlowRunner(pipe, event_level=EventLogLevel.chunk)

        events = []
        async for event in runner.stream_events():
            events.append(event)

        # Should have: pipe_start, stream1_start, stream1_deltas..., stream1_end,
        #              stream2_start, stream2_deltas..., stream2_end, pipe_end
        assert len(events) > 0
        assert events[0].type == "span_start"  # pipe start

await test_flow_runner()
```

---

## Phase 5: Parser Integration

### Task 5.1: Parser Process
**File**: fbp_process.py
**Playground**: Cell 15
**Estimate**: 60 min

**Requirements**:
- Port `Parser` from flow_components.py
- Inherit from `Process`
- Maintain XML parsing state

**Test in Playground**:
```python
# %%
# Cell 15: Parser integration
from promptview.block import Block, BlockChunk
from promptview.prompt.fbp_process import Parser

async def chunk_stream():
    chunks = ["<response>", "<text>Hello", " world", "</text>", "</response>"]
    for chunk in chunks:
        yield BlockChunk(content=chunk)

async def test_parser():
    response_schema = Block(
        text=str
    )

    stream = Stream(chunk_stream())
    parser = Parser(response_schema)
    acc = Accumulator()

    pipeline = stream | parser | acc

    async for _ in pipeline:
        pass

    # Check that parser built structured response
    assert len(acc.result) > 0

await test_parser()
```

---

## Phase 6: Replay Foundation

### Task 6.1: SpanTree I/O Logging
**File**: fbp_process.py, span_tree.py
**Playground**: Cell 16
**Estimate**: 60 min

**Requirements**:
- Modify `Process.__anext__()` to log inputs to span_tree
- Modify `Process.asend()` to log outputs to span_tree
- Only log when `_span_tree` is set

**Test in Playground**:
```python
# %%
# Cell 16: SpanTree I/O logging
async def test_span_io_logging():
    ctx = Context()

    async with ctx:
        controller = StreamController(my_stream_gen, "test", "stream")

        async for ip in controller:
            pass

        # Check that inputs/outputs were logged
        assert len(controller._span_tree.outputs) > 0

await test_span_io_logging()
```

### Task 6.2: Replay Mode - Process Base
**File**: fbp_process.py
**Playground**: Cell 17
**Estimate**: 60 min

**Requirements**:
- Add `_replay_mode: bool` to Process
- Add `_replay_inputs: list`, `_replay_index: int`
- Modify `__anext__()` to use replay buffer when in replay mode
- Add `from_span(span_tree)` class method

**Test in Playground**:
```python
# %%
# Cell 17: Process replay mode
async def test_process_replay():
    # First, execute normally and save
    ctx = Context()

    async with ctx:
        controller = StreamController(my_stream_gen, "test", "stream")

        original_results = []
        async for ip in controller:
            original_results.append(ip)

        span_tree = controller._span_tree
        await span_tree.save()

    # Now replay from span
    ctx2 = Context()

    async with ctx2:
        replayed_controller = await StreamController.from_span(span_tree)

        replayed_results = []
        async for ip in replayed_controller:
            replayed_results.append(ip)

        assert original_results == replayed_results

await test_process_replay()
```

### Task 6.3: Full Replay Test
**File**: fbp_process.py
**Playground**: Cell 18
**Estimate**: 45 min

**Test in Playground**:
```python
# %%
# Cell 18: Full pipeline replay
async def test_full_replay():
    # Execute complex pipeline
    ctx = Context()

    async with ctx:
        pipe = PipeController(my_pipe, "main_pipe", "component")
        runner = FlowRunner(pipe)

        original_events = []
        async for event in runner.stream_events():
            original_events.append(event)

        root_span = ctx._span_tree

    # Replay entire pipeline
    ctx2 = Context()

    async with ctx2:
        runner2 = await FlowRunner.from_span_tree(root_span, ctx2)

        replayed_events = []
        async for event in runner2.stream_events():
            replayed_events.append(event)

        assert len(original_events) == len(replayed_events)

await test_full_replay()
```

---

## Phase 7: Advanced Replay (Future)

### Task 7.1: Partial Replay - Path Navigation
### Task 7.2: Function Registry
### Task 7.3: Resume from Arbitrary Point

---

## Summary

**Total Estimated Time**: ~16 hours across 18 tasks

**Testing Strategy**:
- Each task has corresponding playground cell
- Progressive integration (each phase builds on previous)
- Interactive testing for rapid iteration

**Success Metrics**:
- All playground cells execute without errors
- Can compose pipelines: `stream | parser | accumulator`
- Can execute nested processes with proper span tracking
- Can replay from saved SpanTree

"""

# Implementation starts here

from typing import Any, AsyncGenerator, Callable, Type, TYPE_CHECKING

import json
import asyncio

from promptview.model.versioning.models import ExecutionSpan

if TYPE_CHECKING:
    from .span_tree import SpanTree
    from pydantic import BaseModel
    from promptview.block import Block


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
                await self._span_tree.log_value(ip, io_kind="input")

            # Track that we've yielded at least once
            if not self._did_yield:
                self._did_yield = True

            return ip

        except StopAsyncIteration:
            # Upstream exhausted - cleanup and propagate
            await self.on_stop()
            raise StopAsyncIteration

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

        super().__init__(upstream=GeneratorWrapper(gen))
        self._name = name
        self._save_stream_dir: str | None = None

    def save_stream(self, filepath: str):
        """
        Enable stream persistence - saves each IP to a JSONL file.

        Args:
            filepath: Path to JSONL file to save stream to
        """
        self._save_stream_dir = filepath

    @classmethod
    def load(cls, filepath: str, model: Type["BaseModel"] | None = None, delay: float = 0.0):
        """
        Load a stream from a saved JSONL file.

        Args:
            filepath: Path to JSONL file to load stream from
            model: Optional Pydantic model to deserialize IPs into
            delay: Optional delay between IPs (for simulating streaming)

        Returns:
            New Stream instance that replays from file
        """
        async def load_stream():
            with open(filepath, "r") as f:
                for line in f:
                    if delay > 0:
                        await asyncio.sleep(delay)

                    data = json.loads(line)

                    if model:
                        # Deserialize into Pydantic model
                        ip = model.model_validate(data)
                    else:
                        # Return raw dict
                        ip = data

                    yield ip

        return cls(load_stream(), name=f"stream_from_{filepath}")

    async def __anext__(self):
        """
        Receive next IP from wrapped generator.

        If persistence is enabled, writes IP to file before returning.
        """
        ip = await super().__anext__()

        # Save to file if persistence enabled
        if self._save_stream_dir:
            with open(self._save_stream_dir, "a") as f:
                # Handle Pydantic models
                if hasattr(ip, "model_dump"):
                    data = ip.model_dump()
                else:
                    data = ip

                f.write(json.dumps(data) + "\n")

        return ip


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
        span_type: str = "component",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None
    ):
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
        self._span_tree: "SpanTree | None" = None
        self.resolved_kwargs: dict[str, Any] = {}
        self.index: int = 0  # Set by parent PipeController
        self.parent: "ObservableProcess | None" = None  # Set by parent PipeController
        self._replay_inputs: list | None = None  # Saved inputs for replay mode
        self._replay_outputs: list | None = None  # Saved outputs for replay mode
        self._replay_index: int = 0  # Current position in replay buffer

    @property
    def ctx(self):
        """Get current context."""
        from .context import Context
        ctx = Context.current()
        if ctx is None:
            raise ValueError("StreamController requires Context. Use 'async with Context():'")
        return ctx

    @property
    def span(self):
        """Get the execution span."""
        return self._span_tree.root if self._span_tree else None

    @property
    def span_id(self):
        """Get the span ID."""
        return self._span_tree.id if self._span_tree else None

    async def _resolve_dependencies(self):
        """
        Resolve dependencies using injector.py and log as inputs.

        Returns:
            Tuple of (bound_arguments, resolved_kwargs)
        """
        from .injector import resolve_dependencies_kwargs
        
        
        self._span_tree = await self.ctx.start_span(
            component=self,
            name=self._name,
            span_type=self._span_type,
            tags=self._tags
        )
        
        if self._span_tree.inputs:            
            self._replay_inputs = [v.value for v in self._span_tree.inputs]
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
            if self._span_tree:
                for key, value in kwargs.items():
                    if value is not None:
                        await self._span_tree.log_value(value, io_kind="input")
                    
        if self._span_tree.outputs and not self._span_tree.need_to_replay:
            self._replay_outputs = [v.value for v in self._span_tree.outputs]

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

        event_type = "stream_start" if self._span_type == "stream" else "span_start"

        return StreamEvent(
            type=event_type,
            name=self._name,
            attrs=value_attrs,
            payload=self.span,
            span_id=str(self.span_id) if self.span_id else None,
            path=self.get_execution_path(),
        )

    async def on_value_event(self, payload: Any = None):
        """
        Generate value event.

        Returns:
            StreamEvent with type="stream_delta" or "span_event"
        """
        from .events import StreamEvent

        event_type = "stream_delta" if self._span_type == "stream" else "span_event"

        return StreamEvent(
            type=event_type,
            name=self._name,
            payload=payload,
            span_id=str(self.span_id) if self.span_id else None,
            path=self.get_execution_path(),
        )

    async def on_stop_event(self, payload: Any = None):
        """
        Generate stop event.

        Returns:
            StreamEvent with type="stream_end" or "span_end"
        """
        from .events import StreamEvent

        event_type = "stream_end" if self._span_type == "stream" else "span_end"

        return StreamEvent(
            type=event_type,
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

        event_type = "stream_error" if self._span_type == "stream" else "span_error"

        return StreamEvent(
            type=event_type,
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

    def stream(self, event_level=None):
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
        return FlowRunner(self, event_level=event_level).stream_events()

    def get_response(self):
        """
        Get the response/result from this process.

        For most processes, this is just the last value.
        Subclasses can override to provide more sophisticated response handling.
        """
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
        span_type: str = "stream",
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
        super().__init__(gen_func, name, span_type, tags, args, kwargs, upstream)
        self._stream: Process | None = None
        self._accumulator: Accumulator | None = None
        self._parser: Parser | None = None

    async def on_start(self):
        """
        Build subnetwork and register with context.

        Called automatically on first __anext__() call.
        Creates span, resolves dependencies, logs kwargs as inputs, and builds the internal stream.

        In replay mode (when span has saved outputs), sets up replay buffer instead of
        executing the generator function.
        """
        bound, kwargs =await self._resolve_dependencies()
        gen_instance = self._gen_func(*bound.args, **bound.kwargs)

        # Wrap in Stream process and pipe through Accumulator
        stream = Stream(gen_instance, name=f"{self._name}_stream")
        accumulator = Accumulator()
        self._stream = stream | accumulator
        if self._parser is not None:
            self._stream |= self._parser
        self._accumulator = accumulator
        
    def parse(self, block_schema: "Block"):
        if self._parser is not None:
            raise FlowException("Parser already initialized")
        if self._gen_func is None:
            raise FlowException("StreamController is not initialized")
        self._parser = Parser(response_schema=block_schema)      
        return self


    async def __anext__(self):
        """
        Delegate to internal stream process or replay from saved outputs.

        In replay mode, yields saved outputs from the span instead of executing
        the generator function.
        """
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

                ip = self._replay_outputs[self._replay_index]
                self._replay_index += 1
                self._last_ip = ip

                if not self._did_yield:
                    self._did_yield = True

                return ip
            else:
                # Normal mode: get next IP from stream (which passes through Accumulator)
                ip = await self._stream.__anext__()
                self._last_ip = ip

                # Don't log individual values - they're stream deltas
                # We'll log the final accumulated result in on_stop()

                if not self._did_yield:
                    self._did_yield = True

                return ip
        except StopAsyncIteration:
            # Log the final accumulated result as output
            if self._span_tree and self._accumulator:
                await self._span_tree.log_value(self._accumulator.result, io_kind="output")

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
        elif self._accumulator:
            # Normal mode: return accumulated values
            return self._accumulator.result
        return []


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
        span_type: str = "component",
        tags: list[str] | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        upstream: Process | None = None
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

                # Set parent reference and index on child
                if isinstance(child, (StreamController, PipeController)):
                    child.parent = self
                    child.index = self.index
                    self.index += 1  # Increment for next child

                # Log child as output
                if self._span_tree and isinstance(child, (StreamController, PipeController)):
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

    def __init__(self, root_process: Process, event_level=None):
        """
        Initialize FlowRunner.

        Args:
            root_process: The root process to execute (usually PipeController)
            event_level: Level of events to emit (chunk, span, turn)
        """
        self.stack: list[Process] = [root_process]
        self.last_value: Any = None
        self._output_events = False
        self._error_to_raise: Exception | None = None
        self._last_process: Process | None = None
        self._event_level = event_level
        self._pending_child: Process | None = None  # Child process waiting to be pushed
        self._response_to_send: Any = None  # Response from child to send to parent

    @property
    def current(self) -> Process:
        """Get current process from stack."""
        if not self.stack:
            raise StopAsyncIteration
        return self.stack[-1]

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
        return process

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

    def __aiter__(self):
        """Make FlowRunner async iterable."""
        return self

    async def __anext__(self):
        """
        Execute next step in process network.

        Returns:
            StreamEvent if emitting events, otherwise the value from current process
        """
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

                # If value is a Process (from PipeController), push to stack
                if isinstance(value, (StreamController, PipeController)):
                    self.push(value)
                    # Emit event for child process if needed
                    if self.should_output_events:
                        if event := await self.try_build_value_event(process, value):
                            return event
                    continue

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

                # If there's a parent waiting, save the response to send
                if self.stack:
                    self._response_to_send = response

                # Emit stop event if needed
                if self.should_output_events:
                    if event := await self.try_build_stop_event(process, self.last_value):
                        return event

            except Exception as e:
                # Error occurred, pop from stack
                process = self.pop()

                # Emit error event if needed
                if self.should_output_events:
                    event = await self.try_build_error_event(process, e)
                    if not self.stack:
                        raise e
                    self._error_to_raise = e
                    return event
                else:
                    raise e

        raise StopAsyncIteration

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


# ============================================================================
# Phase 5: Parser Integration
# ============================================================================


class Parser(Process):
    """
    Parser process - transforms XML-tagged chunks into Block structures.
    
    This is a STATEFUL TRANSFORMER in FBP - maintains parsing state,
    buffers chunks, and yields Block events as XML tags are parsed.
    
    The Parser uses lxml's XMLPullParser to incrementally parse streaming
    XML content and build Block structures via BlockBuilderContext.
    
    Example:
        >>> response_schema = Block(text=str)
        >>> parser = Parser(response_schema)
        >>> 
        >>> stream = Stream(chunk_gen())
        >>> pipeline = stream | parser | acc
        >>> async for block_event in pipeline:
        ...     print(block_event)
    """
    
    def __init__(self, response_schema: "Block", upstream: Process | None = None):
        """
        Initialize Parser.
        
        Args:
            response_schema: Block schema defining expected structure
            upstream: Upstream process providing BlockChunk IPs
        """
        super().__init__(upstream)
        from lxml import etree
        from ..block.block9.block_schema import BlockBuilderContext
        
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
        self._event_queue = []  # Queue events to yield across multiple __anext__ calls
        
    def _push_tag(self, tag: str, is_list: bool):
        self._tag_stack.append({"tag": tag, "is_list": is_list})
    
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
        self._full_content += content
        self.parser.feed(content)
    
    def _process_chunk(self, chunk):
        """Process a chunk and add events to queue."""
        if not self._stream_started:
            self._feed_parser(f'<{self._safety_tag}>')
            self._stream_started = True
        
        # Check if res_ctx has queued events first
        if self.res_ctx.has_events():
            event = self.res_ctx.get_event()
            self._event_queue.append(event)
        
        # Process the chunk
        self._write_to_buffer(chunk)
        try:
            self._feed_parser(chunk.content)
        except Exception as e:
            print(self._full_content)
            print(f"Parser Error on content: {chunk.content}")
            raise e
        
        self._try_set_tag_lock(chunk)
        
        # Output chunks in middle of stream
        if self._should_output_chunk():
            for c in self._read_buffer(flush=True):
                self.res_ctx.append(self.current_tag, c)
        
        # Process XML events
        for event, element in self.parser.read_events():
            if element.tag == self._safety_tag:
                continue
            
            if event == 'start':
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
                self.res_ctx.set_view_attr(
                    element.tag,
                    postfix=self._read_buffer(start_from="</"),
                )
                self._pop_tag()
                self._release_tag_lock()
        
        # Collect any new events
        while self.res_ctx.has_events():
            event = self.res_ctx.get_event()
            self._event_queue.append(event)
    
    async def __anext__(self):
        """
        Get next block event.
        
        Parser may need to consume multiple chunks to produce one event,
        or produce multiple events from one chunk.
        """
        # Process queued events first
        if self._event_queue:
            return self._event_queue.pop(0)
        
        # Try to get more chunks and process them
        max_iterations = 20
        for _ in range(max_iterations):
            try:
                # Get next chunk from upstream
                chunk = await super().__anext__()
                
                # Process the chunk
                self._process_chunk(chunk)
                
                # If we now have events, return one
                if self._event_queue:
                    return self._event_queue.pop(0)
            
            except StopAsyncIteration:
                # Upstream exhausted
                # Return any remaining queued events first
                if self._event_queue:
                    return self._event_queue.pop(0)
                # No more events, we're done
                raise StopAsyncIteration
        
        # Hit max iterations without producing event
        raise StopAsyncIteration

