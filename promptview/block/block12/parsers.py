"""
XmlParser - Streaming XML parser that builds blocks from a schema.

Uses Python's expat parser for streaming XML parsing.
Maps XML events to block building operations:
- Start tag → instantiate child schema
- Character data → append content
- End tag → commit and pop stack
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Generator
from xml.parsers import expat

from .block import Block
from .chunk import Chunk, ChunkMeta
from ...prompt.fbp_process import Process

if TYPE_CHECKING:
    from .schema import BlockSchema


class ParserError(Exception):
    """Error during parsing."""
    pass


@dataclass
class ParserEvent:
    """Event emitted by the parser during streaming."""
    path: str
    type: Literal["block_init", "block_commit", "block_delta"]
    block: Block
    chunks: list[Chunk] | None = None


class XmlParser(Process):
    """
    Streaming XML parser that builds blocks from a schema.

    Inherits from Process for FBP pipeline composition:
        stream | parser | accumulator

    Uses Python's expat parser for streaming XML parsing.
    Maps XML events to block building operations:
    - Start tag → instantiate child schema
    - Character data → append content
    - End tag → commit and pop stack

    Example:
        schema = BlockSchema("response")
        schema /= BlockSchema("thinking")
        schema /= BlockSchema("answer")

        # Manual feeding
        parser = XmlParser(schema)
        parser.feed("<response>")
        parser.feed("<thinking>Let me think...</thinking>")
        parser.close()

        # Or in a pipeline
        stream = Stream(chunk_gen())
        parser = XmlParser(schema)
        async for event in stream | parser:
            print(event)
    """

    def __init__(self, schema: "BlockSchema", upstream: Process | None = None, verbose: bool = False):
        super().__init__(upstream)
        self.schema = schema
        self._root: Block | None = None
        self._stack: list[tuple["BlockSchema", Block]] = []
        self._verbose = verbose

        # Expat parser setup
        self._parser = expat.ParserCreate()
        self._parser.buffer_text = False
        self._parser.StartElementHandler = self._on_start
        self._parser.EndElementHandler = self._on_end
        self._parser.CharacterDataHandler = self._on_chardata

        # Chunk tracking for logprobs
        self._chunks: list[tuple[int, int, str, float | None]] = []  # (start_byte, end_byte, content, logprob)
        self._total_bytes = 0

        # Pending event for deferred processing
        self._pending: tuple[str, Any, int] | None = None

        # Output queue for events
        self._output_queue: list[ParserEvent] = []

        # Synthetic root tag handling
        self._root_tag = "_root_"
        self._has_synthetic_root = True

        # Stream exhausted flag
        self._is_stream_exhausted = False

        # Parser closed flag
        self._is_closed = False

        # Create wrapper schema
        from .schema import BlockSchema
        self._wrapper_schema = BlockSchema(name=self._root_tag)
        self._wrapper_schema._raw_append_child(schema.copy())
        self._index = 0

        # Start with synthetic root
        self.feed("<{}>".format(self._root_tag))

    @property
    def result(self) -> Block | None:
        """Get the built block tree, unwrapping synthetic root."""
        if self._root is None:
            return None
        # Unwrap synthetic root - return first real child
        if self._has_synthetic_root and self._root.children:
            return self._root.children[0]
        return self._root

    @property
    def current_block(self) -> Block | None:
        """Get the block currently being built."""
        if not self._stack:
            return None
        return self._stack[-1][1]

    @property
    def current_schema(self) -> "BlockSchema | None":
        """Get the schema of the block currently being built."""
        if not self._stack:
            return None
        return self._stack[-1][0]

    # -------------------------------------------------------------------------
    # Process interface
    # -------------------------------------------------------------------------

    async def on_stop(self):
        """Called when upstream is exhausted - finalize parsing."""
        self.close()

    async def __anext__(self):
        """
        Get next event from parser.

        Consumes chunks from upstream until an event is ready to output.
        """
        # Return queued output if available
        if self._output_queue:
            return self._output_queue.pop(0)
        elif self._is_stream_exhausted:
            raise StopAsyncIteration()

        # Consume upstream chunks until we have output
        while not self._output_queue and not self._is_stream_exhausted:
            try:
                chunk = await super().__anext__()
                # Feed the chunk (may produce output)
                if hasattr(chunk, 'content'):
                    self.feed(chunk.content, logprob=getattr(chunk, 'logprob', None))
                else:
                    self.feed(str(chunk))
            except StopAsyncIteration:
                # Upstream exhausted
                self._is_stream_exhausted = True
                await self.on_stop()
                if self._output_queue:
                    return self._output_queue.pop(0)
                raise

        if self._output_queue:
            return self._output_queue.pop(0)
        raise StopAsyncIteration()

    # -------------------------------------------------------------------------
    # Feeding data
    # -------------------------------------------------------------------------

    def feed(self, text: str, logprob: float | None = None, is_final: bool = False):
        """
        Feed text to the parser.

        Args:
            text: Text content to parse
            logprob: Optional log probability for this chunk
            is_final: Whether this is the last chunk
        """
        data = text.encode("utf-8")
        start = self._total_bytes
        end = start + len(data)
        self._chunks.append((start, end, text, logprob))
        self._total_bytes = end

        try:
            self._parser.Parse(data, is_final)
        except expat.ExpatError as e:
            current_data = "".join(c[2] for c in self._chunks)
            raise ParserError(f"XML parse error: {e}. Current data: {current_data}")

    def close(self):
        """Close the parser and finalize the block tree."""
        if self._is_closed:
            return
        self._is_closed = True

        if self._has_synthetic_root:
            self.feed("</{}>".format(self._root_tag))
        self._parser.Parse(b"", True)
        self._flush_pending(self._total_bytes)

    # -------------------------------------------------------------------------
    # Schema lookup
    # -------------------------------------------------------------------------

    def _get_child_schema(self, name: str) -> "BlockSchema | None":
        """Find a child schema by name."""
        from .schema import BlockSchema

        if self.current_schema is None:
            # At root level - check wrapper's children
            for child in self._wrapper_schema.children:
                if isinstance(child, BlockSchema):
                    if name == child.name or name in child.tags:
                        return child
            return None

        # Search children of current schema
        for child in self.current_schema.children:
            if isinstance(child, BlockSchema):
                if name == child.name or name in child.tags:
                    return child

        return None

    # -------------------------------------------------------------------------
    # Chunk retrieval
    # -------------------------------------------------------------------------

    def _get_chunks_in_range(self, start: int, end: int) -> list[tuple[str, float | None]]:
        """Get chunks overlapping the byte range [start, end)."""
        result = []

        for chunk_start, chunk_end, content, logprob in self._chunks:
            if chunk_start < end and chunk_end > start:
                # Calculate slice indices
                slice_start = max(0, start - chunk_start)
                slice_end = min(len(content.encode("utf-8")), end - chunk_start)

                # Convert byte indices to character indices (approximate for non-ASCII)
                text = content.encode("utf-8")[slice_start:slice_end].decode("utf-8", errors="replace")
                if text:
                    result.append((text, logprob))

        return result

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    def _flush_pending(self, end_byte: int):
        """Process any pending event."""
        if self._pending is None:
            return
        self._index += 1
        event_type, event_data, start_byte = self._pending
        chunks = self._get_chunks_in_range(start_byte, end_byte)
        self._pending = None

        if not chunks:
            return

        if self._verbose:
            print(f"Event {self._index}: {event_type}, data={event_data}, chunks={chunks}")

        if event_type == "start":
            name, attrs = event_data
            self._handle_start(name, attrs, chunks)
        elif event_type == "end":
            name = event_data
            self._handle_end(name, chunks)
        elif event_type == "chardata":
            self._handle_chardata(chunks)

    def _on_start(self, name: str, attrs: dict):
        """Handle start tag event from expat."""
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._pending = ("start", (name, attrs), current_pos)

    def _on_end(self, name: str):
        """Handle end tag event from expat."""
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._pending = ("end", name, current_pos)

    def _on_chardata(self, data: str):
        """Handle character data event from expat."""
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._pending = ("chardata", data, current_pos)

    # -------------------------------------------------------------------------
    # Block building
    # -------------------------------------------------------------------------

    def _handle_start(self, name: str, attrs: dict, chunks: list[tuple[str, float | None]]):
        """Handle opening tag - instantiate block from schema."""
        # Handle synthetic root
        if name == self._root_tag:
            self._root = Block(tags=[self._root_tag])
            self._stack.append((self._wrapper_schema, self._root))
            self._emit_event("block_init", self._root)
            return

        # Find child schema
        child_schema = self._get_child_schema(name)
        if child_schema is None:
            raise ParserError(f"Unknown tag '{name}' - no matching schema found")

        # Instantiate block from schema
        child_block = child_schema._inst_content(
            style=child_schema.style,
            tags=list(child_schema.tags),
            role=child_schema.role,
            attrs=dict(attrs) if attrs else dict(child_schema.attrs),
        )

        # Append to current block
        if self.current_block is not None:
            self.current_block.append_child(child_block)

        # Push to stack
        self._stack.append((child_schema, child_block))
        self._emit_event("block_init", child_block)

    def _handle_end(self, name: str, chunks: list[tuple[str, float | None]]):
        """Handle closing tag - commit and pop stack."""
        # Skip synthetic root
        if name == self._root_tag:
            return

        if not self._stack:
            raise ParserError(f"Unexpected closing tag '{name}' - stack is empty")

        # Validate name matches
        schema, block = self._stack[-1]
        if name != schema.name and name not in schema.tags:
            raise ParserError(f"Mismatched closing tag: expected '{schema.name}', got '{name}'")

        # Pop from stack
        self._stack.pop()

        # Emit commit event
        self._emit_event("block_commit", block)

    def _handle_chardata(self, chunks: list[tuple[str, float | None]]):
        """Handle character data - append to current block."""
        if self.current_block is None:
            return

        # Append each chunk with its logprob
        result_chunks = []
        for content, logprob in chunks:
            if content:  # Skip empty content
                chunk = self.current_block._raw_append(content, logprob=logprob)
                result_chunks.append(chunk)

        if result_chunks:
            self._emit_event("block_delta", self.current_block, result_chunks)

    def _emit_event(self, event_type: str, block: Block, chunks: list[Chunk] | None = None):
        """Emit a parser event."""
        # Build path from stack
        path = "/".join(str(i) for i, _ in enumerate(self._stack))
        event = ParserEvent(path=path, type=event_type, block=block, chunks=chunks)
        self._output_queue.append(event)

        if self._verbose:
            print(f"Emit: {event_type} path={path} block={block}")

    # -------------------------------------------------------------------------
    # Iterator interface (for non-pipeline usage)
    # -------------------------------------------------------------------------

    def events(self) -> Generator[ParserEvent, None, None]:
        """Yield all queued events (for synchronous usage)."""
        while self._output_queue:
            yield self._output_queue.pop(0)

    def __iter__(self):
        """Synchronous iterator for events."""
        return self.events()
