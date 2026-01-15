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
from typing import TYPE_CHECKING, Any, Literal, Generator, TypeVar
from xml.parsers import expat

from .block import Block
from .chunk import BlockChunk, ChunkMeta
from ...prompt.fbp_process import Process
from .schema import BlockSchema, BlockListSchema, BlockList

# if TYPE_CHECKING:
#     from .schema import BlockSchema, BlockListSchema


class ParserError(Exception):
    """Error during parsing."""
    pass


@dataclass
class ParserEvent:
    """Event emitted by the parser during streaming."""
    path: str
    type: Literal["block_init", "block_commit", "block_delta"]
    value: Block | list[BlockChunk]
    
    




TxSchema = TypeVar("TxSchema", BlockSchema, BlockListSchema)

class SchemaCtx[TxSchema]:
    
    def __init__(self, schema: TxSchema, is_root: bool = False):
        self.schema: TxSchema = schema
        self._block: Block | None = None
        self._content_started = False
        self._should_add_newline = False
        self._is_root = is_root
    @property
    def block(self) -> Block:
        if self._block is None:
            raise ValueError("Block is not initialized")
        return self._block

    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> list[SchemaCtx]:
        raise NotImplementedError("init not implemented")
    
    
    def append(self, chunks: list[BlockChunk]):
        raise NotImplementedError("append not implemented")
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        raise NotImplementedError("commit not implemented")
    
    
    def get_child_schema(self, name: str, attrs: dict) -> BlockSchema | BlockListSchema:
        raise NotImplementedError("get_child_schema not implemented")
    
    def build_child_schema(self, name: str, attrs: dict):
        schema = self.get_child_schema(name, attrs)
        if schema is None:
            raise ValueError(f"Unknown tag '{name}' - no matching schema found")
        if isinstance(schema, BlockListSchema):
            return BlockListSchemaCtx(schema)
        elif isinstance(schema, BlockSchema):
            return BlockSchemaCtx(schema)
        else:
            raise ValueError(f"Unknown schema type: {type(schema)}")
        
    def add_newline(self):
        self.block.append_child("")
        self._should_add_newline = False
    
class BlockSchemaCtx(SchemaCtx[BlockSchema]):
    
    
    def get_child_schema(self, name: str, attrs: dict) -> BlockSchema | BlockListSchema:
        schema = self.schema.get_schema(name)
        return schema
           
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> list[SchemaCtx]:
        # self._block = Block(
        #     role=self.schema.role,
        #     tags=list(self.schema.tags),
        #     style=list(self.schema.style),
        #     attrs=dict(attrs) if attrs else dict(self.schema.attrs),
        # )
        # self.append(chunks)
        self._block = self.schema.init_partial(chunks, is_streaming=True)
        return [self]
    
    # def append2(self, chunks: list[BlockChunk]):
    #     for chunk in chunks:
    #         if chunk.content:
    #             if not self._content_started:
    #                 if chunk == "\n":
    #                     self.block.append(chunk.content, logprob=chunk.logprob, use_mutator_style=True)
    #                 self._content_started = True
    #                 self.block.append_child("")
    #             events = self.block.tail.append(chunk.content, logprob=chunk.logprob)
    #     return chunks
    def append(self, chunks: list[BlockChunk]):        
        for chunk in chunks:
            style = None
            if chunk.content:
                if self._should_add_newline:
                    self.add_newline()
                if chunk.is_newline():
                    self._should_add_newline = True
                    style = self.block.mutator.styles[0]
                elif chunk.isspace():
                    style = self.block.mutator.styles[0]
                events = self.block.tail.append(chunk.content, logprob=chunk.logprob, style=style)                
        return chunks
    
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        postfix = Block(chunks)
        self.block.commit(postfix)
        return postfix

class BlockListSchemaCtx(SchemaCtx[BlockListSchema]):
    
    
    def _get_key(self, attrs: dict) -> str:
        if not self.schema.key:
            raise ValueError("key is required for BlockListSchema")
        item_name = attrs.get(self.schema.key)
        if not isinstance(item_name, str):
            raise ValueError(f"Item name '{item_name}' is not a string")
        return item_name
        
    
    def get_child_schema(self, name: str, attrs: dict) -> BlockSchema | BlockListSchema:
        item_name = self._get_key(attrs)
        schema = self.schema.get_schema(item_name)
        if schema is None:
            raise ValueError(f"Unknown item name '{item_name}' - no matching schema found")
        return schema
    
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> list[SchemaCtx]:        
        self._block = BlockList(
            role=self.schema.role,
            tags=list(self.schema.tags),
            style=list(self.schema.style),
            attrs=dict(attrs) if attrs else dict(self.schema.attrs),
        )
        item_name = self._get_key(attrs)
        item_ctx = self.build_child_schema(item_name, attrs)
        item_ctx.init(item_name, attrs, chunks)
        return [self, item_ctx]

class ContextStack:
    
    def __init__(self, schema: "BlockSchema | BlockListSchema", root_name: str | None = None):
        self._stack: list[SchemaCtx] = []
        self._commited_stack: list[SchemaCtx] = []
        self._schema = schema
        self._pending_pop: SchemaCtx | None = None
        self._did_start = False
        self._root_name = root_name
    
    @property
    def curr_block(self) -> Block:
        return self.top().block
    
    def top(self) -> "SchemaCtx":
        if self._pending_pop is not None:
            return self._pending_pop
        return self._stack[-1]
    
    def push(self, schema_ctx: list[SchemaCtx]):
        if self._pending_pop is not None:
            self._pending_pop = None
        if not self.is_empty():
            self.curr_block.append_child(schema_ctx[0].block)
        self._stack.extend(schema_ctx)
        
    def pop(self) -> SchemaCtx:
        schema_ctx = self._stack.pop()
        self._pending_pop = schema_ctx
        return schema_ctx
    
    def is_empty(self) -> bool:
        return not self._stack
    
    def is_root(self) -> bool:
        return len(self._stack) == 1 and self._stack[0].schema == self._schema
        
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]):
        self._pending_pop = None
        if self.is_empty():
            if self._did_start:
                raise ParserError("Unexpected start tag - stack is empty")
            self._did_start = True
            schema_ctx = BlockSchemaCtx(self._schema, is_root=self._root_name == name)
        else:
            schema_ctx = self.top().build_child_schema(name, attrs)
        ctx_list = schema_ctx.init(name, attrs, chunks)
        self.push(ctx_list)
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        schema_ctx = self.pop()
        postfix = schema_ctx.commit(name, chunks)                
        self._commited_stack.append(schema_ctx)        
        return postfix
    
        
    def append(self, chunks: list[BlockChunk]):
        if self._pending_pop is not None:
            if chunks[0] != "\n":
                self._pending_pop = None
            if self.is_root():
                self.top().block.append_child("")
        return self.top().append(chunks)
    
    
    def result(self) -> Block:
        return self._commited_stack[-1].block
    
    
    def top_event_block(self) -> Block:
        return self.top().block.extract()




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

        # Synthetic root tag handling
        self._root_tag = "_root_"
        self._has_synthetic_root = True

        # Extract schema with root tag for wrapping multiple schemas
        self.schema = schema.extract_schema(style="xml", root=self._root_tag)
        if self.schema is None:
            raise ParserError("No schema found to parse against")

        self._root: Block | None = None
        self._stack: list[tuple["BlockSchema", Block]] = []        
        self._verbose = verbose

        # Expat parser setup
        self._parser = expat.ParserCreate()
        self._parser.buffer_text = False
        self._parser.StartElementHandler = self._on_start
        self._parser.EndElementHandler = self._on_end
        self._parser.CharacterDataHandler = self._on_chardata

        # Chunk tracking for logprobs - stores (start_byte, end_byte, Chunk)
        self._chunks: list[tuple[int, int, BlockChunk]] = []
        self._total_bytes = 0

        # Pending event for deferred processing
        self._pending: tuple[str, Any, int] | None = None

        # Output queue for events
        self._output_queue: list[ParserEvent] = []

        # Stream exhausted flag
        self._is_stream_exhausted = False

        # Parser closed flag
        self._is_closed = False

        # Create wrapper schema - if extracted schema already has _root_tag as name,
        # use it directly; otherwise wrap it
        from .schema import BlockSchema
        if self.schema.name == self._root_tag:
            # extract_schema created the wrapper for us
            self._wrapper_schema = self.schema
        else:
            # Single schema - wrap it
            self._wrapper_schema = BlockSchema(name=self._root_tag, style="block", is_root=True)
            self._wrapper_schema._raw_append_child(self.schema)
        self._ctx_stack: ContextStack = ContextStack(self._wrapper_schema, root_name=self._root_tag)
        self._index = 0

        # Start with synthetic root
        self.feed("<{}>".format(self._root_tag))

    # @property
    # def result(self) -> Block | None:
    #     """Get the built block tree, unwrapping synthetic root."""
    #     if self._root is None:
    #         return None
    #     # Unwrap synthetic root - return first real child
    #     if self._has_synthetic_root and self._root.children:
    #         return self._root.children[0]
    #     return self._root
    @property
    def result(self) -> Block | None:
        """Get the built block tree, unwrapping synthetic root."""
        return self._ctx_stack.result()

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
                upstream_chunk = await super().__anext__()
                # Feed the chunk (may produce output)
                if isinstance(upstream_chunk, BlockChunk):
                    self.feed(upstream_chunk)
                elif hasattr(upstream_chunk, 'content'):
                    chunk = BlockChunk(
                        upstream_chunk.content,
                        logprob=getattr(upstream_chunk, 'logprob', None)
                    )
                    self.feed(chunk)
                else:
                    self.feed(str(upstream_chunk))
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

    def feed(self, chunk: BlockChunk | str, logprob: float | None = None, is_final: bool = False):
        """
        Feed a chunk to the parser.

        Args:
            chunk: Chunk object or string content to parse
            logprob: Optional log probability (used if chunk is a string)
            is_final: Whether this is the last chunk
        """
        # Convert string to Chunk if needed
        if isinstance(chunk, str):
            chunk = BlockChunk(chunk, logprob=logprob)

        data = chunk.content.encode("utf-8")
        start = self._total_bytes
        end = start + len(data)
        self._chunks.append((start, end, chunk))
        self._total_bytes = end

        try:
            self._parser.Parse(data, is_final)
        except expat.ExpatError as e:
            current_data = "".join(c[2].content for c in self._chunks)
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

    def _get_chunks_in_range(self, start: int, end: int) -> list[BlockChunk]:
        """Get chunks overlapping the byte range [start, end)."""
        result = []

        for chunk_start, chunk_end, chunk in self._chunks:
            if chunk_start < end and chunk_end > start:
                # Check if we need to split the chunk
                if chunk_end > end:
                    # Split off the part after our range
                    chunk, _ = chunk.split(end - chunk_start)
                if chunk_start < start:
                    # Split off the part before our range
                    _, chunk = chunk.split(start - chunk_start)
                if chunk.content:
                    result.append(chunk)

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
            if event_type == "start":
                print(f"*********************** {event_data[0]} *************************")
            print("__________________")
            print(f"Event {self._index}: {event_type}, data={event_data!r}, chunks={chunks}")
            if event_type == "end":
                print(f"----------------------- {event_data} -------------------------")
            

        if event_type == "start":
            name, attrs = event_data
            self._handle_start(name, attrs, chunks)
        elif event_type == "end":
            name = event_data
            self._handle_end(name, chunks)
        elif event_type == "chardata":
            self._handle_chardata(chunks)
        
        if self._verbose:
            if not self._ctx_stack.is_empty():
                print("===>")                
                self._ctx_stack._stack[0].block.print_debug()

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
    def _handle_start(self, name: str, attrs: dict, chunks: list[BlockChunk]):
        """Handle opening tag - instantiate block from schema."""
        if name == self._root_tag:
            chunks = []
        self._ctx_stack.init(name, attrs, chunks)
        self._emit_event("block_init", self._ctx_stack.top_event_block())
        
        
    def _handle_end(self, name: str, chunks: list[BlockChunk]):
        """Handle closing tag - commit and pop stack."""
        if name == self._root_tag:
            chunks = []
        postfix = self._ctx_stack.commit(name, chunks)        
        self._emit_event("block_commit", postfix)
        
        
    def _handle_chardata(self, chunks: list[BlockChunk]):
        """Handle character data - append to current block."""
        self._ctx_stack.append(chunks)
        self._emit_event("block_delta", chunks)


    def _handle_start2(self, name: str, attrs: dict, chunks: list[BlockChunk]):
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

        # Instantiate block from schema (without initial content - we'll add prefix separately)
        child_block = Block(
            role=child_schema.role,
            tags=list(child_schema.tags),
            style=list(child_schema.style),
            attrs=dict(attrs) if attrs else dict(child_schema.attrs),
        )

        # Add opening tag as prefix chunk with logprob
        prefix_chunks = []
        for chunk in chunks:
            if chunk.content:
                result_chunk = child_block._raw_append(chunk.content, logprob=chunk.logprob, style="prefix")
                prefix_chunks.append(result_chunk)

        # Append to current block
        if self.current_block is not None:
            self.current_block.append_child(child_block)

        # Push to stack
        self._stack.append((child_schema, child_block))
        self._emit_event("block_init", child_block, prefix_chunks if prefix_chunks else None)

    

    
    def _handle_end2(self, name: str, chunks: list[BlockChunk]):
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

        # Add closing tag as postfix chunk with logprob
        postfix_chunks = []
        for chunk in chunks:
            if chunk.content:
                result_chunk = block._raw_append(chunk.content, logprob=chunk.logprob, style="postfix")
                postfix_chunks.append(result_chunk)

        # Pop from stack
        self._stack.pop()

        # Emit commit event
        self._emit_event("block_commit", block, postfix_chunks if postfix_chunks else None)



    def _handle_chardata2(self, chunks: list[BlockChunk]):
        """Handle character data - append to current block."""
        if self.current_block is None:
            return

        # Append each chunk with its logprob
        result_chunks = []
        for chunk in chunks:
            if chunk.content:  # Skip empty content
                result_chunk = self.current_block._raw_append(chunk.content, logprob=chunk.logprob)
                result_chunks.append(result_chunk)

        if result_chunks:
            self._emit_event("block_delta", self.current_block, result_chunks)

    def _emit_event(self, event_type: str, value: Block | list[BlockChunk]):
        """Emit a parser event."""
        # Build path from stack
        # path = "/".join(str(i) for i, _ in enumerate(self._stack))
        path = self._ctx_stack.top().block.path
        event = ParserEvent(path=str(path), type=event_type, value=value)
        self._output_queue.append(event)

        # if self._verbose:
            # print(f"Emit: {event_type} path={path} block={block}")

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
