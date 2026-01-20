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
        block = self.block.append_child()
        self._should_add_newline = False
        return block
    
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
    
    
    # def append(self, chunks: list[BlockChunk]):        
    #     for chunk in chunks:
    #         style = None
    #         if chunk.content:
    #             if not self.block.mutator._initialized:
    #                 if not chunk.is_newline():
    #                     self.add_newline()
    #                     self.block.mutator._initialized = True
    #             else:
    #                 if self._should_add_newline:
    #                     self.add_newline()
    #                 if chunk.is_newline():
    #                     self._should_add_newline = True
    #                     style = self.block.mutator.styles[0]                
                    
    #             # elif chunk.isspace():
    #             #     style = self.block.mutator.styles[0]
    #             events = self.block.tail.append(chunk.content, logprob=chunk.logprob, style=style or chunk.style)                
    #     return chunks
    
    def append(self, chunks: list[BlockChunk]):  
        block = None      
        for chunk in chunks:
            style = None
            if chunk.content:                
                if self._should_add_newline:
                    block = self.add_newline()
                if chunk.is_newline():
                    self._should_add_newline = True
                    style = self.block.mutator.styles[0]                    
                elif chunk.isspace():
                    style = self.block.mutator.styles[0]
                elif not self.block.body:
                    block = self.add_newline()
                events = self.block.tail.append(chunk.content, logprob=chunk.logprob, style=style or chunk.style)                
        if block is not None:
            return block
        return chunks

    
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        postfix = Block(chunks)
        self._should_add_newline = False
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
        self._pending_chunks: list[BlockChunk] = []
    
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
        return len(self._stack) == 1 and self.top().schema == self._schema
    
    
    def _append_pending_chunks(self, chunks: list[BlockChunk]) -> list[BlockChunk]:
        if self._pending_chunks:
            pending_chunks = []
            style = self.top().block.mutator.styles[0]
            for c in self._pending_chunks:
                c.meta.style = style
                pending_chunks.append(c)
            chunks = pending_chunks + chunks
            self._pending_chunks = []
        return chunks
        
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]):
        self._pending_pop = None
        if self.is_empty():
            if self._did_start:
                raise ParserError("Unexpected start tag - stack is empty")
            self._did_start = True
            schema_ctx = BlockSchemaCtx(self._schema, is_root=self._root_name == name)
        else:
            schema_ctx = self.top().build_child_schema(name, attrs)
            
        chunks = self._append_pending_chunks(chunks)
        ctx_list = schema_ctx.init(name, attrs, chunks)
        self.push(ctx_list)
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        chunks = self._append_pending_chunks(chunks)
        schema_ctx = self.pop()
        postfix = schema_ctx.commit(name, chunks)                
        self._commited_stack.append(schema_ctx)        
        return postfix
    
        
    def append(self, chunks: list[BlockChunk]) -> list[BlockChunk]:
        if self._pending_pop is not None:
            if chunks[0] != "\n":
                self._pending_pop = None
            if self.is_root():
                self.top().add_newline()
        if all(chunk.isspace() for chunk in chunks):            
            self._pending_chunks.extend(chunks)
            return []
        else:
            chunks = self._append_pending_chunks(chunks)
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
        self.schema = schema.extract_schema(style="xml", root=self._root_tag, role="assistant")
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
        self._chunk_cursor: int = 0  # Track last consumed chunk for O(N) iteration

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
            self._wrapper_schema = BlockSchema(name=self._root_tag, style="block", is_root=True, role="assistant")
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
        """Get chunks overlapping the byte range [start, end).

        Uses cursor tracking for O(N) total iteration instead of O(N²).
        Since the parser processes sequentially, we can skip fully consumed chunks.
        """
        result = []
        chunks = self._chunks
        n = len(chunks)
        i = self._chunk_cursor

        # Skip chunks that are fully consumed (end before our range starts)
        while i < n and chunks[i][1] <= start:
            i += 1

        # Update cursor to skip fully consumed chunks on next call
        self._chunk_cursor = i

        # Collect overlapping chunks
        while i < n:
            chunk_start, chunk_end, chunk = chunks[i]

            # Stop if chunk starts at or after our range end
            if chunk_start >= end:
                break

            # Check if we need to split the chunk
            if chunk_end > end:
                # Split off the part after our range
                chunk, _ = chunk.split(end - chunk_start)
            if chunk_start < start:
                # Split off the part before our range
                _, chunk = chunk.split(start - chunk_start)
            if chunk.content:
                if chunk.starts_with_tab():
                    lchunk, chunk = chunk.split_tab()
                    result.append(lchunk)                    
                result.append(chunk)

            i += 1

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
                print(f"####################### {event_data} #########################")
            

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
        output = self._ctx_stack.append(chunks)
        if output:
            if isinstance(output, Block):
                self._emit_event("block", output)
            else:
                self._emit_event("block_delta", output)


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
        # path = self._ctx_stack.top().block.path
        path = self._ctx_stack.top().block.tail.path
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


# =============================================================================
# MarkdownParser
# =============================================================================


class MdBlockCtx:
    def __init__(self, tag_name: str, style: str, depth: int):
        self.block = Block(style=style)
        self._should_add_newline = False
        self._tag_name = tag_name
        self.depth = depth
        
    
        
    def add_newline(self):
        block = self.block.append_child()
        self._should_add_newline = False
        return block
    
    
    def append(self, chunks: list[BlockChunk]) -> list[BlockChunk]:
        for chunk in chunks:
            if chunk.content:
                self.block.tail.append(chunk.content, style=chunk.style, logprob=chunk.logprob)                
        return chunks

class MdContextStack:
    def __init__(self):
        self._stack: list[MdBlockCtx] = []
        self._committed_stack: list[MdBlockCtx] = []
        self.init("root", {}, [], "root", 0)
        
        
    def add_newline(self):
        block = self.top().append_child()
        self._should_add_newline = False
        return block


    def push(self, block_ctx: MdBlockCtx):
        if self._stack:
            self._stack[-1].block.append_child(block_ctx.block)
        self._stack.append(block_ctx)

    def pop(self) -> MdBlockCtx:
        return self._stack.pop()
    
    def is_empty(self) -> bool:
        return not self._stack
    
    def top(self) -> MdBlockCtx:
        return self._stack[-1]
    
    def result(self) -> Block:
        return self._committed_stack[-1].block
    
    def init(self, tag_name: str, attrs: dict, chunks: list[BlockChunk], style: str, depth: int):
        block_ctx = MdBlockCtx(tag_name, style, depth)
        self.push(block_ctx)
    
    
    def commit(self, tag_name: str, chunks: list[BlockChunk]):
        pass
    
    def append(self, chunks: list[BlockChunk]) -> list[BlockChunk]:
        self.top().append(chunks)
        return chunks
    
    def result(self) -> Block:
        pass
    
    def top_event_block(self) -> Block:
        pass


class MarkdownParser:
    """
    Streaming markdown parser with incremental token processing.

    Uses markdown-it-py for parsing. Tracks state to emit events only for
    NEW tokens - open/close/inline handlers are called once per event.

    Accepts BlockChunk objects and maps events to their corresponding chunks.

    Token nesting in markdown-it-py:
    - nesting=1  → OPEN (like XML start tag)
    - nesting=-1 → CLOSE (like XML end tag)
    - nesting=0  → self-closing (inline content)

    Example:
        parser = MarkdownParser()
        parser.feed(BlockChunk("# Hello"))
        parser.feed(BlockChunk("\\n\\nWorld"))
        parser.close()
    """

    def __init__(self, verbose: bool = False):
        from markdown_it import MarkdownIt

        self._md = MarkdownIt()
        self._verbose = verbose

        # Buffer for accumulating content (string)
        self._buffer: str = ""

        # Chunk tracking for mapping events to BlockChunks
        # Each entry is (start_byte, end_byte, BlockChunk)
        self._chunks: list[tuple[int, int, BlockChunk]] = []
        self._total_bytes: int = 0

        # Stack for tracking currently open blocks
        # Each entry is (tag_name, inline_content_so_far, content_start_byte)
        self._stack: list[tuple[str, str, int]] = []

        # Track committed (closed) blocks count
        self._committed_count: int = 0

        # Track last inline content for the current open block
        self._last_inline_content: str = ""

        # Parser closed flag
        self._is_closed: bool = False
        self._ctx_stack: MdContextStack = MdContextStack()
        self._index: int = 0
        

    def feed(self, chunk: BlockChunk | str, logprob: float | None = None) -> None:
        """
        Feed a chunk to the parser.

        Args:
            chunk: BlockChunk or string content to parse
            logprob: Optional log probability (used if chunk is a string)
        """
        if self._is_closed:
            raise ParserError("Parser is closed")

        # Convert string to BlockChunk if needed
        if isinstance(chunk, str):
            chunk = BlockChunk(chunk, logprob=logprob)

        # Split chunk at newlines and store each sub-chunk separately
        for sub_chunk in self._split_chunk_at_newlines(chunk):
            # print(sub_chunk)
            data = sub_chunk.content.encode("utf-8")
            start = self._total_bytes
            end = start + len(data)
            self._chunks.append((start, end, sub_chunk))
            self._total_bytes = end
            self._buffer += sub_chunk.content

        self._parse_incremental()
        print("--------------------------------")

    def close(self) -> None:
        """Close the parser and finalize."""
        if self._is_closed:
            return

        self._is_closed = True
        self._parse_final()

    # =========================================================================
    # Chunk tracking and retrieval
    # =========================================================================

    def _get_byte_position_for_char_index(self, char_index: int) -> int:
        """Convert character index in buffer to byte position."""
        return len(self._buffer[:char_index].encode("utf-8"))

    def _get_chunks_in_range(self, start: int, end: int) -> list[BlockChunk]:
        """
        Get chunks overlapping the byte range [start, end).

        Similar to XmlParser._get_chunks_in_range.
        Splits chunks at newline boundaries.
        """
        result = []

        for chunk_start, chunk_end, chunk in self._chunks:
            # Check if chunk overlaps with [start, end)
            if chunk_start < end and chunk_end > start:
                # Calculate the overlap
                overlap_start = max(chunk_start, start)
                overlap_end = min(chunk_end, end)

                if overlap_start == chunk_start and overlap_end == chunk_end:
                    # Full chunk is within range
                    result.append(chunk)
                else:
                    # Need to split the chunk at byte boundaries
                    working_chunk = chunk
                    if chunk_end > end:
                        working_chunk, _ = working_chunk.split(end - chunk_start)
                    if chunk_start < start:
                        _, working_chunk = working_chunk.split(start - chunk_start)
                    if working_chunk.content:
                        result.append(working_chunk)

        return result

    def _split_chunk_at_newlines(self, chunk: BlockChunk) -> list[BlockChunk]:
        """
        Split a chunk at newline boundaries.

        Returns list of chunks, separating newlines from other content.
        E.g., '\\nTh' -> ['\\n', 'Th']
        E.g., 'Hello\\nWorld' -> ['Hello', '\\n', 'World']
        """
        result = []
        content = chunk.content
        current_pos = 0

        while current_pos < len(content):
            newline_pos = content.find('\n', current_pos)

            if newline_pos == -1:
                # No more newlines - add rest of content
                if current_pos < len(content):
                    remaining = content[current_pos:]
                    if remaining:
                        result.append(BlockChunk(
                            remaining,
                            logprob=chunk.logprob,
                            style=chunk.style
                        ))
                break
            else:
                # Found a newline
                # Add content before newline (if any)
                if newline_pos > current_pos:
                    before = content[current_pos:newline_pos]
                    result.append(BlockChunk(
                        before,
                        logprob=chunk.logprob,
                        style=chunk.style
                    ))

                # Add the newline itself
                result.append(BlockChunk(
                    '\n',
                    logprob=chunk.logprob,
                    style=chunk.style
                ))

                current_pos = newline_pos + 1

        return result if result else [chunk]

    def _get_chunks_for_content(self, content: str, search_start: int = 0) -> list[BlockChunk]:
        """
        Get chunks that correspond to the given content string.

        Args:
            content: The content string to find
            search_start: Character index to start searching from

        Returns:
            List of BlockChunks that make up this content
        """
        # Find content in buffer
        char_index = self._buffer.find(content, search_start)
        if char_index == -1:
            return []

        # Convert to byte positions
        start_byte = self._get_byte_position_for_char_index(char_index)
        end_byte = self._get_byte_position_for_char_index(char_index + len(content))

        return self._get_chunks_in_range(start_byte, end_byte)

    def _get_line_byte_range(self, line_start: int, line_end: int) -> tuple[int, int]:
        """
        Get byte range for a line range from token.map.

        Args:
            line_start: Start line number (0-indexed)
            line_end: End line number (exclusive)

        Returns:
            (start_byte, end_byte)
        """
        lines = self._buffer.split("\n")

        # Calculate character index for start of line_start
        char_start = sum(len(lines[i]) + 1 for i in range(line_start))

        # Calculate character index for end of line_end - 1
        char_end = sum(len(lines[i]) + 1 for i in range(line_end))

        start_byte = self._get_byte_position_for_char_index(char_start)
        end_byte = self._get_byte_position_for_char_index(char_end)

        return start_byte, end_byte

    def _parse_incremental(self) -> None:
        """
        Parse and emit only NEW events by diffing against previous state.
        """
        tokens = self._md.parse(self._buffer)
        self._process_tokens_incremental(tokens, is_final=False)

    def _parse_final(self) -> None:
        """Final parse - close any remaining open blocks."""
        tokens = self._md.parse(self._buffer)
        self._process_tokens_incremental(tokens, is_final=True)

    def _process_tokens_incremental(self, tokens: list, is_final: bool) -> None:
        """
        Process tokens incrementally, emitting only new events.

        Strategy:
        1. Count complete blocks (open+inline+close sequences)
        2. Skip already committed blocks
        3. For the current (possibly incomplete) block:
           - Emit OPEN if not yet opened
           - Emit INLINE delta (new content only)
           - Emit CLOSE only if block is complete AND is_final or next block started
        """
        # Parse tokens into block structures
        blocks = self._parse_into_blocks(tokens)

        # Process each block
        for i, block in enumerate(blocks):
            if i < self._committed_count:
                # Already processed and committed
                continue

            is_complete = block["is_complete"]
            tag_name = block["tag_name"]
            inline_content = block["inline_content"]
            open_token = block["open_token"]

            # Check if this block is newly opened
            current_stack_depth = len(self._stack)
            block_index = i - self._committed_count

            # Only emit OPEN when we have inline content (tag is stable by then)
            # For headings: # vs ## vs ### is only known after seeing the space
            needs_open = block_index >= current_stack_depth
            has_inline = bool(inline_content)

            if needs_open and has_inline:
                # Now we have inline content, so the tag is stable - emit OPEN
                open_chunks = []
                if open_token.map:
                    start_byte, end_byte = self._get_line_byte_range(open_token.map[0], open_token.map[0] + 1)
                    open_chunks = self._get_chunks_in_range(start_byte, end_byte)

                self._handle_open(open_token, open_chunks)

                # Track: (tag_name, content_so_far, byte_position_of_content_start, has_emitted_open)
                content_start_byte = self._total_bytes - len(self._buffer.encode("utf-8"))
                if open_token.map:
                    content_start_byte, _ = self._get_line_byte_range(open_token.map[0], open_token.map[1])
                self._stack.append((tag_name, "", content_start_byte))
                self._last_inline_content = ""

            # Emit inline delta if content changed (only if OPEN was emitted)
            if len(self._stack) > 0 and block_index < len(self._stack):
                stack_idx = block_index
                stored_tag, prev_content, content_start_byte = self._stack[stack_idx]
                if inline_content != prev_content:
                    # New content - emit only the delta
                    new_content = inline_content[len(prev_content):]
                    if new_content:
                        # Get chunks for the new content (already split at newlines in feed)
                        full_content_in_buffer = self._buffer.find(inline_content)
                        if full_content_in_buffer != -1:
                            delta_start = full_content_in_buffer + len(prev_content)
                            delta_chunks = self._get_chunks_for_content(new_content, delta_start)
                        else:
                            delta_chunks = []

                        self._handle_inline_delta(new_content, block["inline_token"], delta_chunks)
                    self._stack[stack_idx] = (tag_name, inline_content, content_start_byte)

            # Check if we should close this block
            if is_complete:
                # Block is complete - check if we should commit it
                # Commit if: is_final OR there's a next block
                should_commit = is_final or (i < len(blocks) - 1)
                if should_commit:
                    # If OPEN wasn't emitted yet (no inline content came), emit it now
                    if block_index >= len(self._stack):
                        open_chunks = []
                        if open_token.map:
                            start_byte, end_byte = self._get_line_byte_range(open_token.map[0], open_token.map[0] + 1)
                            open_chunks = self._get_chunks_in_range(start_byte, end_byte)
                        self._handle_open(open_token, open_chunks)
                        content_start_byte = self._total_bytes - len(self._buffer.encode("utf-8"))
                        if open_token.map:
                            content_start_byte, _ = self._get_line_byte_range(open_token.map[0], open_token.map[1])
                        self._stack.append((tag_name, "", content_start_byte))

                    # Get chunks for closing
                    close_chunks = []
                    close_token = block["close_token"]
                    if close_token and close_token.map:
                        start_byte, end_byte = self._get_line_byte_range(close_token.map[0], close_token.map[1])
                        close_chunks = self._get_chunks_in_range(start_byte, end_byte)

                    self._handle_close(close_token, close_chunks)
                    self._stack.pop() if self._stack else None
                    self._committed_count += 1

    def _parse_into_blocks(self, tokens: list) -> list[dict]:
        """
        Parse flat token list into block structures.

        Returns list of:
        {
            "tag_name": str,
            "open_token": token,
            "inline_token": token or None,
            "inline_content": str,
            "close_token": token or None,
            "is_complete": bool
        }
        """
        blocks = []
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token.nesting == 1:
                # Opening token - use token.tag for actual HTML tag (h1, h2, p, etc.)
                tag_name = token.tag if token.tag else token.type.replace("_open", "")
                block = {
                    "tag_name": tag_name,
                    "open_token": token,
                    "inline_token": None,
                    "inline_content": "",
                    "close_token": None,
                    "is_complete": False,
                }

                # Look for inline and close
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j]
                    if next_token.type == "inline":
                        block["inline_token"] = next_token
                        block["inline_content"] = next_token.content
                    elif next_token.nesting == -1 and (next_token.tag == tag_name or next_token.type == f"{token.type.replace('_open', '')}_close"):
                        block["close_token"] = next_token
                        block["is_complete"] = True
                        i = j
                        break
                    elif next_token.nesting == 1:
                        # Nested block - stop here
                        i = j - 1
                        break
                    j += 1

                blocks.append(block)

            elif token.type == "fence" or token.type == "code_block":
                # Self-contained code block
                blocks.append({
                    "tag_name": "code_block",
                    "open_token": token,
                    "inline_token": token,
                    "inline_content": token.content,
                    "close_token": token,
                    "is_complete": True,
                })

            elif token.type == "hr":
                # Self-contained hr
                blocks.append({
                    "tag_name": "hr",
                    "open_token": token,
                    "inline_token": None,
                    "inline_content": "",
                    "close_token": token,
                    "is_complete": True,
                })

            i += 1

        return blocks

    def _process_token(self, token) -> None:
        """
        Route token to appropriate handler (non-incremental).

        Args:
            token: The markdown-it token
        """
        self._index += 1
        chunks = []
        if token.map:
            start_byte, end_byte = self._get_line_byte_range(token.map[0], token.map[1])
            chunks = self._get_chunks_in_range(start_byte, end_byte)

        if token.nesting == 1:
            self._handle_open(token, chunks)
        elif token.nesting == -1:
            self._handle_close(token, chunks)
        elif token.type == "inline":
            self._handle_inline(token, chunks)
        elif token.type == "fence" or token.type == "code_block":
            self._handle_code_block(token, chunks)
        elif token.type == "hr":
            self._handle_hr(token, chunks)

    # =========================================================================
    # Handlers - Override these to build your block structure
    # =========================================================================

    def _handle_open(self, token, chunks: list[BlockChunk]) -> None:
        """
        Handle OPEN token (nesting=1).

        Called ONCE when a block-level element opens.

        Args:
            token: The markdown-it token
            chunks: BlockChunks that make up this opening

        Token attributes:
        - token.type: e.g., "heading_open"
        - token.tag: e.g., "h1", "h2", "p", "blockquote"
        - token.markup: e.g., "#" for h1, "##" for h2
        - token.map: [start_line, end_line]
        """
        # Use token.tag for the actual HTML tag (h1, h2, p, etc.)
        # Fall back to token.type without _open suffix
        tag_name = token.tag if token.tag else token.type.replace("_open", "")
        print(f"{self._index} [OPEN] {tag_name} chunks={[c.content for c in chunks]} nesting={token.nesting}")        
        if token.type == "heading_open":
            depth = int(token.tag.replace("h", ""))
            if not self._ctx_stack.is_empty() and depth <= self._ctx_stack.top().depth:
                self._ctx_stack.pop()
            self._ctx_stack.init(tag_name, {}, chunks, style="md", depth=depth)    
        elif token.type == "paragraph_open":
            self._ctx_stack.init(tag_name, {}, chunks, style="p", depth=token.nesting)
        

    def _handle_close(self, token, chunks: list[BlockChunk]) -> None:
        """
        Handle CLOSE token (nesting=-1).

        Called ONCE when a block-level element closes.

        Args:
            token: The markdown-it token
            chunks: BlockChunks that make up this closing

        Token attributes:
        - token.type: e.g., "heading_close"
        - token.tag: e.g., "h1", "h2", "p", "blockquote"
        """
        # Use token.tag for the actual HTML tag (h1, h2, p, etc.)
        tag_name = token.tag if token and token.tag else (token.type.replace("_close", "") if token else "unknown")
        print(f"{self._index} [CLOSE] {tag_name} chunks={[c.content for c in chunks]}")
        if token.type == "heading_close":
            # self._ctx_stack.
            pass
        elif token.type == "paragraph_close":
            self._ctx_stack.pop()

    def _handle_inline(self, token, chunks: list[BlockChunk]) -> None:
        """
        Handle full INLINE token (for non-incremental use).

        Args:
            token: The markdown-it token
            chunks: BlockChunks that make up this content

        Token attributes:
        - token.content: full text content
        - token.children: list of child tokens
        """
        print(f"{self._index} [INLINE] content={token.content!r} chunks={[c.content for c in chunks]}")
        self._ctx_stack.append(chunks)

    def _handle_inline_delta(self, new_content: str, token, chunks: list[BlockChunk]) -> None:
        """
        Handle INLINE delta - only the NEW content since last parse.

        Args:
            new_content: Only the new text (delta from previous)
            token: The full inline token (for context)
            chunks: BlockChunks that make up the NEW content only
        """
        print(f"{self._index} [INLINE_DELTA] +{new_content!r} chunks={[c.content for c in chunks]}")
        self._ctx_stack.append(chunks)

    def _handle_code_block(self, token, chunks: list[BlockChunk]) -> None:
        """
        Handle CODE BLOCK token (fence or code_block).

        Args:
            token: The markdown-it token
            chunks: BlockChunks that make up this code block

        Token attributes:
        - token.info: language (e.g., "python")
        - token.content: the code content
        - token.markup: "```" or "~~~"
        """
        print(f"[CODE_BLOCK] lang={token.info!r} chunks={[c.content for c in chunks]}")

    def _handle_hr(self, token, chunks: list[BlockChunk]) -> None:
        """
        Handle HORIZONTAL RULE token.

        Args:
            token: The markdown-it token
            chunks: BlockChunks that make up this hr

        Token attributes:
        - token.markup: "---" or "***" or "___"
        """
        print(f"[HR] markup={token.markup!r} chunks={[c.content for c in chunks]}")
