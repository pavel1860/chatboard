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
from .path import IndexPath
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
    type: Literal["block_init", "block_commit", "block_delta", "block"]
    value: Block | list[BlockChunk]
    
    def __init__(self, path: str, type: Literal["block_init", "block_commit", "block_delta", "block"], value: Block | list[BlockChunk]):
        self.path = path
        self.type = type
        # self.value = value
        # if not isinstance(value, Block) and not isinstance(value, list) and not value is None:
            # raise ValueError(f"Unknown value type: {type(value)}")
        if isinstance(value, Block):
            self.value = value.extract()
            # print("-------------------------------")
            # self.value.print_debug()
            if len(self.value.body) > 0:
                raise ParserError(f"Block Event was emitted with non empty body. Path: {path} Block:\n {value.debug_tree()}")
        elif isinstance(value, list):
            self.value = value
        elif isinstance(value, BlockChunk):
            self.value = [value]
        elif value is None:
            self.value = value
        else:
            raise ValueError(f"Unknown value type: {type(value)}")
        
        
    @property
    def int_path(self) -> tuple[int, ...]:
        if not self.path:
            return ()
        return tuple(int(p) for p in self.path.split("."))
    

    def get_value(self) -> Block | list[BlockChunk]:
        if isinstance(self.value, Block):
            return self.value.copy()
        elif isinstance(self.value, list):
            return [c.copy() for c in self.value]
        else:
            raise ValueError(f"Unknown value type: {type(self.value)}")



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
    
    @property
    def path(self) -> str:
        return str(self.block.path)
    
    
    def get_style(self) -> str:
        return self.block.mutator.get_style()

    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> SchemaCtx:
        raise NotImplementedError("init not implemented")
    
    
    def append(self, chunks: list[BlockChunk], style: str | None = None):
        return self.block.append(chunks, style=style)
    
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
            if schema.is_block_type:
                return BlockMarkdownSchemaCtx(schema)
            return BlockSchemaCtx(schema)
        else:
            raise ValueError(f"Unknown schema type: {type(schema)}")
        
    def add_newline(self, chunks: list[BlockChunk] | None = None):
        block = self.block.append_child()
        self._should_add_newline = False
        return block
    
class BlockSchemaCtx(SchemaCtx[BlockSchema]):
    
    
    def get_child_schema(self, name: str, attrs: dict) -> BlockSchema | BlockListSchema:
        schema = self.schema.get_schema(name)
        return schema
           
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> SchemaCtx:
        self._block = self.schema.init_partial(chunks, is_streaming=True)
        return self
    
    
    # def append2(self, chunks: list[BlockChunk]):  
    #     block = None    
    #     for chunk in chunks:
    #         style = None
    #         if chunk.content:                
    #             if self._should_add_newline:
    #                 block = self.add_newline()
    #                 yield block
    #             if chunk.is_newline():
    #                 self._should_add_newline = True
    #                 style = self.block.mutator.styles[0]                    
    #             elif chunk.isspace():
    #                 style = self.block.mutator.styles[0]
    #             elif not self.block.body:
    #                 block = self.add_newline()
    #                 yield block                    
    #             res = self.block.tail.append(chunk.content, logprob=chunk.logprob, style=style or chunk.style)
    #             yield res
                
                    

    
    
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
    
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> SchemaCtx:        
        self._block = BlockList(
            role=self.schema.role,
            tags=list(self.schema.tags),
            style=list(self.schema.style),
            attrs=dict(attrs) if attrs else dict(self.schema.attrs),
        )
        # item_name = self._get_key(attrs)
        # item_ctx = self.build_child_schema(item_name, attrs)
        # item_ctx.init(item_name, attrs, chunks)
        # self._block.append_child(item_ctx.block)
        return self
    
    
    def commit(self, name: str, chunks: list[BlockChunk]) -> list[Block]:
        return None
        postfix = Block(chunks)
        self._should_add_newline = False
        self.block.commit(postfix)
        return postfix
        # events = []
        # postfix = self.block.commit(chunks)
        # events.append(postfix)
        # return events
    
class BlockMarkdownSchemaCtx(BlockSchemaCtx):
    
    def __init__(self, schema: TxSchema, is_root: bool = False):
        super().__init__(schema, is_root)
        self._md_parser = MarkdownParser()
        self._found_content = False
        self._markdown_started = False
        self._markdown_ended = False
        
        
    # @property
    # def block(self) -> Block:
    #     return self._md_parser.result
    def get_style(self) -> str:
        return "md"
    
    
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]) -> SchemaCtx:    
        self._block = self.schema.init_partial(chunks, is_streaming=True)        
        return self

        
        
    def append(self, chunks: list[BlockChunk], style: str | None = None):                      
        for chunk in chunks:
            if self._markdown_ended:
                yield from super().append(chunks) 
                return               
            if not self._found_content:
                if chunk.is_newline():
                    self._found_content = True
                events = self.block.tail.append(chunk.content, logprob=chunk.logprob, style=chunk.style)
                yield events
            else:
                # skip tabs
                # if chunk.starts_with_tab():
                    # continue
                
                if not self._markdown_started:
                    self._markdown_started = True
                    events = self._md_parser.feed(chunk)
                    block = self.block.append_child(self._md_parser.result, copy=False)
                    print("### MD ROOT BLOCK ###")
                    block.print_debug()
                    # yield block
                    if len(block.body):
                        yield block.copy(False)
                        yield block.body[0].copy(False)
                    else:
                        yield block
                # print(chunk)
                else:
                    events = self._md_parser.feed(chunk)
                    for event in events:
                        yield event
        
    
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        postfix = super().commit(name, chunks)
        self._md_parser.close()
        self._markdown_ended = True
        return postfix

# ContextState = Literal["none", "init", "content", "commit"]
ContextState = Literal["wait_for_block", "in_block", "in_markdown"]

class ContextStack:
    
    def __init__(self, schema: "BlockSchema | BlockListSchema", output_queue: list[ParserEvent], root_name: str | None = None, verbose: bool = False, verbose_markdown: bool = False):
        self._stack: list[SchemaCtx] = []
        self._commited_stack: list[SchemaCtx] = []
        self._schema = schema
        self._pending_pop: SchemaCtx | None = None
        self._did_start = False
        self._root_name = root_name
        self._pending_chunks: list[BlockChunk] = []
        self._state: ContextState = "none"
        self.events = output_queue
        self.index = -1
        self._verbose = verbose
        self._verbose_markdown = verbose_markdown
    
    @property
    def curr_block(self) -> Block:
        return self.top().block
    
    def top(self, use_pending: bool = True) -> "SchemaCtx":
        if use_pending and self._pending_pop is not None:
            return self._pending_pop
        return self._stack[-1]
    
    def push(self, schema_ctx: SchemaCtx):
        if self._pending_pop is not None:
            self._pending_pop = None
        if not self.is_empty():
            self.curr_block.append_child(schema_ctx.block)
        self._stack.append(schema_ctx)
        
    def pop(self) -> SchemaCtx:
        schema_ctx = self._stack.pop()
        # if isinstance(schema_ctx, BlockListSchemaCtx):            
            # schema_ctx = self._stack.pop()
            # events.append(schema_ctx)
        self._pending_pop = schema_ctx
        return schema_ctx
    
    
    def set_index(self, index: int):
        self.index = index
        if not self.is_empty() and isinstance(self.top(), BlockMarkdownSchemaCtx):
            self.top()._md_parser.set_index(index)
    
    def is_empty(self) -> bool:
        return not self._stack
    
    def is_root(self) -> bool:
        return len(self._stack) == 1 and self.top().schema == self._schema
    
    
    def _append_pending_chunks(self, chunks: list[BlockChunk]) -> list[BlockChunk]:
        if self._pending_chunks and not isinstance(self.top(), BlockMarkdownSchemaCtx):
        # if self._pending_chunks:
            pending_chunks = []
            style = self.top().block.mutator.get_style()
            for c in self._pending_chunks:
                c.meta.style = style
                pending_chunks.append(c)
            chunks = pending_chunks + chunks
            self._pending_chunks = []
        return chunks
    
    def _push_pending_chunk(self, chunk: BlockChunk):
        self._pending_chunks.append(chunk)
        
        
    def _prepend_pending_chunks(self, block: Block):        
        if self._pending_chunks:
            style = self.top().get_style()
            for chunk in reversed(self._pending_chunks):
                block.prepend(chunk.content, style=style, logprob=chunk.logprob)
            self._pending_chunks = []
        return block
        
    def init(self, name: str, attrs: dict, chunks: list[BlockChunk]):
        if name == self._root_name:
            chunks = []
        self._state = "in_block"
        self._pending_pop = None
        if self.is_empty():
            if self._did_start:
                raise ParserError("Unexpected start tag - stack is empty")
            self._did_start = True
            schema_ctx = BlockSchemaCtx(self._schema, is_root=self._root_name == name)
        else:
            schema_ctx = self.top().build_child_schema(name, attrs)
            
            # handle case when the schema is a list schema
            if isinstance(schema_ctx, BlockListSchemaCtx):
                schema_ctx = schema_ctx.init(name, attrs, chunks)                
                self.push(schema_ctx)
                # events.append(ParserEvent(path=schema_ctx.path, type="block_init", value=schema_ctx.block))
                self._add_event("block_init", schema_ctx.path, schema_ctx.block)
                schema_ctx = self.top().build_child_schema(name, attrs)
            
        # chunks = self._append_pending_chunks(chunks)
        schema_ctx = schema_ctx.init(name, attrs, chunks)
        
        if isinstance(schema_ctx, BlockMarkdownSchemaCtx):
            schema_ctx._md_parser._verbose = self._verbose_markdown
            # schema_ctx._md_parser._verbose = self._verbose
            # self._state = "in_markdown"
                
        self.push(schema_ctx)  
        self._prepend_pending_chunks(schema_ctx.block)
        self._add_event("block_init", schema_ctx.path, schema_ctx.block)
        # events.append(ParserEvent(path=schema_ctx.path, type="block_init", value=schema_ctx.block))
        # return events
    
    def commit(self, name: str, chunks: list[BlockChunk]):
        if name == self._root_name:
            chunks = []
        self._state = "in_block"
        # chunks = self._append_pending_chunks(chunks)
        if self.top(False).schema.name != name:
            if isinstance(self.top(False), BlockListSchemaCtx):
                schema_ctx = self.pop()                
                self._add_event("block_commit", schema_ctx.path, None)
                if self.top(False).schema.name != name:
                    raise ParserError(f"Expected {name} but got {self.top().schema.name}")
            else:
                raise ParserError(f"Expected {name} but got {self.top().schema.name}")
            # print(f"Expected {name} but got {self.top().schema.name}")
        schema_ctx = self.pop()
        
        postfix = schema_ctx.commit(name, chunks)
        self._add_event("block_commit", schema_ctx.path, postfix)
        self._prepend_pending_chunks(postfix)
        self._commited_stack.append(schema_ctx)        
    
    def _add_event(self, type: Literal["block_init", "block_commit", "block_delta", "block"], path: str | IndexPath , value: Block | list[BlockChunk]):
        if isinstance(path, IndexPath):
            path = str(path)
        self.events.append(ParserEvent(path=path, type=type, value=value))
        
        
    def append(self, chunks: list[BlockChunk]):
        top = self.top()
        style = None
        for chunk in chunks:            
            # if chunk.starts_with_tab() and self._state in ["wait_for_block", "in_markdown"]:                
            if chunk.starts_with_tab() and self._state in ["wait_for_block"]:                
                self._push_pending_chunk(chunk)
                continue
            elif chunk.is_newline() and self._state == "in_block":
                # self._state = "wait_for_block" if not isinstance(top, BlockMarkdownSchemaCtx) else "in_markdown"                         
                self._state = "wait_for_block"
                style = top.get_style()
                chunk.meta.style = style
            for event in top.append([chunk], style=style):
                if isinstance(event, Block):
                    self._add_event("block", str(event.path), event)
                    self._prepend_pending_chunks(event)
                    self._state = "in_block"
                    # print(f"append Block!")
                else:
                    self._add_event("block_delta", str(self.top().path), event)
    
    
    
    
    
    
    
    
    # def _append_none(self, chunk: BlockChunk):
    #     if chunk.starts_with_tab():
    #         self._push_pending_chunk(chunk)
    #         return
    #     elif chunk.is_newline():
    #         pass
    #     else:
    #         self._state = "content"
    #         _chunks = self._append_pending_chunks([chunk])
    #         self.top().append(_chunks)                                
            
    # def _append_init(self, chunk: BlockChunk):
    #     top_ctx = self.top()
    #     if chunk.is_newline():
    #         event = top_ctx.append([chunk])
    #         self._add_event("block_init", top_ctx.path, event)
    #         block = top_ctx.add_newline()
    #         self._add_event("block", block.path, block)
    #         self._state = "none"
    #     else:
    #         event = top_ctx.append([chunk])            
    #         _chunks = self._append_pending_chunks([chunk])
    #         events = top_ctx.append(_chunks)    
    #         self._add_event("block_delta", top_ctx.path, events)
    #         self._state = "content"
                                        
    # def _append_content(self, chunk: BlockChunk):
    #     top_ctx = self.top()
    #     if chunk.is_newline():
    #         event = top_ctx.append([chunk])
    #         self._add_event("block_delta", top_ctx.path, event)
    #         block = top_ctx.add_newline()
    #         self._add_event("block", block.path, block)
    #         self._state = "none"
    #     else:                        
    #         _chunks = self._append_pending_chunks([chunk])
    #         top_ctx.append(_chunks)
    #         self._add_event("block_delta", top_ctx.path, _chunks)
    #         self._state = "content"
    
    # def _append_commit(self, chunk: BlockChunk):
    #     top_ctx = self.top()
    #     if chunk.is_newline():
    #         event = top_ctx.append([chunk])
    #         self._add_event("block_delta", top_ctx.path, event)
    #         self._state = "none"
    #     else:
    #         event = top_ctx.append([chunk])
    #         self._add_event("block_delta", top_ctx.path, event)
    #         self._state = "commit"
    #         _chunks = self._append_pending_chunks([chunk])
    #         top_ctx.append(_chunks)
    
    # def append(self, chunks: list[BlockChunk]):
    #     for chunk in chunks:
    #         if self._state == "none":
    #             self._append_none(chunk)
    #         elif self._state == "init":
    #             self._append_init(chunk)
    #         elif self._state == "content":
    #             self._append_content(chunk)
    #         elif self._state == "commit":
    #             self._append_commit(chunk)
    #         else:
    #             raise ParserError(f"Unknown state: {self._state}")
    
    

    
    
        
    def append2(self, chunks: list[BlockChunk]) -> list[ParserEvent]:
        # handle suffix data 
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
        events = []
        schema_ctx = self.top()
        for event in schema_ctx.append(chunks):
            if isinstance(event, Block):
                events.append(ParserEvent(path=str(event.path), type="block", value=event))
            else:
                events.append(ParserEvent(path=str(self.top().block.tail.path), type="block_delta", value=event))
        return events
    
    
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

    def __init__(
        self, 
        schema: "BlockSchema", 
        upstream: Process | None = None, 
        verbose: bool = False,
        exhaust_stream_on_exception: bool = True,
        verbose_debug: bool = False,
        verbose_markdown: bool = False
    ):
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
        self._exhaust_stream_on_exception = exhaust_stream_on_exception
        self._verbose_debug = verbose_debug
        self._verbose_markdown = verbose_markdown
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
        self._index = -1
        self._ctx_stack: ContextStack = ContextStack(self._wrapper_schema, output_queue=self._output_queue, root_name=self._root_tag, verbose=verbose, verbose_markdown=verbose_markdown)
        
        self._exception_to_raise: Exception | None = None

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

    
    def get_chunks(self) -> list[BlockChunk]:
        return [c[2] for c in self._chunks if self._root_tag not in c[2].content]
    # -------------------------------------------------------------------------
    # Process interface
    # -------------------------------------------------------------------------

    async def on_stop(self):
        """Called when upstream is exhausted - finalize parsing."""
        self.close()

    async def _drain_stream(self):
        """Consume remaining upstream chunks without parsing, for reproducibility."""
        try:
            while True:
                upstream_chunk = await super().__anext__()
                if isinstance(upstream_chunk, BlockChunk):
                    chunk = upstream_chunk
                elif hasattr(upstream_chunk, 'content'):
                    chunk = BlockChunk(
                        upstream_chunk.content,
                        logprob=getattr(upstream_chunk, 'logprob', None)
                    )
                else:
                    chunk = BlockChunk(str(upstream_chunk))
                # Store raw chunk data without feeding to parser
                data = chunk.content.encode("utf-8")
                start = self._total_bytes
                end = start + len(data)
                self._chunks.append((start, end, chunk))
                self._total_bytes = end
        except (StopAsyncIteration, Exception):
            self._is_stream_exhausted = True

    async def __anext__(self):
        """
        Get next event from parser.

        Consumes chunks from upstream until an event is ready to output.
        """
        # Re-raise stored exception (stream already drained)
        if self._exception_to_raise:
            raise self._exception_to_raise

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
            except Exception as e:
                if self._exhaust_stream_on_exception:
                    await self._drain_stream()
                    self._exception_to_raise = e
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

        if not self._exception_to_raise:
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
                    if chunk.content:
                        result.append(chunk)
                else:                    
                    result.append(chunk)

            i += 1

        return result

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------
    
    def inc_index(self):
        self._index += 1
        self._ctx_stack.set_index(self._index)

    def _flush_pending(self, end_byte: int):
        """Process any pending event."""
        if self._pending is None:
            return
        self.inc_index()
        event_type, event_data, start_byte = self._pending
        chunks = self._get_chunks_in_range(start_byte, end_byte)
        self._pending = None

        # if not chunks:
        #     return

        if self._verbose:
            if event_type == "start":
                print(f"*********************** {event_data[0]} *************************")
            if event_type in ["chardata", "end"]:
                print("__________________")
            print(f"Event {self._index}: {event_type}, data={event_data!r}, chunks={chunks}")
            if event_type == "end":
                # print(f"####################### {event_data} #########################")
                print(f"^^^^^^^^^^^^^^^^^^^^^^^ {event_data} ^^^^^^^^^^^^^^^^^^^^^^^^")        
            
        if event_type == "start":
            name, attrs = event_data            
            self._ctx_stack.init(name, attrs, chunks)
        elif event_type == "end":
            name = event_data
            self._ctx_stack.commit(name, chunks)
        elif event_type == "chardata":
            self._ctx_stack.append(chunks)
        # print(f"({self._ctx_stack._state}) {self._ctx_stack._pending_chunks}")
        
        
        if self._verbose_debug:
            if not self._ctx_stack.is_empty():
                print("---")                
                self._ctx_stack._stack[0].block.print_debug()

    def _on_start(self, name: str, attrs: dict):
        """Handle start tag event from expat."""
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        # print("start", name, attrs)
        self._pending = ("start", (name, attrs), current_pos)

    def _on_end(self, name: str):
        """Handle end tag event from expat."""
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        # print("end", name)
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
        self._output_queue += self._ctx_stack.init(name, attrs, chunks)
        # self._emit_event("block_init", self._ctx_stack.top_event_block())
        
        
    def _handle_end(self, name: str, chunks: list[BlockChunk]):
        """Handle closing tag - commit and pop stack."""
        if name == self._root_tag:
            chunks = []
        self._output_queue += self._ctx_stack.commit(name, chunks)        
        
        
        
    def _handle_chardata(self, chunks: list[BlockChunk]):
        """Handle character data - append to current block."""
        output = self._ctx_stack.append(chunks)
        self._output_queue += output


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
        prefix_chunks = [
            BlockChunk(c.content, logprob=c.logprob, style="prefix")
            for c in chunks if c.content
        ]
        if prefix_chunks:
            child_block._raw_append(prefix_chunks)

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
        postfix_chunks = [
            BlockChunk(c.content, logprob=c.logprob, style="postfix")
            for c in chunks if c.content
        ]
        if postfix_chunks:
            block._raw_append(postfix_chunks)

        # Pop from stack
        self._stack.pop()

        # Emit commit event
        self._emit_event("block_commit", block, postfix_chunks if postfix_chunks else None)



    def _handle_chardata2(self, chunks: list[BlockChunk]):
        """Handle character data - append to current block."""
        if self.current_block is None:
            return

        # Filter non-empty chunks
        valid_chunks = [c for c in chunks if c.content]
        if valid_chunks:
            result_chunks = self.current_block._raw_append(valid_chunks)
            self._emit_event("block_delta", self.current_block, result_chunks)

    def _emit_event(self, event_type: str, value: Block | list[BlockChunk]):
        """Emit a parser event."""
        # Build path from stack
        # path = "/".join(str(i) for i, _ in enumerate(self._stack))
        # path = self._ctx_stack.top().block.path
        if isinstance(value, Block):
            path = value.path
        else:
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
    def __init__(self, tag_name: str, chunks: list[BlockChunk], attrs: dict, style: str, depth: int):
        from .mutator import MutatorMeta
        config = MutatorMeta.resolve([style])
        self.block = config.create_block(chunks, tags=[tag_name], attrs=attrs)
        # self.block = Block(style=style)
        self._should_add_newline = False
        self._tag_name = tag_name
        self.depth = depth
        self._should_add_newline = False
        
    
        
    def add_newline(self):
        block = self.block.append_child()
        self._should_add_newline = False
        return block
    
    
    def append(self, chunk: BlockChunk) -> list[BlockChunk]:        
        return self.block.tail.append(chunk.content, style=chunk.style, logprob=chunk.logprob)                
        
    
class MdContextStack:
    def __init__(self):
        self._stack: list[MdBlockCtx] = []
        self._committed_stack: list[MdBlockCtx] = []
        self.init("md-root", {}, [], "root", 0)
        self._should_add_newline = False
        
        
    def add_newline(self):
        block = self.top().block.append_child()
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
        # Reset newline flag when starting a new block
        self._should_add_newline = False
        block_ctx = MdBlockCtx(tag_name, chunks, attrs, style, depth)
        self.push(block_ctx)
    
    
    def commit(self, tag_name: str, chunks: list[BlockChunk]):
        if len(self._stack) > 1:  # Don't pop the root md-root
            ctx = self.pop()
            self._committed_stack.append(ctx)
    
    # def append(self, chunks: list[BlockChunk]) -> Block |list[BlockChunk]:
    #     return self.top().append(chunks)
    
    def append(self, chunks: list[BlockChunk]) -> Block | list[BlockChunk]:
        block = None        
        for chunk in chunks:
            style = None
            if self._should_add_newline:
                if len(self._stack) > 1 and "md" not in self.top().block.style:
                    self.pop()
                block = self.add_newline()
            if chunk.is_newline():
                self._should_add_newline = True
                style = self.top().block.mutator.get_style()
            if chunk.content:
                self.top().block.tail.append(chunk.content, style=style or chunk.style, logprob=chunk.logprob)                
        if block:
            return block
        return chunks
    
    def result(self) -> Block:
        return self._stack[0].block
    
    def top_event_block(self) -> Block:
        pass


@dataclass
class MdEvent:
    """Event from markdown state machine."""
    type: Literal["open", "delta", "close"]
    tag: str
    content: str  # The actual content for this event (may be subset of chunk)
    chunk: BlockChunk  # Original chunk (for logprob etc.)
    level: int = 0  # For headings: 1-6


class MarkdownParser:
    """
    Line-based streaming markdown parser.

    Uses a simple state machine instead of re-parsing. Each chunk is processed
    incrementally and events are returned immediately for streaming.

    Supports:
    - Headings: #, ##, ###, etc.
    - Paragraphs: text until blank line

    Example:
        parser = MarkdownParser()
        for chunk in chunks:
            outputs = parser.feed(chunk)
            for output in outputs:
                send_to_frontend(output)
    """

    # States
    STATE_LINE_START = "line_start"
    STATE_IN_MARKUP = "in_markup"
    STATE_IN_CONTENT = "in_content"

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._index = 0
        # State machine
        self._state = self.STATE_LINE_START
        self._markup_buffer = ""  # Accumulated markup (e.g., "##")
        self._current_tag: str | None = None  # Current block tag (h1, h2, p)
        self._current_level: int = 0  # Heading level
        self._pending_newlines: int = 0  # Track newlines for paragraph close

        # Content accumulator for current chunk
        self._content_buffer = ""  # Content accumulated within current chunk

        # Context stack for building blocks
        self._ctx_stack: MdContextStack = MdContextStack()

        # Parser closed flag
        self._is_closed: bool = False

    @property
    def result(self) -> Block:
        return self._ctx_stack.result()

    def feed(self, chunk: BlockChunk | str, logprob: float | None = None) -> list[Block | list[BlockChunk]]:
        """
        Feed a chunk to the parser.

        Args:
            chunk: BlockChunk or string content to parse
            logprob: Optional log probability (used if chunk is a string)

        Returns:
            List of outputs to stream (Block on new structure, chunks for deltas)
        """
        if self._is_closed:
            raise ParserError("Parser is closed")

        if isinstance(chunk, str):
            chunk = BlockChunk(chunk, logprob=logprob)

        outputs = []

        # Split chunk at newlines for consistent handling
        sub_chunks = self._split_at_newlines(chunk)

        for sub_chunk in sub_chunks:
            # Process chunk and generate events
            events = self._process_chunk(sub_chunk)

            # Handle each event and collect outputs
            for event in events:
                output = self._handle_event(event)
                if output is not None:
                    outputs.append(output)

            if self._verbose:
                print(f"[MD] fed {sub_chunk.content!r} → {len(events)} events, {len(outputs)} outputs")

        return outputs

    def _split_at_newlines(self, chunk: BlockChunk) -> list[BlockChunk]:
        """
        Split a chunk at newline boundaries.

        Returns a list of chunks where newlines are separate chunks.
        This ensures consistent handling of structural boundaries.
        """
        content = chunk.content
        if '\n' not in content:
            return [chunk]

        result = []
        current_pos = 0

        for i, char in enumerate(content):
            if char == '\n':
                # Add content before newline (if any)
                if i > current_pos:
                    before_content = content[current_pos:i]
                    before_meta = ChunkMeta(
                        start=chunk.meta.start + current_pos,
                        end=chunk.meta.start + i,
                        logprob=chunk.logprob,
                        style=chunk.style,
                    )
                    result.append(BlockChunk(before_content, meta=before_meta))

                # Add newline as separate chunk
                nl_meta = ChunkMeta(
                    start=chunk.meta.start + i,
                    end=chunk.meta.start + i + 1,
                    logprob=chunk.logprob,
                    style=chunk.style,
                )
                result.append(BlockChunk('\n', meta=nl_meta))
                current_pos = i + 1

        # Add remaining content after last newline (if any)
        if current_pos < len(content):
            after_content = content[current_pos:]
            after_meta = ChunkMeta(
                start=chunk.meta.start + current_pos,
                end=chunk.meta.end,
                logprob=chunk.logprob,
                style=chunk.style,
            )
            result.append(BlockChunk(after_content, meta=after_meta))

        return result
    
    def set_index(self, index: int):
        self._index = index

    def close(self) -> list[Block | list[BlockChunk]]:
        """Close the parser and finalize any open blocks."""
        if self._is_closed:
            return []

        self._is_closed = True
        outputs = []

        # Close any open block
        if self._current_tag:
            close_event = MdEvent("close", self._current_tag, "", BlockChunk(""), self._current_level)
            output = self._handle_event(close_event)
            if output:
                outputs.append(output)

        return outputs

    # =========================================================================
    # State Machine - Event Generation
    # =========================================================================

    def _process_chunk(self, chunk: BlockChunk) -> list[MdEvent]:
        """
        Process a chunk through the state machine.
        Returns list of events with appropriate content slices.
        """
        events = []
        content = chunk.content

        # Reset content buffer for this chunk
        self._content_buffer = ""

        for i, char in enumerate(content):
            char_events = self._process_char(char, chunk)
            events.extend(char_events)

        # If there's accumulated content, emit a final delta
        if self._content_buffer and self._current_tag:
            events.append(MdEvent(
                "delta",
                self._current_tag,
                self._content_buffer,
                BlockChunk(self._content_buffer, logprob=chunk.logprob, style=chunk.style),
                self._current_level
            ))
            self._content_buffer = ""

        return events

    def _process_char(self, char: str, chunk: BlockChunk) -> list[MdEvent]:
        """Process a single character, return events."""
        events = []
        if self._verbose:
            print(f"{self._index}  [ MD CHAR ] {char} state: {self._state}")
        if self._state == self.STATE_LINE_START:
            if char == '#':
                self._markup_buffer += char
                self._state = self.STATE_IN_MARKUP
            elif char == '\n':
                # Blank line
                self._pending_newlines += 1
                if self._current_tag == 'p' and self._pending_newlines >= 1:
                    # Emit any pending content first
                    if self._content_buffer:
                        events.append(MdEvent(
                            "delta", "p", self._content_buffer,
                            BlockChunk(self._content_buffer, logprob=chunk.logprob, style=chunk.style), 0
                        ))
                        self._content_buffer = ""
                    # Close paragraph on blank line
                    events.append(MdEvent("close", "p", "", chunk, 0))
                    self._current_tag = None
                    self._current_level = 0
            else:
                # Start of paragraph
                self._pending_newlines = 0
                if self._current_tag is None:
                    self._current_tag = 'p'
                    events.append(MdEvent("open", "p", "", chunk, 0))
                self._state = self.STATE_IN_CONTENT
                self._content_buffer += char

        elif self._state == self.STATE_IN_MARKUP:
            if char == '#':
                self._markup_buffer += char
            elif char == ' ':
                # End of heading markup: "## " → h2
                level = len(self._markup_buffer)
                tag = f"h{level}"

                # Close previous block if needed
                if self._current_tag:
                    events.append(MdEvent("close", self._current_tag, "", chunk, self._current_level))

                self._current_tag = tag
                self._current_level = level
                markup_content = self._markup_buffer + " "
                self._markup_buffer = ""
                self._pending_newlines = 0
                self._state = self.STATE_IN_CONTENT
                events.append(MdEvent(
                    "open", tag, markup_content,
                    BlockChunk(markup_content, logprob=chunk.logprob, style="md"), level
                ))
            elif char == '\n':
                # Just "#" without space - treat as paragraph content
                self._markup_buffer = ""
                self._state = self.STATE_LINE_START
            else:
                # Not a heading - treat as paragraph
                if self._current_tag is None:
                    self._current_tag = 'p'
                    events.append(MdEvent("open", "p", "", chunk, 0))
                self._state = self.STATE_IN_CONTENT
                self._content_buffer += self._markup_buffer + char
                self._markup_buffer = ""

        elif self._state == self.STATE_IN_CONTENT:
            if char == '\n':
                self._pending_newlines += 1

                if self._current_tag and self._current_tag.startswith('h'):
                    # Include the newline in the content buffer
                    self._content_buffer += char
                    # Emit content including newline
                    if self._content_buffer:
                        events.append(MdEvent(
                            "delta", self._current_tag, self._content_buffer,
                            BlockChunk(self._content_buffer, logprob=chunk.logprob, style=chunk.style),
                            self._current_level
                        ))
                        self._content_buffer = ""
                    # Headings close on newline
                    events.append(MdEvent("close", self._current_tag, "", chunk, self._current_level))
                    self._current_tag = None
                    self._current_level = 0
                    self._state = self.STATE_LINE_START
                else:
                    # Paragraph continues, include newline in content
                    self._content_buffer += char
                    self._state = self.STATE_LINE_START
            else:
                self._pending_newlines = 0
                self._content_buffer += char

        return events

    # =========================================================================
    # Event Handling - Side Effects
    # =========================================================================

    def _handle_event(self, event: MdEvent) -> Block | list[BlockChunk] | None:
        """
        Handle an event by calling ctx_stack methods.
        Returns the output for streaming.
        """
        # self._index += 1
        if self._verbose:
            print(f"{self._index}  [ MD EVENT ] {event.type} {event.tag} content={event.content!r}")

        if event.type == "open":
            return self._handle_open(event)
        elif event.type == "delta":
            return self._handle_delta(event)
        elif event.type == "close":
            return self._handle_close(event)

        return None

    def _handle_open(self, event: MdEvent) -> Block | None:
        """Handle OPEN event - initialize new block."""
        style = "md" if event.tag.startswith('h') else "p"
        # Only pass chunks for elements with markup (headings)
        # Paragraphs have no markup - content comes via delta events
        chunks = [event.chunk] if event.tag.startswith('h') and event.content else []
        self._ctx_stack.init(event.tag, {}, chunks, style, event.level)
        return self._ctx_stack.top().block

    def _handle_delta(self, event: MdEvent) -> list[BlockChunk] | None:
        """Handle DELTA event - append content to current block."""
        # Skip empty deltas
        if not event.content:
            return None
        # Use the event's chunk which contains just the content
        output = self._ctx_stack.append([event.chunk])
        return output

    def _handle_close(self, event: MdEvent) -> Block | None:
        """Handle CLOSE event - finalize block."""
        result = self._ctx_stack.commit(event.tag, [event.chunk])
        return result

