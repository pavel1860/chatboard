from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from xml.parsers import expat


from .block import Block
from .schema import BlockListSchema, BlockList
from .span import BlockChunk, BlockChunkList, BlockSpanEvent
from ...prompt.fbp_process import Process

if TYPE_CHECKING:
    from .schema import BlockSchema, BlockListSchema


class ParserError(Exception):
    """Error during parsing."""
    pass


@dataclass
class ParserEvent:
    path: str
    type: Literal["block_stream", "block_init", "block_commit", "block_delta", "block"] | BlockSpanEvent
    value: Block | BlockChunkList
    
    
# ParserState = Literal["start_tag_prefix", "start_tag_content", "start_tag_postfix", "tag_body", "end_tag_prefix", "end_tag_content", "end_tag_postfix"]
ParserState = Literal["start", "body", "end"]

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
        parser.feed_str("<response>")
        parser.feed_str("<thinking>Let me think...</thinking>")
        parser.close()

        # Or in a pipeline
        stream = Stream(chunk_gen())
        parser = XmlParser(schema)
        async for block in stream | parser:
            print(block)
    """

    def __init__(self, schema: "BlockSchema", upstream: Process | None = None, verbose: bool = False):
        super().__init__(upstream)
        self.schema = schema.extract_schema(style="xml")
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
        self._chunks: list[tuple[int, int, BlockChunk]] = []  # (start_byte, end_byte, chunk)
        self._total_bytes = 0

        # Pending event for deferred processing
        self._pending: tuple[str, Any, int] | None = None
        
        self._pending_event: tuple[Literal["start", "end"], Any, list[BlockChunk]] | None = None
        self._pending_prefix: list[BlockChunk] = []
        self._pending_has_ending = False
        self._in_text_chunks = False
        self._temp_block: Block | None = None

        # Output queue for blocks
        self._output_queue: list[ParserEvent] = []

        # Synthetic root tag handling - always use synthetic root for consistent parsing
        self._root_tag = "_root_tag_"
        self._has_synthetic_root = True
        self._state: ParserState | None = None
        self._index = 0
        
        self._is_stream_exhausted = False
        self._last_data_type = None

        # Buffer for incomplete escape sequences (e.g., '\' at end of chunk)
        self._escape_buffer: BlockChunk | None = None

        # Create wrapper schema that contains the real schema as child
        # But if schema is already a wrapper (no name), use it directly
        from .schema import BlockSchema
        if self.schema.name is None or (self.schema.span and self.schema.span.is_empty):
            # Schema is already a wrapper - use it as the root schema
            self._wrapper_schema = self.schema
        else:
            # Schema has content - wrap it with RootMutator
            self._wrapper_schema = BlockSchema(name=self._root_tag, style="root", _auto_handle=False)
            # self._wrapper_schema.children.append(self.schema)
            self._wrapper_schema.append_child(self.schema)
        # self._wrapper_schema = BlockSchema(name=self._root_tag, style="root")
        # self._wrapper_schema.children.append(self.schema)

        self.feed_str(f"<{self._root_tag}>")

    @property
    def result(self) -> Block | None:
        """Get the built block tree, unwrapping synthetic root if needed."""
        if self._root is None:
            return None
        # Unwrap synthetic root if it has exactly one child
        # if self._has_synthetic_root and len(self._root.body) == 1:
        #     return self._root.children[1]
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
    # stack management
    # -------------------------------------------------------------------------
    
    def _pop(self) -> tuple["BlockSchema", Block]:
        return self._stack.pop()
    
    def _push(self, schema: "BlockSchema", block: Block):
        self._stack.append((schema, block))
    
    def _top(self) -> tuple["BlockSchema", Block]:
        return self._stack[-1]
    
    def _top_or_none(self) -> tuple["BlockSchema", Block] | None:
        if not self._stack:
            return None
        return self._stack[-1]
    
    
    def _push_event(self, path: str, type: Literal["block_init", "block_commit", "block_delta"], value: Block | BlockChunkList):
        self._output_queue.append(ParserEvent(path=path, type=type, value=value))
    
    
    def _should_pop(self, chunks: list[BlockChunk]) -> bool:
        if not self._stack:
            return False
        schema, block = self._top()
        return block.mutator.is_last_block_open(chunks)
        
        

    # -------------------------------------------------------------------------
    # Process interface
    # -------------------------------------------------------------------------

    async def on_stop(self):
        """Called when upstream is exhausted - finalize parsing."""
        self.close()

    async def __anext__(self):
        """
        Get next block from parser.

        Consumes chunks from upstream until a block is ready to output.
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
                    self.feed(BlockChunk(content=chunk.content))
                else:
                    self.feed(BlockChunk(content=str(chunk)))
            except StopAsyncIteration:
                # Upstream exhausted
                self._is_stream_exhausted = True
                if self._output_queue:
                    return self._output_queue.pop(0)

        return self._output_queue.pop(0)

    # -------------------------------------------------------------------------
    # Feeding data
    # -------------------------------------------------------------------------

    def feed(self, chunk: BlockChunk, is_final: bool = False):
        """
        Feed a chunk to the parser.

        Args:
            chunk: Chunk with content (and optional logprob)
            is_final: Whether this is the last chunk
        """
        # Handle escape sequence buffering
        if self._escape_buffer is not None:
            # Merge buffered content with current chunk
            merged_content = self._escape_buffer.content + chunk.content
            # Use the logprob from the current chunk (or could average them)
            chunk = BlockChunk(content=merged_content, logprob=chunk.logprob)
            self._escape_buffer = None

        # Check if chunk ends with backslash (incomplete escape sequence)
        if not is_final and chunk.content.endswith("\\"):
            self._escape_buffer = chunk
            return

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

    def feed_str(self, text: str, is_final: bool = False):
        """Feed a string to the parser (convenience method)."""
        self.feed(BlockChunk(content=text), is_final)

    def close(self):
        """Close the parser and finalize the block tree."""
        # Flush any buffered escape sequence before closing
        if self._escape_buffer is not None:
            buffered = self._escape_buffer
            self._escape_buffer = None
            data = buffered.content.encode("utf-8")
            start = self._total_bytes
            end = start + len(data)
            self._chunks.append((start, end, buffered))
            self._total_bytes = end
            self._parser.Parse(data, False)

        if self._has_synthetic_root:
            self.feed_str(f"</{self._root_tag}>")
        self._parser.Parse(b"", True)
        self._flush_pending(self._total_bytes)

    # -------------------------------------------------------------------------
    # Schema lookup
    # -------------------------------------------------------------------------

    def _get_child_schema(self, name: str) -> "BlockSchema | None":
        """
        Find a child schema by name.

        Searches current schema's children for a matching name/tag.
        """
        from .schema import BlockSchema

        if self.current_schema is None:
            # At root level - check if name matches root schema
            if name in self.schema.tags or name == self.schema.name:
                return self.schema
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
        
        # print(f"---------------[{start}, {end}]-------------------")
        def print_chunks(chunk, chunk_start, chunk_end):
            start_sign = "<" if chunk_start < start else "="
            end_sign = ">" if chunk_end > end else "="
            start_cond = "√" if chunk_start < end else "x"
            end_cond = "√" if chunk_end > start else "x"
            cond = "√" if chunk_start < end and chunk_end > start else "x"
            split_end = "split end" if chunk_end > end else ""
            split_start = "split start" if chunk_start < start else ""
                
            # print(chunk_start,"<", f"'{end}'", start_cond, "&",  chunk_end, ">", f"'{start}'", end_cond, "|", chunk, cond, split_start, split_end)         
        for chunk_start, chunk_end, chunk in self._chunks:
            print_chunks(chunk, chunk_start, chunk_end)
            if chunk_start < end and chunk_end > start:
                if chunk_end > end:
                    # chunk, _ = chunk.split(chunk_end - end)
                    chunk, _ = chunk.split(end - chunk_start)
                if chunk_start < start:
                    # _, chunk = chunk.split(start - chunk_start)
                    _, chunk = chunk.split(start - chunk_start)
                result.append(chunk)
                
        return result

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------
    
            
    def _stage_event(self, event_type: Literal["start", "end", "chardata"], event_data: tuple[str, dict | None] | str, chunks: list[BlockChunk]) -> tuple[Literal["start", "end", "chardata", "start_postfix", "end_postfix", "root_text", "body"] | None, Any, list[BlockChunk]]:
        
        # def get_chunk_kind(chunks: list[BlockChunk]) -> Literal["text", "space", "newline"]:
        chunk_kind=None
        if event_type == "chardata":
            if all(chunk.is_line_end for chunk in chunks):
                chunk_kind = "newline"
            elif all(chunk.isspace() for chunk in chunks):
                chunk_kind = "space"
            else:
                chunk_kind = "text"
                
        def add_prefix(chunks: list[BlockChunk]):
            chunks = self._pending_prefix + chunks
            self._pending_prefix = []
            return chunks
         
        if chunk_kind == "space":
            self._pending_prefix.extend(chunks)
            return None, None, []
            
        if self._state is None:
            if event_type == "start":
                self._state = "start"
                return "start", event_data, add_prefix(chunks)
            elif event_type == "end":
                self._state = "end"
                raise ParserError(f"Unexpected end event at root level")
            else:
                return "root_text", None, add_prefix(chunks)
        elif self._state == "start":
            if event_type == "start":
                return "start", event_data, add_prefix(chunks)
            elif event_type == "end":
                return "end", event_data, add_prefix(chunks)
            else: # chardata
                if chunk_kind == "newline":
                    return "start_postfix", event_data, add_prefix(chunks)
                else:
                    self._state = "body"
                    return "body", event_data, add_prefix(chunks)
                
        elif self._state == "body":
            if event_type == "start":
                self._state = "start"
                return "start", event_data, add_prefix(chunks)
            elif event_type == "end":
                self._state = "end"
                return "end", event_data, add_prefix(chunks)
            else: # chardata
                return "body", event_data, add_prefix(chunks)
        elif self._state == "end":
            if event_type == "start":
                self._state = "start"
                self._pop()
                return "start", event_data, add_prefix(chunks)
            elif event_type == "end":
                self._state = "end"
                self._pop()
                return "end", event_data, add_prefix(chunks)
            else: # chardata
                if chunk_kind == "newline":
                    return "end_postfix", event_data, add_prefix(chunks)
                else:
                    if len(self._stack) <= 2:
                        self._pop()
                        return "body", event_data, add_prefix(chunks)
                    # elif len(self._stack) == 1:
                    #     self._pop()
                    #     return "body", event_data, add_prefix(chunks)
                    else:
                        raise ParserError(f"Unexpected character data at end level: {event_data}")

    def _flush_pending(self, end_byte: int):
        """Process any pending event."""
        if self._pending is None:
            return
        self._index += 1
        # print(self._index, self._pending)
        event_type, event_data, start_byte = self._pending
        chunks = self._get_chunks_in_range(start_byte, end_byte)
        self._pending = None

        if not chunks:
            return
        
        if self._verbose and event_type != "chardata":
            print(f"############# {event_type} - {repr(event_data)} ############")
        # print(self._index, event_type, repr(event_data), chunks)
        event_type, event_data, chunks = self._stage_event(event_type, event_data, chunks)
        if event_type is None:
            return
        if self._verbose:
            print(self._index,self._state,")", event_type, repr(event_data), chunks)            
            # print("Handling event:", event_type, event_data, chunks)
            
        # if self._verbose and self.current_block is not None:
        #     print("---------------------------------")
        #     self.current_block.print_debug()
        #     print("---------------------------------")
        if event_type == "start":
            name, attrs = event_data
            self._handle_start(name, attrs, chunks)
        elif event_type == "end":
            name = event_data
            self._handle_end(name, chunks)
        elif event_type == "body":
            self._handle_chardata(chunks)
        elif event_type == "start_postfix":
            self._handle_start_postfix(chunks)
        elif event_type == "end_postfix":
            self._handle_end_postfix(chunks)
        elif event_type == "root_text":
            self._handle_root_text(chunks)
            
        if self._verbose and self._stack:
            print("---------------------------------")
            self._stack[0][1].print()
            print("---------------------------------")


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
        # print(f"{repr(data)}")
        current_pos = self._parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._pending = ("chardata", data, current_pos)
        
    def _try_append_to_block_postfix(self, chunks: list[BlockChunk]):
        schema, block = self._top()
        postfix = block.mutator.block_postfix
        if postfix is None:
            return False
        if all(chunk.is_line_end for chunk in chunks):
            postfix.append_postfix(chunks)
            return True
        return False
        
        

    # -------------------------------------------------------------------------
    # Block building
    # -------------------------------------------------------------------------
    
    def _push_block(self, schema: BlockSchema, block: Block):
        # Append to current block
        if self.current_block is not None:
            self.current_block.append_child(block)
        else:
            self._root = block

        # Push to stack
        self._stack.append((schema, block))
        return block
    
    def _is_top_list_schema(self) -> bool:
        if not self._stack:
            return False
        return isinstance(self._stack[-1][0], BlockListSchema)

    def _handle_start(self, name: str, attrs: dict, chunks: list[BlockChunk]):
        """Handle opening tag - instantiate block from schema."""

        # Handle synthetic root
        if name == self._root_tag:
            # Initialize root block with wrapper schema
            self._root = self._wrapper_schema.instantiate_partial()
            # self._stack.append((self._wrapper_schema, self._root))
            self._push_block(self._wrapper_schema, self._root)
            self._push_event(self._root.path.indices_str(),"block_init", self._root.extract())
            # self._push_event(self._root.path.indices_str(),"block_init", self._root.copy(False))
            return

        if self._is_top_list_schema():
            name = attrs["name"]

        # Find child schema
        child_schema = self._get_child_schema(name)
        if child_schema is None:
            raise ParserError(f"Unknown tag '{name}' - no matching schema found")


        if isinstance(child_schema, BlockListSchema):
            if self._stack[-1][0] != child_schema:
                child_block = child_schema.instantiate(chunks)
                child_block =self._push_block(child_schema, child_block)
                # Queue init event for list schema block
                self._push_event(child_block.path.indices_str(),"block_init", child_block.extract())
            name = attrs["name"]
            child_schema = self._get_child_schema(name)
            if child_schema is None:
                raise ParserError(f"Unknown tag '{name}' - no matching schema found")


        # Instantiate block from schema
        child_block = child_schema.instantiate_partial(chunks)

        # Set attributes if any
        if attrs and hasattr(child_block, 'attrs'):
            child_block.attrs = attrs

        # Append to current block
        child_block = self._push_block(child_schema, child_block)

        # Queue init event for FBP streaming
        self._push_event(child_block.path.indices_str(),"block_init", child_block.extract())

    def _handle_end(self, name: str, chunks: list[BlockChunk]):
        """Handle closing tag - commit and pop stack."""
        # Skip synthetic root
        if name == self._root_tag:
            return

        if not self._stack:
            raise ParserError(f"Unexpected closing tag '{name}' - stack is empty")

        # Pop from stack
        # schema, block = self._stack.pop()
        schema, block = self._top()

        # Validate name matches
        if name != schema.name and name not in schema.tags:
            raise ParserError(f"Mismatched closing tag: expected '{schema.name}', got '{name}'")

        # Commit could be called here for validation
        end_block = block.commit(chunks)
        self._temp_block = None
        # Queue commit event for FBP streaming
        self._push_event(block.path.indices_str(),"block_commit", end_block.copy_head())

    def _handle_chardata(self, chunks: list[BlockChunk]):
        """Handle character data - append to current block and queue for streaming."""
        if self.current_block is None:
            return

        events = self.current_block.append(chunks)

        # Queue delta event for FBP streaming
        for event in events:
            if isinstance(event, Block):
                self._temp_block = event
                self._push_event(event.path.indices_str(), "block", event.copy())    
            else:
                path = self._temp_block.path if self._temp_block is not None else self.current_block.path
                self._push_event(path.indices_str(), event.event, event)
            # event_type = event.event if isinstance(event, BlockChunkList) else "block"
            
        
        
    def _handle_start_postfix(self, chunks: list[BlockChunk]):
        """Handle postfix - append to current block and queue for streaming."""
        if self.current_block is None:
            return
        event = self.current_block.head.append_postfix(chunks)

        # Queue delta event for FBP streaming
        self._push_event(self.current_block.path.indices_str(),event.event, event)
        
    def _handle_end_postfix(self, chunks: list[BlockChunk]):
        if self.current_block is None:
            return
        event = self.current_block.mutator.block_postfix.append_postfix(chunks)

        # Queue delta event for FBP streaming
        self._push_event(self.current_block.path.indices_str(),event.event, event)
        
    def _handle_root_text(self, chunks: list[BlockChunk]):
        pass
    
