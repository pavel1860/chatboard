from __future__ import annotations
from typing import TYPE_CHECKING, Any
from xml.parsers import expat


from .block import Block
from .schema import BlockListSchema, BlockList
from .span import Chunk
from ...prompt.fbp_process import Process

if TYPE_CHECKING:
    from .schema import BlockSchema, BlockListSchema


class ParserError(Exception):
    """Error during parsing."""
    pass


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

    def __init__(self, schema: "BlockSchema", upstream: Process | None = None):
        super().__init__(upstream)
        self.schema = schema.extract_schema(style="xml")
        self._root: Block | None = None
        self._stack: list[tuple["BlockSchema", Block]] = []

        # Expat parser setup
        self._parser = expat.ParserCreate()
        self._parser.buffer_text = False
        self._parser.StartElementHandler = self._on_start
        self._parser.EndElementHandler = self._on_end
        self._parser.CharacterDataHandler = self._on_chardata

        # Chunk tracking for logprobs
        self._chunks: list[tuple[int, int, Chunk]] = []  # (start_byte, end_byte, chunk)
        self._total_bytes = 0

        # Pending event for deferred processing
        self._pending: tuple[str, Any, int] | None = None

        # Output queue for blocks
        self._output_queue: list[Block] = []

        # Synthetic root tag handling - always use synthetic root for consistent parsing
        self._root_tag = "_root_tag_"
        self._has_synthetic_root = True

        # Create wrapper schema that contains the real schema as child
        # But if schema is already a wrapper (no name), use it directly
        from .schema import BlockSchema
        if self.schema.name is None or (self.schema.span and self.schema.span.is_empty):
            # Schema is already a wrapper - use it as the root schema
            self._wrapper_schema = self.schema
        else:
            # Schema has content - wrap it
            self._wrapper_schema = BlockSchema(name=self._root_tag, style=[])
            self._wrapper_schema.children.append(self.schema)

        self.feed_str(f"<{self._root_tag}>")

    @property
    def result(self) -> Block | None:
        """Get the built block tree, unwrapping synthetic root if needed."""
        if self._root is None:
            return None
        # Unwrap synthetic root if it has exactly one child
        if self._has_synthetic_root and len(self._root.body) == 1:
            return self._root.body[0]
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
        Get next block from parser.

        Consumes chunks from upstream until a block is ready to output.
        """
        # Return queued output if available
        if self._output_queue:
            return self._output_queue.pop(0)

        # Consume upstream chunks until we have output
        while not self._output_queue:
            try:
                chunk = await super().__anext__()
                # Feed the chunk (may produce output)
                if hasattr(chunk, 'content'):
                    self.feed(Chunk(content=chunk.content))
                else:
                    self.feed(Chunk(content=str(chunk)))
            except StopAsyncIteration:
                # Upstream exhausted
                if self._output_queue:
                    return self._output_queue.pop(0)
                raise

        return self._output_queue.pop(0)

    # -------------------------------------------------------------------------
    # Feeding data
    # -------------------------------------------------------------------------

    def feed(self, chunk: Chunk, is_final: bool = False):
        """
        Feed a chunk to the parser.

        Args:
            chunk: Chunk with content (and optional logprob)
            is_final: Whether this is the last chunk
        """
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
        self.feed(Chunk(content=text), is_final)

    def close(self):
        """Close the parser and finalize the block tree."""
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
    
    def _get_chunks_in_range(self, start: int, end: int) -> list[Chunk]:
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
                
            print(chunk_start,"<", f"'{end}'", start_cond, "&",  chunk_end, ">", f"'{start}'", end_cond, "|", chunk, cond, split_start, split_end)         
        for chunk_start, chunk_end, chunk in self._chunks:
            # print_chunks(chunk, chunk_start, chunk_end)
            if chunk_start < end and chunk_end > start:
                if chunk_end > end:
                    # chunk, _ = chunk.split(chunk_end - end)
                    chunk, _ = chunk.split(end - start)
                if chunk_start < start:
                    # _, chunk = chunk.split(start - chunk_start)
                    _, chunk = chunk.split(start - chunk_start)
                result.append(chunk)
                
        return result

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    def _flush_pending(self, end_byte: int):
        """Process any pending event."""
        if self._pending is None:
            return

        event_type, event_data, start_byte = self._pending
        chunks = self._get_chunks_in_range(start_byte, end_byte)
        self._pending = None

        if not chunks:
            return
        print(event_type, repr(event_data), chunks)
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
    
    def _push_block(self, schema: BlockSchema, block: Block):
        # Append to current block
        if self.current_block is not None:
            self.current_block.append_child(block)
        else:
            self._root = block

        # Push to stack
        self._stack.append((schema, block))
        
    def _is_top_list_schema(self) -> bool:
        if not self._stack:
            return False
        return isinstance(self._stack[-1][0], BlockListSchema)

    def _handle_start(self, name: str, attrs: dict, chunks: list[Chunk]):
        """Handle opening tag - instantiate block from schema."""

        # Handle synthetic root
        if name == self._root_tag:
            # Initialize root block with wrapper schema
            self._root = self._wrapper_schema.instantiate_partial()
            self._stack.append((self._wrapper_schema, self._root))
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
                self._push_block(child_schema, child_block)
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
        self._push_block(child_schema, child_block)

    def _handle_end(self, name: str, chunks: list[Chunk]):
        """Handle closing tag - commit and pop stack."""
        # Skip synthetic root
        if name == self._root_tag:
            return

        if not self._stack:
            raise ParserError(f"Unexpected closing tag '{name}' - stack is empty")

        # Pop from stack
        schema, block = self._stack.pop()

        # Validate name matches
        if name != schema.name and name not in schema.tags:
            raise ParserError(f"Mismatched closing tag: expected '{schema.name}', got '{name}'")

        # Commit could be called here for validation
        block.commit(chunks)

    def _handle_chardata(self, chunks: list[Chunk]):
        """Handle character data - append to current block."""
        if self.current_block is None:
            return

        self.current_block.append(chunks)
        # Append content from chunks
        # for chunk in chunks:
        #     # Skip whitespace-only content between tags
        #     self.current_block.append(chunk)
            # if chunk.content.strip():
                # self.current_block.append_content(chunk.content)
