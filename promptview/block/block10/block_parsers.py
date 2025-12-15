from ...prompt.fbp_process import Process
from .block import BlockSchema, BlockBase, ContentType, BlockChunk, BlockListSchema, BlockList
from .block_builder import BlockBuilderContext








class ParserError(Exception):
    pass


class XmlParser(Process):
    def __init__(self, schema: BlockSchema):
        from xml.parsers import expat
        super().__init__()
        self.context = BlockBuilderContext(schema)
        # self.build_ctx = StreamingBlockBuilder(schema)
        self.parser = expat.ParserCreate()
        self.parser.buffer_text = False
        self.parser.StartElementHandler = self._on_start
        self.parser.EndElementHandler = self._on_end
        self.parser.CharacterDataHandler = self._on_chardata
        self.root_tag = "_root_tag_"
        
        self.chunks = []  # (start_byte, end_byte, chunk)
        self.total_bytes = 0
        self.pending = None  # (event_type, event_data, start_byte)
        self.chunk_queue = []
        self._tag_path = []
        self._has_synthetic_root_tag = False
        if self.context.schema.is_wrapper:
            # self.feed(BlockChunk(content=f"<{self.context.schema.name}>"))
            self.feed(BlockChunk(content=f"<{self.root_tag}>"))
            self._has_synthetic_root_tag = True
        
        
    @property
    def result(self):
        return self.context._root
    
    def feed(self, chunk: "BlockChunk", isfinal=False):
        from xml.parsers.expat import ExpatError
        # print(chunk.content)
        # data = chunk.content.encode() if isinstance(chunk.data, str) else chunk.data
        data = chunk.content.encode("utf-8")
        start = self.total_bytes
        end = start + len(data)
        self.chunks.append((start, end, chunk))
        self.total_bytes = end
        try:
            self.parser.Parse(data, isfinal)
        except ExpatError as e:
            current_data = "".join([c[2].content for c in self.chunks])
            if e.code == 4:
                raise ParserError(f"Invalid XML token: {data}. \nCurrent data: {current_data}")
            elif e.code == 9:
                raise ParserError(f"Unexpected end of file: {data}. \nCurrent data: {current_data}")
            else:
                raise e

        
        
    def _push_block(self, block: "BlockBase"):
        self.chunk_queue.append(block)
        
    def _push_block_list(self, blocks: list["BlockBase"]):
        self.chunk_queue.extend(blocks)
        
    def _pop_block(self):
        return self.chunk_queue.pop(0)
    
    def _has_outputs(self):
        return len(self.chunk_queue) > 0
    
    def close(self):
        if self._has_synthetic_root_tag:
            # self.feed(BlockChunk(content=f"</{self.context.schema.name}>"))        
            self.feed(BlockChunk(content=f"</{self.root_tag}>"))        
        self.parser.Parse(b'', True)
        # Flush any pending event
        # self._flush_pending(self.total_bytes)
    
    def _get_chunks_in_range(self, start, end):
        """
        Return all chunks overlapping [start, end), splitting chunks at boundaries.

        When a chunk partially overlaps the range, it is split so that only the
        portion within [start, end) is returned. This handles cases where the LLM
        returns mixed content like "Hello<tag>" in a single chunk.

        The original chunks in self.chunks remain intact - only the returned
        list contains the split portions.

        Args:
            start: Start byte position (inclusive)
            end: End byte position (exclusive)

        Returns:
            List of BlockChunk objects, potentially split at byte boundaries
        """

        result = []
        for chunk_start, chunk_end, chunk in self.chunks:
            # Check if chunk overlaps with [start, end)
            if chunk_start < end and chunk_end > start:
                # Calculate the overlap
                overlap_start = max(chunk_start, start)
                overlap_end = min(chunk_end, end)

                # Check if we need to split (chunk extends beyond the range)
                if overlap_start == chunk_start and overlap_end == chunk_end:
                    # Full chunk is within range, no split needed
                    result.append(chunk)
                else:
                    # Need to split the chunk - extract only the overlapping portion
                    content_bytes = chunk.content.encode("utf-8")

                    # Calculate byte offsets relative to chunk start
                    slice_start = overlap_start - chunk_start
                    slice_end = overlap_end - chunk_start

                    # Adjust slice boundaries to respect UTF-8 character boundaries
                    # UTF-8 continuation bytes start with 10xxxxxx (0x80-0xBF)
                    # Move slice_start forward to skip continuation bytes
                    while slice_start < len(content_bytes) and (content_bytes[slice_start] & 0xC0) == 0x80:
                        slice_start += 1

                    # Move slice_end forward to include full character
                    while slice_end < len(content_bytes) and (content_bytes[slice_end] & 0xC0) == 0x80:
                        slice_end += 1

                    # Extract the slice and decode back to string
                    sliced_bytes = content_bytes[slice_start:slice_end]

                    # Skip if slice is empty after boundary adjustment
                    if not sliced_bytes:
                        continue

                    sliced_content = sliced_bytes.decode("utf-8")

                    # Create a new chunk with the sliced content
                    # Preserve logprob from original chunk
                    split_chunk = BlockChunk(
                        content=sliced_content,
                        logprob=chunk.logprob,
                        # prefix=chunk.prefix if slice_start == 0 else "",
                        # postfix=chunk.postfix if slice_end == len(content_bytes) else "",
                    )
                    result.append(split_chunk)
        return result
    

    def _flush_pending(self, end_byte):
        if self.pending is None:
            return
        
        event_type, event_data, start_byte = self.pending
        chunks = self._get_chunks_in_range(start_byte, end_byte)
        metas = [c.logprob for c in chunks]
        print(event_type, [repr(c.content) for c in chunks])
        if event_type == 'start':
            name, attrs = event_data
            if name == self.root_tag:
                return
            block = self.context.instantiate(name, chunks, attrs=attrs, ignore_style=True, ignore_name=True)
            self._push_block(block)
            # self._push_block_list(blocks)
            # print(f"StartElement '{name}' {attrs or ''} from chunks: {metas}")
        elif event_type == 'end':
            view = self.context.commit(chunks)
            self._push_block(view)
            # self._push_block(view.postfix)
            # self.build_ctx.commit_view()
            # print(f"EndElement '{event_data}' from chunks: {metas}")
        elif event_type == 'chardata':            
            for chunk in chunks:
                cb = self.context.append(chunk)
                self._push_block(cb)
            # print(f"CharData {repr(event_data)} from chunks: {metas}")
        
        self.pending = None
    
    
    def _on_start(self, name, attrs):
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._tag_path.append(name)
        self.pending = ('start', (name, attrs), current_pos)
    
    def _on_end(self, name):
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._tag_path.pop()
        self.pending = ('end', name, current_pos)
    
    def _on_chardata(self, data):
        # print(f"chardata: '{data}'")
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        # For chardata we could compute end directly, but for consistency
        # we'll use the deferred approach too
        self.pending = ('chardata', data, current_pos)
    
    # def _on_start(self, name, attrs):
    #     print("instantiate",f"'{name}'", attrs)
    #     self.context.instantiate(name, attrs)
        
    #     # self.pending = ('start', (name, attrs), current_pos)
    
    # def _on_end(self, name):
    #     print("commit", f"'{name}'")
    #     current_pos = self.parser.CurrentByteIndex
    #     self.context.commit()
    #     # self.pending = ('end', name, current_pos)
    
    # def _on_chardata(self, data):
    #     print("append", repr(data))
    #     current_pos = self.parser.CurrentByteIndex
    #     self.context.append(data)
        
        
    async def on_stop(self):
        self.close()

    
    async def __anext__(self):
        while not self._has_outputs():
            value = await super().__anext__()
            self.feed(value)        
        block = self._pop_block()
        return block
