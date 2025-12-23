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
        self.used_chunks = []
        self._last_used_chunk_idx = -1
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
        return self.context.result
    
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
    
    def _last_flush_kind(self):
        if len(self.used_chunks) == 0:
            return None
        return self.used_chunks[-1][0]
    
    def close(self):
        if self._has_synthetic_root_tag:
            # self.feed(BlockChunk(content=f"</{self.context.schema.name}>"))        
            self.feed(BlockChunk(content=f"</{self.root_tag}>"))        
        self.parser.Parse(b'', True)
        # Flush any pending event
        self._flush_pending(self.total_bytes)
    
    def _get_chunks_in_range(self, event_type, start, end):
        """Return all chunks overlapping [start, end)"""
        result = []
        idx = 0
        isspace = False
        has_content = False
        as_child = False
        start_offset = None
        end_offset = None
        prefix_chunks = []
        for chunk_start, chunk_end, chunk in self.chunks:
            if chunk_start < end and chunk_end > start:
                for offset, (c, kind) in enumerate(chunk.iter_kind()):
                    if kind == "newline":
                        isspace = True
                        break
                    elif kind == "space":
                        pass
                    else:
                        has_content = True
                        isspace = False
                if start_offset is None and end < chunk_end:
                    start_offset = end - chunk_start
                if end_offset is None and start > chunk_start:
                    end_offset = start - chunk_start
                result.append((chunk_start, chunk_end, chunk))
                self._last_used_chunk_idx = idx
                idx += 1
                
        if event_type == "start":
            if self._last_flush_kind() == "start":
                as_child = True
        elif event_type == "end":
            pass
        elif event_type == "chardata":
            if self._last_flush_kind() == "start":
                if isspace:
                    as_child = False
                    event_type = "start"
                else:
                    as_child = True   
            elif self._last_flush_kind() == "end":
                has_content = True
                
        self.used_chunks.append((event_type, self.chunks[:idx]))
        # self.chunks = self.chunks[idx:] 
        if not has_content:
            return [], False, None, None    
        return result, as_child, start_offset, end_offset
    
        
    def _get_chunks_in_range4(self, event_type, start, end):
        """Return all chunks overlapping [start, end)"""
        result = []
        idx = 0
        isspace = False
        as_child = False
        for chunk_start, chunk_end, chunk in self.chunks:
            if chunk_start < end and chunk_end > start:
                for offset, (c, kind) in enumerate(chunk.iter_kind()):
                    if kind == "newline":
                        isspace = True
                        break
                    elif kind == "space":
                        pass
                    else:
                        isspace = False
                result.append(chunk)
                idx += 1
                
        if event_type == "start":
            if self._last_flush_kind() == "start":
                as_child = True
        elif event_type == "end":
            pass
        elif event_type == "chardata":
            if self._last_flush_kind() == "start":
                if isspace:
                    as_child = False
                    event_type = "start"
                else:
                    as_child = True
        # if not isspace and self._last_flush_kind() == "start":
            
                
        self.used_chunks.append((event_type, self.chunks[:idx]))
        self.chunks = self.chunks[idx:]        
        return result, as_child
            
        
    def _get_chunks_in_range3(self, start, end):
        """Return all chunks overlapping [start, end)"""
        result = []
        for chunk_start, chunk_end, chunk in self.chunks:
            if chunk_start < end and chunk_end > start:
                result.append(chunk)
        return result
    
    def _get_chunks_in_range2(self, start, end):
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
        # print(f"<flush_pending '{event_type}'>")
        # print("pending |",repr(event_data), "start_byte:", start_byte, "end_byte:", end_byte)
        chunks, as_child, start_offset, end_offset = self._get_chunks_in_range(event_type, start_byte, end_byte)
        # print("get chunks |", event_type, "chunks:", [(s,e,repr(c.content), c.id) for s,e,c in chunks], "start_offset:", start_offset, "end_offset:", end_offset)
        # print("</flush_pending>")
        chunks = [c for _, _, c in chunks]
        if len(chunks) == 0:
            return
        if event_type == 'start':
            name, attrs = event_data
            if name == self.root_tag:
                self.context.init_root()
                return
            block = self.context.instantiate(name, chunks, attrs=attrs, style=None)
            self._push_block(block)
        elif event_type == 'end':
            name = event_data
            if name == self.root_tag:
                return
            view = self.context.commit(chunks)
            self._push_block(view)            
        elif event_type == 'chardata':            
            # for chunk in chunks:
            cb = self.context.append(chunks, as_child=as_child, start_offset=start_offset, end_offset=end_offset)            
            self._push_block(cb)        
        # self.context._root._block.print_debug()
        # print("--------------------------------")
        self.pending = None
    
    
    def _on_start(self, name, attrs):        
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._tag_path.append(name)
        # print('start##', (repr(name), attrs), current_pos)
        self.pending = ('start', (name, attrs), current_pos)
    
    def _on_end(self, name):        
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        self._tag_path.pop()
        # print('end##', repr(name), current_pos)
        self.pending = ('end', name, current_pos)
    
    def _on_chardata(self, data):
        # print(f"chardata: '{data}'")
        
        current_pos = self.parser.CurrentByteIndex
        self._flush_pending(current_pos)
        # For chardata we could compute end directly, but for consistency
        # we'll use the deferred approach too
        # print('chardata##', repr(data), current_pos)
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
