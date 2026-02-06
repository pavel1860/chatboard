from dataclasses import dataclass
import xml.sax
from typing import TYPE_CHECKING, Any, AsyncGenerator
from queue import SimpleQueue


from ..prompt.events import StreamEvent
from ..block import BlockChunk, Block
# if TYPE_CHECKING:
    


class BlockBuffer:
    
    def __init__(self):
        self.buffer = []
        self.queue = SimpleQueue()
        
    def add(self, block: "BlockChunk"):
        # print("adding block:", f"'{block.content}'")
        self.buffer.append(block)
        
    def get(self):
        return self.queue.get()
    
    
    def empty(self):
        return self.queue.empty()
        
        
    def flush(self, tag: str, depth: int, name: str, attrs: dict[str, Any] | None = None):
        for block in self.buffer:
            # print("flushing block:", tag, name, block.content)
            self.queue.put((tag, depth, name, attrs, block))
        self.buffer = []

    def to_str(self):
        return ", ".join([f"[{block.content}]" for block in self.buffer])

class SaxStreamParser(xml.sax.ContentHandler):
            
    start_tag: str = 'tag_start'
    end_tag: str = 'tag_end'
    text_tag: str = 'chunk'
    
    def __init__(
        self, 
        block_buffer: BlockBuffer,
        start_tag: str | None = None, 
        end_tag: str | None = None, 
        text_tag: str | None = None,
    ):
        self.start_tag = start_tag or self.start_tag
        self.end_tag = end_tag or self.end_tag
        self.text_tag = text_tag or self.text_tag
        self.block_buffer = block_buffer
        self.buffer = []
        self.depth = 0
        self.current_tag = None

    def startElement(self, name, attrs):
        print("parser start##", name, self.buffer, "buffer:", self.block_buffer.to_str())
        text = self.flush_buffer()
        self.block_buffer.flush(self.start_tag, self.depth, name, dict(attrs))
        # self.queue.put(StreamEvent(type=self.start_tag, name=name, attrs=dict(attrs), depth=self.depth, payload=text))
        self.current_tag = name
        self.buffer = []
        self.depth += 1

    def characters(self, content):
        print("parser text##", f"'{content}' =>", self.buffer, "buffer:", self.block_buffer.to_str())
        self.block_buffer.flush(self.text_tag, self.depth, self.current_tag)
        self.buffer.append(content)
        # self.queue.put(StreamEvent(type=self.text_tag, name = self.current_tag, payload=content, depth=self.depth))        

    def endElement(self, name):
        print("parser end##", name, self.buffer, "buffer:", self.block_buffer.to_str())
        text = self.flush_buffer()
        self.block_buffer.flush(self.end_tag, self.depth, name)
        self.depth -= 1
        # self.queue.put(StreamEvent(type=self.end_tag, name=name, payload=text, depth=self.depth))

    def flush_buffer(self):
        # print("flushing buffer", self.buffer)
        text = ''.join(self.buffer).strip()
        # print("flushing buffer:", f"'{text}'")
        self.buffer = []
        if text:
            return text
        # self.buffer = []
        












































class SaxBlockParser(xml.sax.ContentHandler):
    def __init__(self, output_format: str):        
        self.queue = SimpleQueue()
        self.start_tag = "tag_start"
        self.end_tag = "tag_end"
        self.text_tag = "chunk"
        self.parser = xml.sax.make_parser()
        self.parser.setFeature(xml.sax.handler.feature_external_ges, False)        
        self.parser.setContentHandler(self)
        self._schema = output_format
        
    
    
    def startElement(self, name, attrs):
        print("parser>>", name, attrs)
        text = self.flush_buffer()
        self.queue.put(StreamEvent(type=self.start_tag, name=name, attrs=dict(attrs), depth=self.depth, payload=text))
        self.current_tag = name
        self.buffer = []
        self.depth += 1

    def characters(self, content):
        self.buffer.append(content)
        self.queue.put(StreamEvent(type=self.text_tag, name = self.current_tag, payload=content, depth=self.depth))

    def endElement(self, name):
        text = self.flush_buffer()
        self.depth -= 1
        self.queue.put(StreamEvent(type=self.end_tag, name=name, payload=text, depth=self.depth))

    def flush_buffer(self):
        text = ''.join(self.buffer).strip()
        if text:
            return text
        self.buffer = []
        

        
        
        
        
async def stream_parser(
    stream: AsyncGenerator[str, None], 
    start_tag: str | None = None, 
    end_tag: str | None = None, 
    text_tag: str | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
    queue = SimpleQueue()
    handler = SaxStreamParser(queue, start_tag, end_tag, text_tag)

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_external_ges, False)
    parser.setContentHandler(handler)
    
    async for chunk in stream:
        parser.feed(chunk)
        while not queue.empty():
            yield queue.get()
    parser.close()





class StreamParser:
    def __init__(
        self,
        stream: AsyncGenerator[str, None],
        start_tag: str | None = None,
        end_tag: str | None = None,
        text_tag: str | None = None
    ):
        self.stream = stream
        self.queue = SimpleQueue()
        self.handler = SaxStreamParser(self.queue, start_tag, end_tag, text_tag)

        self.parser = xml.sax.make_parser()
        self.parser.setFeature(xml.sax.handler.feature_external_ges, False)
        self.parser.setContentHandler(self.handler)
        self._chunk_iter = self._generator()

    def __aiter__(self):
        self._chunk_iter = self._generator()
        return self
    

    async def __anext__(self):
        return await self._chunk_iter.__anext__()
    
    def preprocess(self, chunk: "BlockChunk"):
        return chunk.content
    
    def postprocess(self, event: StreamEvent):
        return 

    async def _generator(self):
        async for chunk in self.stream:
            self.parser.feed(chunk)
            while not self.queue.empty():
                yield self.queue.get()
        self.parser.close()