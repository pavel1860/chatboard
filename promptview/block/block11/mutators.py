from __future__ import annotations
from typing import TYPE_CHECKING, Generator

from .span import Span, Chunk, split_chunks
from .block import Block, Mutator, ContentType

if TYPE_CHECKING:
    pass




class XmlMutator(Mutator):
    styles = ["xml"]
    
    
    @property
    def head(self) -> Span:
        return self.block.children[0].span
    
    @property
    def body(self) -> list[Block]:        
        return self.block.children[0].children
    
    
    @property
    def content(self) -> str:
        return self.block.children[0].span.content_text
    
    @property
    def block_end(self) -> Span:
        return self.block.children[1].span
    
    def render(self, block: Block) -> Block:
        with Block() as xml_blk:
            with xml_blk(block.content, tags=["opening-tag"]) as content:
                content.append_prefix("<")
                content.prepend_postfix(">")    
                for child in block.body:
                    content.append_child(child)
            with xml_blk(block.content, tags=["closing-tag"]) as postfix:
                postfix.append_prefix("</")
                postfix.prepend_postfix(">")
        return xml_blk
    
    
    def instantiate(self, content: ContentType | None = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None) -> Block:
        with Block(role=role, tags=tags) as block:
            with block(content) as head:
                pass
            # with block() as body:
            #     pass
        return block
    
    
    def init(self, chunks: list[Chunk], tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None) -> Block:
        prev_chunks, start_chunk, post = split_chunks(chunks, "<")
        content_chunks, end_chunk, post_chunks = split_chunks(post, ">")
        with Block() as xml_blk:
            with xml_blk(content_chunks, tags=["opening-tag"]) as content:
                content.append_prefix(prev_chunks + start_chunk)
                content.append_postfix(end_chunk + post_chunks)
        return xml_blk
    
    def commit(self, chunks: list[Chunk]) -> Block:
        prev_chunks, start_chunk, post = split_chunks(chunks, "<")
        content_chunks, end_chunk, post_chunks = split_chunks(post, ">")
        with self.block as blk:
            with blk(content_chunks, tags=["opening-tag"]) as end_tag:
                end_tag.append_prefix(prev_chunks + start_chunk)
                end_tag.append_postfix(end_chunk + post_chunks)
        return blk
    
    # def commit(self, content: ContentType) -> Block:
    #     self.block.append_child(content)
    #     return self.block
    
    
    # def on_text(self, block: Block, chunks: list[Chunk]) -> Block:
    #     pass
    
    # # def on_symbol(self, block: Block, chunks: list[Chunk]) -> Block:
    # #     pass
        
    # # def on_start(self, block: Block, chunks: list[Chunk]) -> Block:
    # #     pass
    # def on_chunks(self, block: Block, chunks: list[Chunk]) -> Block:
    #     pass
    
    # def on_end(self, block: Block, chunks: list[Chunk]) -> Block:
    #     pass
    
    
    # def parse(self, prefix: list[Chunk], content: list[Chunk], postfix: list[Chunk]) -> Generator[list[Chunk], Block, None]:
    #     # self.block[0]
    #     start_tag = False
    #     end_tag = False
    #     with Block(prefix=prefix, content=content, postfix=postfix) as block:
    #         chunks = yield block
    #         for chunk in chunks:
    #             block += chunk
    #             if not start_tag:                    
    #                 if chunk.is_line_end:
    #                     start_tag = True
    #             else:
    #                 if chunk.isspace():
                        
                    
            
            