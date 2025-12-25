from __future__ import annotations
from typing import TYPE_CHECKING

from .span import Span, Chunk
from .block import Block, Mutator, ContentType

if TYPE_CHECKING:
    pass




class XmlMutator(Mutator):
    styles = ["xml"]
    
    
    def get_body(self) -> list[Block]:
        return self.block.children[1].children
    
    def get_content(self) -> str:
        return self.block.children[0].content
    
    def get_head(self) -> Span | None:
        return self.block.children[0].head
    
    def render(self, block: Block) -> Block:
        with Block() as xml_blk:
            with xml_blk(block.content, tags=["opening-tag"]) as content:
                content.prepend_prefix("<")
                content.append_postfix(">")    
            with xml_blk() as body:
                for child in block.body:
                    body.append_child(child)
            with xml_blk(block.content, tags=["closing-tag"]) as postfix:
                postfix.prepend_prefix("</")
                postfix.append_postfix(">")
        return xml_blk
    
    
    def instantiate(self, content: ContentType | None = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None) -> Block:
        with Block(role=role, tags=tags, style=style) as block:
            with block(content) as head:
                pass
            with block() as body:
                pass
        return block
    
    
    
    def commit(self, content: ContentType) -> Block:
        self.block.append_child(content)
        return self.block