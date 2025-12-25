from __future__ import annotations
from typing import TYPE_CHECKING

from .span import Span, Chunk
from .block import Block, Mutator

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
                    body /= child
            with xml_blk(block.content, tags=["closing-tag"]) as postfix:
                postfix.prepend_prefix("</")
                postfix.append_postfix(">")
        return xml_blk