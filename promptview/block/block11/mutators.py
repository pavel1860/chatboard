from __future__ import annotations
from typing import TYPE_CHECKING

from .span import Span, Chunk
from .block import Block, Mutator

if TYPE_CHECKING:
    pass




class XmlMutator(Mutator):
    styles = ["xml"]
    
    def render(self, block: Block) -> Block:
        # with Block() as blk:
        #     with blk(block.content, tags=["opening-tag"]) as content:
        #         content.prefix_prepend("<")
        #         content.postfix_append(">")
        #     with blk() as body:
        #         for child in block.children:
        #             body /= child
        #     with blk(block.content, tags=["closing-tag"]) as postfix:
        #         postfix.prefix_prepend("</")
        #         postfix.postfix_append(">")
        return block