from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator

from .span import Span, BlockChunk, chunks_contain, split_chunks
from .block import Block, Mutator, ContentType
from .path import Path

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
        if len(self.block.children) == 1:
            # return self.body[-1].span
            if len(self.block.children[0].children) == 0:
                return self.block.children[0].span
            else:
                return self.block.children[0].children[-1].span
        return self.block.children[1].span
    
    
    @property
    def block_postfix(self) -> Span | None:
        if len(self.block.children) == 2:
            return self.block.children[1].span
        return None
    
    
    def is_head_open(self, chunks: list[BlockChunk]) -> bool:
        if self.block.children[0].children:
            return False
        if chunks_contain(self.block.children[0].span.postfix, ">"):
            # if all(chunk.isspace() or chunk.is_line_end for chunk in chunks):
            #     return True
            if all(chunk.is_line_end for chunk in chunks):
                return True
            if all(chunk.isspace() for chunk in chunks):
                if self.block.children[0].has_newline():
                    return False                
                return True
            
            else:
                return False
        else:
            return True
                
        
    def render_attrs(self, block: Block) -> str:
        attrs = ""
        for k, v in block.attrs.items():
            attrs += f"{k}=\"{v}\""
            
        if attrs:
            attrs = " " + attrs
        return attrs
    
    def render(self, block: Block, path: Path) -> Block:
        with Block() as xml_blk:
            tag_name =block.content.lower().replace(" ", "_")
            with xml_blk(tag_name + self.render_attrs(block), tags=["opening-tag"]) as content:
                content.append_prefix("<")
                content.prepend_postfix(">")    
                for child in block.body:
                    content.append_child(child)
                content.indent_body()
            with xml_blk(tag_name, tags=["closing-tag"]) as postfix:
                postfix.append_prefix("</")
                postfix.prepend_postfix(">")
        return xml_blk
    
    # def extract(self) -> Block:
        # return Block(self.head)
        
    
    def instantiate(self, content: ContentType | None = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None, attrs: dict[str, Any] | None = None) -> Block:
        with Block(role=role, tags=tags, attrs=attrs) as block:
            with block(content) as head:
                pass
            # with block() as body:
            #     pass
        return block
    
    
    def init(self, chunks: list[BlockChunk], tags: list[str] | None = None, role: str | None = None, style: str | list[str] | None = None, attrs: dict[str, Any] | None = None, _auto_handle: bool = True) -> Block:
        prev_chunks, start_chunk, post = split_chunks(chunks, "<")
        content_chunks, end_chunk, post_chunks = split_chunks(post, ">")
        with Block(tags=tags, role=role, attrs=attrs, _auto_handle=_auto_handle) as xml_blk:
            with xml_blk(content_chunks, tags=["opening-tag"]) as content:
                content.append_prefix(prev_chunks + start_chunk)
                content.append_postfix(end_chunk + post_chunks)
        return xml_blk
    
    def commit(self, chunks: list[BlockChunk]) -> Block:
        prev_chunks, start_chunk, post = split_chunks(chunks, "</")
        content_chunks, end_chunk, post_chunks = split_chunks(post, ">")
        if self.is_last_block_open(chunks):
            self.body[-1].add_newline()
        
        with Block(content_chunks, tags=["closing-tag"]) as end_tag:
            end_tag.append_prefix(prev_chunks + start_chunk)
            end_tag.append_postfix(end_chunk + post_chunks)
        self.block.mutator.append_child(end_tag, to_body=False)
        return end_tag
    
    



class RootMutator(Mutator):
    """
    Mutator for parser root blocks.

    Structures markdown code fence blocks similar to XmlMutator:
    - head: The opening fence (```xml)
    - body: The actual XML content inside the wrapper
    - block_postfix: The closing fence (```)

    Structure:
        RootBlock(span='```xml', children=[wrapper, closing_fence])
          wrapper (is_wrapper=True, children=[xml_content...])
          closing_fence (span='```')
    """
    styles = ["root"]

    @property
    def head(self) -> Span:
        """The opening markdown fence (```xml)."""
        return self.block.span

    @property
    def body(self) -> list[Block]:
        """The actual content inside the wrapper."""
        
        return self.block.children
        
        # First child is the wrapper containing actual content
        # first_child = self.block.children[0]
        # if first_child.is_wrapper:
        #     return first_child.body        

    @property
    def content(self) -> str:
        """Content of the head span."""
        return self.block.span.content_text

    @property
    def block_end(self) -> Span:
        """The closing fence span."""
        if len(self.block.children) >= 2:
            return self.block.children[-1].span
        if self.block.children:
            return self.block.children[-1].mutator.block_end
        return self.block.span

    @property
    def block_postfix(self) -> Span | None:
        """The closing fence block's span."""
        if len(self.block.children) >= 2:
            return self.block.children[-1].span
        return None
    
    
    # def instantiate(self, content: ContentType | None = None, role: str | None = None, tags: list[str] | None = None, style: str | None = None, attrs: dict[str, Any] | None = None) -> Block:
    #     with Block(content, role=role, tags=tags, attrs=attrs) as block:
    #         with block(content) as head:
    #             pass
    #         # with block() as body:
    #         #     pass
    #     return block



class MarkdownMutator(Mutator):
    styles = ["markdown", "md"]


    def render(self, block: Block, path: Path) -> Block:
        block.prepend_prefix("#" * (path.depth + 1) + " ")
        return block
        
                

   

class ToolDescriptionMutator(Mutator):
    styles = ["tool-desc"]
    
    def render(self, block: Block, path: Path) -> Block:

        description = block.get_one("description")
        parameters = block.get_one("parameters")

        # Build new output
        with Block("# Name: " + block.attrs.get("name", "")) as blk:
            with blk("## Purpose") as purpose:
                purpose /= description.body[0].content
            with blk("## Parameters") as params:
                for param in parameters.children:
                    with params(param.content) as param_blk:
                        param_blk /= param.body
                        if hasattr(param, 'type_str') and param.type_str is not None:
                            param_blk /= "Type:", param.type_str
                        if hasattr(param, 'is_required'):
                            param_blk /= "Required:", param.is_required                        
        return blk

    
    