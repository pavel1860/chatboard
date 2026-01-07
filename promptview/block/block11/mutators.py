from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator, Iterator, Literal

from .span import BlockChunkList, Span, BlockChunk, chunks_contain, split_chunks
from .block import Block, Mutator, ContentType, BlockChildren
from .path import Path

if TYPE_CHECKING:
    pass




class XmlMutator(Mutator):
    styles = ["xml"]
    
    
    @property
    def head(self) -> Block:
        return self.block.children[0]
    
    @property
    def body(self) -> BlockChildren:        
        return self.block.children[0].children
    
    @property
    def tail(self) -> Block:
        if self.did_commit:
            return self.block.children[1]
        return super().tail
    
    @property
    def content(self) -> str:    
        return self.block.children[0].span.content.text
    
    
    # def current_span(self) -> Span:
    #     if self.did_commit:
    #         return self.block.children[1].span
    #     return super().current_span()
    
    
    def iter_delimiters(self) -> Iterator[Block]:
        yield self.block.children[0].span
        length = len(self.body)
        for i in range(length):
            yield self.body[i].mutator.tail
        # if self.did_commit:
        #     yield self.block.children[1]


    def join(self, sep: BlockChunk):
        for span in self.iter_delimiters():            
            span.append([sep])
    
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
    
    
    def on_newline(self, chunk: BlockChunk):
        if self.did_commit:
            return self.block_end.append([chunk], style="newline")
        return super().on_newline(chunk)           
        
    def render_attrs(self, block: Block) -> str:
        attrs = ""
        for k, v in block.attrs.items():
            attrs += f"{k}=\"{v}\""
            
        if attrs:
            attrs = " " + attrs
        return attrs
    
    
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        prefix, post = chunks.split_prefix("<")
        content, postfix = post.split_postfix(">")

        with Block(attrs=attrs) as xml_blk:
            with xml_blk(content.snake_case(), tags=["opening-tag"]) as content:
                content.prepend(prefix or "<", style="xml")
                content.append(postfix or ">", style="xml")
                with content() as body:
                    pass
            
        return xml_blk
    
    def commit(self, chunks: BlockChunkList | None = None) -> Block | None:
        if chunks is None:
            prefix = []
            postfix = []
            content = self.block.content
        else:
            prefix, post = chunks.split_prefix("</")
            content, postfix = post.split_postfix(">")
        # if self.is_last_block_open(chunks):
        #     self.body[-1].add_newline()
        
        with Block(content, tags=["closing-tag"]) as end_tag:
            end_tag.prepend(prefix or "</", style="xml")
            end_tag.append(postfix or ">", style="xml")
        self.block.mutator.append_child(end_tag, to_body=False)
                
        return end_tag
    
    
    def on_child(self, child: Block) -> Block:
        if "closing-tag" in child.tags:
            return child
        return child.indent()
    
    
    



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
    state: Literal["prefix", "content", "postfix"] = "prefix"

    @property
    def head(self) -> Block:
        """The opening markdown fence (```xml)."""
        return self.block.children[0]

    @property
    def body(self) -> BlockChildren:
        """The actual content inside the wrapper."""
        return self.block.children[1].children
        
        # First child is the wrapper containing actual content
        # first_child = self.block.children[0]
        # if first_child.is_wrapper:
        #     return first_child.body        

    @property
    def content(self) -> str:
        """Content of the head span."""
        return self.block.children[0].span.content_text

    @property
    def tail(self) -> Block:
        """The closing fence span."""
        
        if len(self.block.children) == 1:
            return self.block.children[0]
        elif len(self.block.children) == 2:
            return self.block.children[1].mutator.tail
        else:
            return self.block.children[2]
        # if not len(self.block.children[1]):
        #     return self.block.children[0]
        # elif not len(self.block.children[2]):
        #     return self.block.children[1].mutator.tail
        # else:
        #     return self.block.children[2]
            

    # @property
    # def block_postfix(self) -> Span | None:
    #     """The closing fence block's span."""
    #     if len(self.block.children) >= 2:
    #         return self.block.children[-1].span
    #     return None
    
    
    def on_newline(self, chunk: BlockChunk):
        return self.on_text([chunk])
    
    # def on_text(self, chunks: list[BlockChunk]):
    #     if self.state == "prefix":
    #         return self.block.children[0].append(chunks)
    #     elif self.state == "content":
    #         return self.block.children[1].append(chunks)
    #     elif self.state == "postfix":
    #         if len(self.block.children) == 2:
    #             return self.block.mutator.append_child(Block(chunks), to_body=False)
    #         return self.block.children[2].append(chunks)
    
    def on_text(self, chunks: list[BlockChunk]):
        if self.state == "prefix":
            return self.block.children[0].append(chunks)
        elif self.state == "content":            
            self.state = "postfix"
            return self.block.mutator.append_child(Block(chunks), to_body=False)            
        elif self.state == "postfix":            
            return self.block.children[2].append(chunks)
        
    def on_child(self, child: Block):
        self.state = "content"
        return child
    
    def extract(self) -> Block:
        return self.block.children[1].copy_head()
    
    
    # def render(self, block: Block, path: Path) -> Block:
    #     with Block(style="root") as root:
    #         with root("prefix", style="prefix") as prefix:
    #             pass
    #         with root("content", style="content") as content:
    #             content /= block
    #         with root("postfix", style="postfix") as postfix:
    #             pass
    #     return root
    
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        with Block("root", tags=["root"], attrs=attrs) as root_blk:            
            with root_blk(tags=["root_prefix"]) as pre:                
                pass            
            with root_blk(tags=["root_content"]) as content:                
                pass   
            # with root_blk(tags=["root_postfix"]) as post:
            #     pass          
        return root_blk
    


class MarkdownMutator(Mutator):
    styles = ["markdown", "md"]


    # def render(self, block: Block, path: Path) -> Block:
    #     block.prepend_prefix("#" * (path.depth + 1) + " ")
    #     return block
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        block = Block(chunks)
        block.prepend("#" * (path.depth + 1) + " ", style="md")
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

    
    