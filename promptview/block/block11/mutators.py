from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator, Iterator, Literal

from .span import BlockChunkList, Span, BlockChunk, chunks_contain, split_chunks
from .block import Block, Mutator, Stylizer, ContentType, BlockChildren
from .path import Path

if TYPE_CHECKING:
    pass



class BlockMutator(Mutator):
    styles = ["block"]
    
    # def append(self, span: Span, chunk: BlockChunk) -> Generator[BlockChunkList | Block, Any, Any]:
    #     yield span.append([chunk])
    #     if chunk.is_line_end:            
    #         yield self.append_child(Block())
    
    
    # def append(self, span: Span, chunk: BlockChunk) -> Generator[BlockChunkList | Span, Any, Any]:
    #     yield span.append([chunk])
    #     if chunk.is_line_end:            
    #         yield span.append_next()
    
    
    def append(self, chunk: BlockChunk) -> Generator[BlockChunkList | Block, Any, Any]:
        yield self.tail.span.append([chunk])
        if chunk.is_line_end:            
            yield from self.append_child(Block())
            
    def append_child(self, child: Block, add_newline: bool = True) -> Generator[BlockChunkList | Block, Any, Any]:
        if add_newline and not self.tail.span.has_newline():
            yield self.tail.span.append([BlockChunk(content="\n", style="block")])
        
    


class MarkdownMutator(BlockMutator):
    styles = ["md"]


    # def render(self, block: Block, path: Path) -> Block:
    #     block.prepend_prefix("#" * (path.depth + 1) + " ")
    #     return block
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        block = Block(chunks)
        block.prepend("#" * (path.depth + 1) + " ", style="md")
        return block
    


class MarkdownListStylizer(Stylizer):
    styles = ["list"]
    
    def append_child(self, child: Block) -> Generator[BlockChunkList | Block, Any, Any]:
        yield child.span.prepend([BlockChunk("* ")], style="list")
        
        



class XmlMutator(Mutator):
    styles = ["xml"]
    
    
    
    @property
    def content(self) -> str:    
        return self.block.children[0].span.content.text
    
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
    
    # def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Span:
    #     prefix, post = chunks.split_prefix("<")
    #     content, postfix = post.split_postfix(">")
    #     with Span(prefix + content.snake_case() + postfix) as content_span:
    #         for match in content.find_all(regex=r"([a-z])([A-Z])=([a-z])"):
    #             content_span(match[0], match[1], tags=["snake_case"])
    #     return 

    
    # def commit(self, span: Span, body: list[Span], chunks: BlockChunkList | None = None) -> Block | None:
    #     if chunks is None:
    #         prefix = []
    #         postfix = []
    #         content = self.head.span.content
    #     else:
    #         prefix, post = chunks.split_prefix("</")
    #         content, postfix = post.split_postfix(">")
    #     return Span(prefix + content.snake_case() + postfix)
    
    
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        prefix, post = chunks.split_prefix("<")
        content, postfix = post.split_postfix(">")

        with Block(attrs=attrs) as xml_blk:
            with xml_blk(content.snake_case(), tags=["opening-tag"]) as content:
                content.prepend(prefix or "<", style="xml")
                content.append(postfix or ">", style="xml")
                # with content() as body:
                #     pass
            
        return xml_blk
    
    def commit(self, chunks: BlockChunkList | None = None, add_newline: bool = True) -> Block | None:
        if chunks is None:
            prefix = []
            postfix = []
            content = self.block.content
        else:
            prefix, post = chunks.split_prefix("</")
            content, postfix = post.split_postfix(">")
        # if self.is_last_block_open(chunks):
        #     self.body[-1].add_newline()
        
        if add_newline and not self.tail.span.has_newline():
            self.tail.span.append([BlockChunk(content="\n", style="xml")])
        
        with Block(content, tags=["closing-tag"]) as end_tag:
            end_tag.prepend(prefix or "</", style="xml")
            end_tag.append(postfix or ">", style="xml")
        self.block.append_child(end_tag, to_body=False)
        
        # if self.block.subtree_size() > 1 and not self.tail.span.has_newline():
        #     self.tail.span.append([BlockChunk(content="\n", style="xml")])
        return end_tag
    
    
    def append(self, chunk: BlockChunk) -> Generator[BlockChunkList | Block, Any, Any]:
        if not chunk.is_line_end:
            if len(self.body) == 0:
                # child = self.add_empty_child()                
                # yield child
                yield Block()
                
                
            

    # def append_child(self, child: Block) -> Generator[BlockChunkList | Block, Any, Any]:
    #     subtree_size = self.block.subtree_size()
    #     if subtree_size == 0:
    #         if child.tree_size() > 1:
    #             yield self.head.span.append([BlockChunk(content="\n", style="xml")])
    #     elif subtree_size == 1:
    #         child.indent()
    #         self.body[-1].indent()
    #         if not self.head.span.has_newline():
    #             yield self.head.span.append([BlockChunk(content="\n", style="xml")])
    #             yield self.tail.span.append([BlockChunk(content="\n", style="xml")])
    #     elif subtree_size > 1 and not self.tail.span.has_newline():
    #         child.indent()
    #         yield self.tail.span.append([BlockChunk(content="\n", style="xml")])
    def append_child(self, child: Block, add_newline: bool = True) -> Generator[BlockChunkList | Block, Any, Any]:
        child.indent(style="xml")
        if len(self.body) == 0:
            if add_newline and not self.head.span.has_newline():        
                yield self.head.span.append([BlockChunk(content="\n", style="xml")])
        else:
            if add_newline and not self.tail.span.has_newline():
                yield self.tail.span.append([BlockChunk(content="\n", style="xml")])
        
        
    
    
    
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
    
    # @property
    # def block_end(self) -> Span:
    #     if len(self.block.children) == 1:
    #         # return self.body[-1].span
    #         if len(self.block.children[0].children) == 0:
    #             return self.block.children[0].span
    #         else:
    #             return self.block.children[0].children[-1].span
    #     return self.block.children[1].span
    
    
    # @property
    # def block_postfix(self) -> Span | None:
    #     if len(self.block.children) == 2:
    #         return self.block.children[1].span
    #     return None
    
    
    # def is_head_open(self, chunks: list[BlockChunk]) -> bool:
    #     if self.block.children[0].children:
    #         return False
    #     if chunks_contain(self.block.children[0].span.postfix, ">"):
    #         # if all(chunk.isspace() or chunk.is_line_end for chunk in chunks):
    #         #     return True
    #         if all(chunk.is_line_end for chunk in chunks):
    #             return True
    #         if all(chunk.isspace() for chunk in chunks):
    #             if self.block.children[0].has_newline():
    #                 return False                
    #             return True
            
    #         else:
    #             return False
    #     else:
    #         return True
    
    
    # def on_newline(self, chunk: BlockChunk):
    #     if self.did_commit:
    #         return self.block_end.append([chunk], style="newline")
    #     return super().on_newline(chunk)           
        
    def render_attrs(self, block: Block) -> str:
        attrs = ""
        for k, v in block.attrs.items():
            attrs += f"{k}=\"{v}\""
            
        if attrs:
            attrs = " " + attrs
        return attrs
    
    
    
    
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

    # @property
    # def head(self) -> Block:
    #     """The opening markdown fence (```xml)."""
    #     return self.block.children[0]

    # @property
    # def body(self) -> BlockChildren:
    #     """The actual content inside the wrapper."""
    #     return self.block.children[1].children
        
        # First child is the wrapper containing actual content
        # first_child = self.block.children[0]
        # if first_child.is_wrapper:
        #     return first_child.body        

    @property
    def content(self) -> str:
        """Content of the head span."""
        return self.block.children[0].span.content_text

    # @property
    # def tail(self) -> Block:
    #     """The closing fence span."""
        
    #     if len(self.block.children) == 1:
    #         return self.block.children[0]
    #     elif len(self.block.children) == 2:
    #         return self.block.children[1].mutator.tail
    #     else:
    #         return self.block.children[2]
        
    # def append_child(self, child: Block, add_newline: bool = True) -> Generator[BlockChunkList | Block, Any, Any]:
    #     self.state = "content"
    #     yield None
        
    
        
    def init(self, chunks: BlockChunkList, path: Path, attrs: dict[str, Any] | None = None) -> Block:
        with Block(tags=["root"], attrs=attrs) as root_blk:            
            with root_blk(tags=["root_prefix"]) as pre:                
                pass            
            # with root_blk(tags=["root_content"]) as content:                
            #     pass   
            # with root_blk(tags=["root_postfix"]) as post:
            #     pass          
        return root_blk
    
    
    def append(self, chunk: BlockChunk) -> Generator[BlockChunkList | Block, Any, Any]:
        if len(self.body) > 1 and not "root_postfix" in self.block.children[-1].tags:
            print("!!!!!!!!!!!!!!!")
            yield self.create_empty_child(tags=["root_postfix"])
            

    # @property
    # def block_postfix(self) -> Span | None:
    #     """The closing fence block's span."""
    #     if len(self.block.children) >= 2:
    #         return self.block.children[-1].span
    #     return None
    
    
    # def on_newline(self, chunk: BlockChunk):
    #     return self.on_text([chunk])
    
    # def on_text(self, chunks: list[BlockChunk]):
    #     if self.state == "prefix":
    #         return self.block.children[0].append(chunks)
    #     elif self.state == "content":
    #         return self.block.children[1].append(chunks)
    #     elif self.state == "postfix":
    #         if len(self.block.children) == 2:
    #             return self.block.mutator.append_child(Block(chunks), to_body=False)
    #         return self.block.children[2].append(chunks)
    
    # def on_text(self, chunks: list[BlockChunk]):
    #     if self.state == "prefix":
    #         return self.block.children[0].append(chunks)
    #     elif self.state == "content":            
    #         self.state = "postfix"
    #         return self.block.mutator.append_child(Block(chunks), to_body=False)            
    #     elif self.state == "postfix":            
    #         return self.block.children[2].append(chunks)
        
    # def on_child(self, child: Block):
    #     self.state = "content"
    #     return child
    
    def extract(self) -> Block:
        if len(self.block.children) <= 1:        
            return self.block.copy(deep=False)
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

    
    
    
