from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Type, TYPE_CHECKING, overload
from uuid import uuid4
from abc import ABC

from .chunk import Chunk, BlockText
from .span import Span, SpanAnchor, VirtualBlockText

if TYPE_CHECKING:
    pass


# Type alias for chunk mapping during copy
ChunkMap = dict[str, Chunk]  # old_chunk_id -> new_chunk


def _generate_id() -> str:
    """Generate a short unique ID for blocks."""
    return uuid4().hex[:8]

def parse_style(style: str | list[str] | None) -> list[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []

ContentType = str | list[str] | list[Chunk] | Chunk


class BlockBase(ABC):
    """
    Common functionality for both BlockSchema and Block.

    Holds:
    - Style for rendering format (xml, markdown, plain)
    - Attributes (key-value pairs)
    """

    
    __slots__ = [
        "styles", 
        "tags", 
        "parent", 
        "_block_text", 
        "_span", 
        "children",
        "_postfix_span",
        "_prefix_span",
    ]
    
    def __init__(
        self, 
        content: ContentType | None = None, 
        children: list["BlockBase"] | None = None,
        style: str | None = None, 
        tags: list[str] = [],
        parent: "Block | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
    ):
        self.styles = parse_style(style) if style is not None else styles or []
        self.tags = tags
        self.parent = parent
        self._block_text = block_text or BlockText()
        content = self.promote_content(content)
        chunks = self._block_text.extend(content)
        # self._span = Span(start=SpanAnchor(chunk=chunks[0], offset=0), end=SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content)))
        self._span = Span.from_chunks(chunks)
        self.children = children or []
        self._postfix_span: Span | None = None
        self._prefix_span: Span | None = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    
    @property
    def end_chunk(self) -> Chunk:
        return self._span.end.chunk
    
    @property
    def start_chunk(self) -> Chunk:
        return self._span.start.chunk
    
    @property
    def end_postfix_chunk(self) -> Chunk | None:
        if self._postfix_span is None:
            return self.end_chunk
        return self._postfix_span.end.chunk
    
    @property
    def start_prefix_chunk(self) -> Chunk | None:
        if self._prefix_span is None:
            return self.start_chunk
        return self._prefix_span.start.chunk

    
    def promote_content(self, content: "ContentType | None") -> list[Chunk]:
        if content is None:
            return [Chunk(content="")]
        if isinstance(content, str):
            return [Chunk(content)]
        elif isinstance(content, list):
            return content
        elif isinstance(content, Chunk):
            return [content]
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        return content
    
    def promote_block_content(self, content: ContentType | "BlockBase" | None) -> "BlockBase":
        if isinstance(content, Block):
            return content
        elif content is None:
            return Block(parent=self, block_text=self._block_text)
        else:
            content = self.promote_content(content)
            return Block(content=content, block_text=self._block_text)    
    
    def append(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        if sep:
            content = [Chunk(content=sep)] + content
        chunks = self._block_text.extend(content, after=self.end_chunk)
        self._span.end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
        
    def prepend(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        if sep:
            content = content + [Chunk(content=sep)]
        chunks = self._block_text.left_extend(content, before=self.start_chunk)
        self._span.start = SpanAnchor(chunk=chunks[0], offset=0)
        
    def postfix_append(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        if sep:
            content = content + [Chunk(content=sep)]
        chunks = self._block_text.extend(content, after=self.end_postfix_chunk)
        self._postfix_span = Span.from_chunks(chunks)
        
    def prefix_prepend(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        if sep:
            content = [Chunk(content=sep)] + content
        chunks = self._block_text.left_extend(content)
        self._prefix_span = Span.from_chunks(chunks)
        
        
    def insert(self, index: int, content: ContentType):
        content = self.promote_content(content)
        self._block_text.insert(index, content)
        
    def remove(self, index: int):
        self._block_text.remove(index)
        
    
    def append_child(self, child_content: ContentType):
        block = self.promote_block_content(child_content)
        # if self.children:
        #     self.children[-1].postfix_append("\n")
        # else:
        #     self.postfix_append("\n")
        self.append_block_child(block)
        return block
        
    def append_block_child(self, block: "Block"):
        self.children.append(block)
        block.parent = self
        return block
        
        
    def render(self) -> str:
        from .block_transformers import transform
        block = self.copy()
        block = transform(block)
        return block._block_text.text()

    def copy(self) -> "BlockBase":
        """
        Create a deep copy of this block and its subtree.

        Copies the underlying BlockText and rebuilds spans to point to new chunks.
        """
        # Copy the BlockText (creates new chunks with same content)
        new_block_text = self._block_text.fork()

        # Build chunk mapping: old_id -> new_chunk
        chunk_map = {}
        old_chunks = list(self._block_text)
        new_chunks = list(new_block_text)
        for old, new in zip(old_chunks, new_chunks):
            chunk_map[old.id] = new

        # Copy block tree with remapped spans
        return self._copy_tree(chunk_map, new_block_text)

    def _copy_tree(self, chunk_map: dict, new_block_text: BlockText) -> "BlockBase":
        """Copy this block using chunk mapping."""
        new_block = Block(
            styles=list(self.styles),
            tags=list(self.tags),
            block_text=new_block_text,
        )

        # Remap spans
        new_block._span = Span(
            start=SpanAnchor(chunk_map[self._span.start.chunk.id], self._span.start.offset),
            end=SpanAnchor(chunk_map[self._span.end.chunk.id], self._span.end.offset),
        )

        if self._prefix_span:
            new_block._prefix_span = Span(
                start=SpanAnchor(chunk_map[self._prefix_span.start.chunk.id], self._prefix_span.start.offset),
                end=SpanAnchor(chunk_map[self._prefix_span.end.chunk.id], self._prefix_span.end.offset),
            )

        if self._postfix_span:
            new_block._postfix_span = Span(
                start=SpanAnchor(chunk_map[self._postfix_span.start.chunk.id], self._postfix_span.start.offset),
                end=SpanAnchor(chunk_map[self._postfix_span.end.chunk.id], self._postfix_span.end.offset),
            )

        # Copy children
        for child in self.children:
            new_block.append_block_child(child._copy_tree(chunk_map, new_block_text))

        return new_block
    

    def traverse(self) -> Iterator["BlockBase"]:
        yield self
        for child in self.children:
            yield from child.traverse()
    
    
    def print(self):
        print(self.render())
    
class Block(BlockBase):
    """
    Block is a tree node with structure and style.
    """
    
    
    def __init__(
        self, 
        content: ContentType | None = None, 
        style: str | None = None,
        tags: list[str] = [],
        parent: "Block | None" = None,
        styles: list[str] = [], 
        block_text: BlockText | None = None,
    ):
        super().__init__(content, style=style, tags=tags, parent=parent, block_text=block_text, styles=styles)

 
        
    def __call__(
        self, 
        content: ContentType | BlockBase | None = None, 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "Block":         
        block = self.promote_block_content(content)
        self.append_block_child(block)
        return block
