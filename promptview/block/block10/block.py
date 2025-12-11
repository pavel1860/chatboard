from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Type, TYPE_CHECKING, overload
from uuid import uuid4
from abc import ABC

from .chunk import Chunk, BlockText
from .span import Span, SpanAnchor, VirtualBlockText
from .path import Path

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
        "role",
    ]
    
    def __init__(
        self, 
        content: ContentType | None = None, 
        children: list["BlockBase"] | None = None,
        role: str | None = None,
        style: str | None = None, 
        tags: list[str] | None = None,
        parent: "BlockBase | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
    ):
        self.styles = parse_style(style) if style is not None else styles or []
        self.role = role
        self.tags = tags or []
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

    def __len__(self) -> int:
        """Return number of children."""
        return len(self.children)

    def __bool__(self) -> bool:
        """Block is truthy if it has children."""
        return len(self.children) > 0
    
    
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

    # -------------------------------------------------------------------------
    # Boundary properties (for entire subtree)
    # -------------------------------------------------------------------------

    @property
    def boundary_start(self) -> Chunk | None:
        """
        Get the first chunk of this block's entire subtree.

        Returns the earliest chunk considering:
        - prefix span (if present)
        - content span
        """
        return self.start_prefix_chunk

    @property
    def boundary_end(self) -> Chunk | None:
        """
        Get the last chunk of this block's entire subtree.

        Returns the latest chunk considering:
        - postfix span of deepest last child
        - or this block's postfix/content span if no children
        """
        if self.children:
            # Recursively get boundary_end of the last child
            return self.children[-1].boundary_end
        else:
            # No children - return this block's end
            return self.end_postfix_chunk

    def get_boundaries(self) -> tuple[Chunk | None, Chunk | None]:
        """
        Get the start and end chunks of this block's entire subtree.

        Returns:
            Tuple of (start_chunk, end_chunk) covering the entire block tree
            including prefix, content, children, and postfix.

        Example:
            start, end = block.get_boundaries()
            # Fork just this block's chunks
            new_text = block._block_text.fork(start=start, end=end)
        """
        return (self.boundary_start, self.boundary_end)

    # -------------------------------------------------------------------------
    # Path properties
    # -------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """
        Compute current path dynamically.

        Returns a Path object with both index-based and tag-based paths.
        """
        indices = []
        tags = []

        # Walk up from this block to root, collecting indices and tags
        current = self
        while current.parent is not None:
            # Get index of current in parent's children
            idx = current.parent.children.index(current)
            indices.append(idx)

            # Collect first tag if present
            if current.tags:
                tags.append(current.tags[0])

            current = current.parent

        # Reverse since we collected from leaf to root
        indices.reverse()
        tags.reverse()

        return Path(indices, tags)

    @property
    def tag_path(self) -> list[str]:
        """Get tag-based path as a list of strings."""
        return list(self.path.tags)

    @property
    def depth(self) -> int:
        """Get depth in tree (0 for root)."""
        return self.path.depth

    @property
    def index(self) -> int | None:
        """Get index in parent's children list."""
        if self.parent is None:
            return None
        return self.parent.children.index(self)

    def path_get(self, path: str | list[int] | Path) -> "BlockBase | None":
        """
        Get block at the given path relative to this block.

        Args:
            path: Path as string "0.2.1", list [0, 2, 1], or Path object

        Returns:
            Block at path, or None if not found
        """
        if isinstance(path, str):
            path = Path.from_string(path)
        elif isinstance(path, list):
            path = Path(path)

        target = self
        for idx in path.indices:
            if idx >= len(target.children):
                return None
            target = target.children[idx]

        return target

    def path_exists(self, path: str | list[int] | Path) -> bool:
        """Check if a path exists relative to this block."""
        return self.path_get(path) is not None

    # -------------------------------------------------------------------------
    # Tag-based search methods
    # -------------------------------------------------------------------------

    def traverse(self) -> Iterator["BlockBase"]:
        """
        Iterate over this block and all descendants (pre-order depth-first).

        Yields:
            This block, then recursively all descendants
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    def get_all(self, tags: str | list[str]) -> list["BlockBase"]:
        """
        Get all blocks matching a tag path.

        Supports dot-notation for nested tag searches:
        - "response" - find all blocks with tag "response"
        - "response.thinking" - find "thinking" blocks that are descendants of "response"

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            List of matching blocks
        """
        if isinstance(tags, str):
            tags = tags.split(".")

        if not tags:
            return []

        # Find all blocks matching first tag
        candidates = [b for b in self.traverse() if tags[0] in b.tags]

        # Filter through remaining tags
        for tag in tags[1:]:
            next_candidates = []
            for blk in candidates:
                for child in blk.traverse():
                    if child is not blk and tag in child.tags:
                        next_candidates.append(child)
            candidates = next_candidates

        return candidates

    def get_one(self, tags: str | list[str]) -> "BlockBase":
        """
        Get the first block matching a tag path.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block

        Raises:
            ValueError: If no matching block found
        """
        result = self.get_all(tags)
        if not result:
            raise ValueError(f'Tag path "{tags}" does not exist')
        return result[0]

    def get_one_or_none(self, tags: str | list[str]) -> "BlockBase | None":
        """
        Get the first block matching a tag path, or None if not found.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block, or None
        """
        result = self.get_all(tags)
        return result[0] if result else None

    def get(self, tag: str) -> "BlockBase | None":
        """
        Get the first direct or nested child with the given tag.

        Simple recursive search - does not support dot-notation paths.

        Args:
            tag: Tag to search for

        Returns:
            First matching block, or None
        """
        if tag in self.tags:
            return self
        for child in self.children:
            if tag in child.tags:
                return child
            if (block := child.get(tag)) is not None:
                return block
        return None

    def get_last(self, tag: str) -> "BlockBase | None":
        """
        Get the last block with the given tag.

        Args:
            tag: Tag to search for

        Returns:
            Last matching block, or None
        """
        result = None
        for blk in self.traverse():
            if tag in blk.tags:
                result = blk
        return result


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
    
    def promote_block_content(self, content: ContentType | "BlockBase" | None, style: str | None = None, tags: list[str] | None = None, role: str | None = None) -> "BlockBase":
        if isinstance(content, Block):
            return content
        elif content is None:
            return Block(parent=self, block_text=self._block_text, style=style, tags=tags, role=role)
        else:
            content = self.promote_content(content)
            return Block(content=content, block_text=self._block_text, style=style, tags=tags, role=role)    
        
    def _append_separator(self, content: list[Chunk], sep: str | None, append: bool = True):
        if sep:
            if not content[-1].is_line_end:
                if append:
                    content = [Chunk(content=sep)] + content
                else:
                    content = content + [Chunk(content=sep)]
        return content
    
    def append(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=True)
        chunks = self._block_text.extend(content, after=self.end_chunk)
        self._span.end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
        
    def prepend(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=False)
        chunks = self._block_text.left_extend(content, before=self.start_chunk)
        self._span.start = SpanAnchor(chunk=chunks[0], offset=0)
        
    def postfix_append(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=True)
        chunks = self._block_text.extend(content, after=self.end_postfix_chunk)
        self._postfix_span = Span.from_chunks(chunks)
        
    def prefix_prepend(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=False)
        chunks = self._block_text.left_extend(content, before=self.start_prefix_chunk)
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
        start, end = self.get_boundaries()
        new_block_text = self._block_text.fork(start, end)

        # Build chunk mapping: old_id -> new_chunk
        # Only iterate over chunks within our boundaries
        chunk_map = {}
        current = start
        new_chunk_iter = iter(new_block_text)
        while current is not None:
            chunk_map[current.id] = next(new_chunk_iter)
            if current is end:
                break
            current = current.next

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
        
        
    def __itruediv__(self, other: ContentType):
        self.append_child(other)
        return self
    
    def __add__(self, other: ContentType):
        self.append(other)
        return self
    
    def __radd__(self, other: ContentType):
        self.prepend(other)
        return self
    
    def __iadd__(self, other: ContentType):
        self.append(other)
        return self
    
    def __and__(self, other: ContentType):
        self.append(other, sep="")
        return self
    
    def __rand__(self, other: ContentType):
        self.prepend(other, sep="")
        return self
    
    def __iand__(self, other: ContentType):
        self.append(other, sep="")
        return self
    
    # def __isub__(self, other: ContentType):
    #     self.
        
        
    
    
class Block(BlockBase):
    """
    Block is a tree node with structure and style.
    """
    
    
    def __init__(
        self, 
        content: ContentType | None = None, 
        children: list["BlockBase"] | None = None,
        role: str | None = None,
        style: str | None = None,
        tags: list[str] | None = None,
        parent: "Block | None" = None,
        styles: list[str] | None = None, 
        block_text: BlockText | None = None,        
    ):
        super().__init__(content, children=children, role=role, style=style, tags=tags, parent=parent, block_text=block_text, styles=styles)

 
        
    def __call__(
        self, 
        content: ContentType | BlockBase | None = None, 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "Block":         
        block = self.promote_block_content(content, style=style, tags=tags, role=role)
        self.append_block_child(block)
        return block
