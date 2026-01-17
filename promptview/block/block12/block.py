"""
Block - Unified tree node with local text and chunk metadata.

Block12 simplifies the block system by:
1. Each block owns its own text string (local, not shared)
2. Using the block tree for structure
3. Chunk positions are relative to the block's local text
4. Delegating style-aware operations to Mutators

Usage:
    with Block("Header") as b:
        b /= "Hello"
        b /= "World"
"""

from __future__ import annotations
from typing import Any, Iterator, TYPE_CHECKING, Self, Type, Union, SupportsIndex, overload
from collections import UserList
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
import re

from promptview.utils.function_utils import is_overridden

from .chunk import ChunkMeta, BlockChunk

if TYPE_CHECKING:
    from .mutator import Mutator
    from .schema import BlockSchema, BlockListSchema, BlockList
    from .path import IndexPath, TagPath


def _generate_id() -> str:
    """Generate a short unique ID."""
    from uuid import uuid4
    return uuid4().hex[:8]


# Type for content that can be passed to Block
ContentType = Union[str, int, float, bool,  "Block", list[BlockChunk], None]


# =========================================================================
# Chunk Helper Functions
# =========================================================================

def chunks_contain(chunks: list[BlockChunk], s: str) -> bool:
    """
    Check if a string is present in the chunks (may span multiple chunks).

    Args:
        chunks: List of Chunk objects
        s: String to search for

    Returns:
        True if string is found, False otherwise
    """
    if not chunks or not s:
        return False
    full_text = "".join(c.content for c in chunks)
    return s in full_text


def split_chunks(
    chunks: list[BlockChunk],
    sep: str
) -> tuple[list[BlockChunk], list[BlockChunk], list[BlockChunk]]:
    """
    Split chunks on a separator that may span multiple chunks.

    Args:
        chunks: List of Chunk objects
        sep: Separator string to split on

    Returns: (before, separator, after)
        - before: Chunk objects before the separator
        - separator: Chunk objects that make up the separator (empty if not found)
        - after: Chunk objects after the separator

    Note: If separator falls mid-chunk, that chunk is split using Chunk.split()
    """
    if not chunks or not sep:
        return list(chunks), [], []

    # Build full text
    full_text = "".join(c.content for c in chunks)

    # Find separator
    sep_idx = full_text.find(sep)
    if sep_idx == -1:
        return list(chunks), [], []

    sep_end_idx = sep_idx + len(sep)

    # Track position as we iterate
    pos = 0
    before: list[BlockChunk] = []
    separator: list[BlockChunk] = []
    after: list[BlockChunk] = []

    for chunk in chunks:
        chunk_start = pos
        chunk_end = pos + len(chunk.content)

        if chunk_end <= sep_idx:
            # Whole chunk is before separator
            before.append(chunk)
        elif chunk_start >= sep_end_idx:
            # Whole chunk is after separator
            after.append(chunk)
        elif chunk_start >= sep_idx and chunk_end <= sep_end_idx:
            # Whole chunk is within separator
            separator.append(chunk)
        else:
            # Chunk overlaps with separator boundary
            if chunk_start < sep_idx:
                # Chunk starts before separator
                split_offset = sep_idx - chunk_start
                left_chunk, right_chunk = chunk.split(split_offset)
                if left_chunk.content:
                    before.append(left_chunk)

                if chunk_end <= sep_end_idx:
                    # Rest of chunk is part of separator
                    if right_chunk.content:
                        separator.append(right_chunk)
                else:
                    # Separator ends within this chunk too
                    sep_part_len = sep_end_idx - sep_idx
                    sep_chunk, after_chunk = right_chunk.split(sep_part_len)
                    if sep_chunk.content:
                        separator.append(sep_chunk)
                    if after_chunk.content:
                        after.append(after_chunk)
            else:
                # Chunk starts within separator but extends past it
                sep_part_len = sep_end_idx - chunk_start
                sep_chunk, after_chunk = chunk.split(sep_part_len)
                if sep_chunk.content:
                    separator.append(sep_chunk)
                if after_chunk.content:
                    after.append(after_chunk)

        pos = chunk_end

    return before, separator, after


class BlockChildren(UserList["Block"]):

    def __init__(self, parent: Block, items: list[Block] | None = None):
        self.parent: Block = parent
        UserList.__init__(self, items)

    def __iter__(self) -> Iterator["Block"]:
        """Iterate over children without relying on IndexError."""
        return iter(self.data)

    @overload
    def __getitem__(self, index: SupportsIndex) -> 'Block': ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: SupportsIndex | slice) -> 'Block | Self':
        result = self.data[index]
        if isinstance(index, slice):
            return self.__class__(self.parent, result)
        return result



class Block:
    """
    Tree node with local text and chunk metadata.

    Each block owns its own text string. When rendering, text is
    concatenated depth-first through the tree.

    Chunk metadata uses positions relative to the block's local text.

    Public operations (append, append_child) delegate to the mutator
    for style-aware behavior. Raw operations (_raw_*) bypass the mutator.

    Usage:
        with Block("Header") as b:
            b /= "First child"
            b /= "Second child"

    Attributes:
        parent: Parent block (None for root)
        children: Child blocks
        chunks: Chunk metadata (positions relative to local _text)
        role: Role identifier (e.g., "user", "assistant")
        tags: Tag list for querying
        style: Style list for rendering/mutator selection
        attrs: Arbitrary attributes
        _text: This block's local text content
    """

    __slots__ = [
        "parent", "children", "chunks",
        "_role", "tags", "style", "attrs", "_text", "id", "mutator", "stylizers"
    ]

    def __init__(
        self,
        content: ContentType = None,
        *,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """
        Create a block with optional initial content.

        Args:
            content: Initial content (str, int, float, bool, Block)
            role: Role identifier for the block
            tags: List of tags for categorization
            style: Style string or list of styles
            attrs: Arbitrary attributes
        """
        from .mutator import Mutator, Stylizer
        # Tree structure
        self.parent: Block | None = None
        self.children: BlockChildren = BlockChildren(parent=self)

        # Chunk metadata (relative to local _text)
        self.chunks: list[ChunkMeta] = []

        # Block metadata
        self._role = role
        self.tags = tags or []
        self.style = _parse_style(style)
        self.attrs = attrs or {}

        # Local text for this block
        self._text: str = ""

        # ID
        self.id: str = _generate_id()

        # Mutator (lazy initialized)
        self.mutator: Mutator = Mutator(self)
        self.stylizers: list[Stylizer] = []
        # Handle initial content
        if content is not None:
            if isinstance(content, Block):
                # Copy content from another block
                content = content.get_chunks()
                # self._raw_append(content.text)
            if isinstance(content, (list, tuple)) and all(isinstance(item, BlockChunk) for item in content):
                for chunk in content:
                    self._raw_append(chunk.content, logprob=chunk.logprob, style=chunk.style)
            elif isinstance(content, (str, int, float, bool)):
                self._raw_append(str(content))
            else:
                raise ValueError(f"Invalid content type: {type(content)}")

    # =========================================================================
    # Basic Properties
    # =========================================================================

    @property
    def root(self) -> Block:
        """Get the root block of this tree."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def text(self) -> str:
        """Get this block's local text content."""
        return self._text


    @property
    def is_root(self) -> bool:
        """True if this is the root block."""
        return self.parent is None

    @property
    def role(self) -> str:
        """
        Get the role for this block.

        Returns the block's own role if set, otherwise "user" as default.
        Role is inherited from parent when a block is added as a child.
        """
        if self._role is not None:
            return self._role
        return "user"

    @role.setter
    def role(self, value: str | None) -> None:
        """
        Set the role for this block and propagate to nested children.

        When setting a role, all nested children that don't have an explicitly
        set role will inherit this role.
        """
        self._role = value
        if value is not None:
            for child in self.children:
                for block in child.iter_depth_first():
                    if block._role is None:
                        block._role = value

    @property
    def is_wrapper(self) -> bool:
        """True if this block is a wrapper (has no local text)."""
        return self.is_empty

    @property
    def is_rendered(self) -> bool:
        """True if this block has been rendered."""
        from .mutator import Mutator
        return self.mutator is not None and not type(self.mutator) is Mutator
    
    
    def is_leaf(self) -> bool:
        """True if block has no children."""
        return len(self.body) == 0


    @property
    def depth(self) -> int:
        """Depth in tree (0 for root)."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    @property
    def length(self) -> int:
        """Length of this block's local text."""
        return len(self._text)

    @property
    def is_empty(self) -> bool:
        """True if block has no local text content."""
        return len(self._text) == 0

    @property
    def path(self) -> "IndexPath":
        """
        Get the index path for this block.

        Returns an IndexPath representing the block's position in the
        logical tree via indices (e.g., "0.2.1").

        Uses mutator.body for navigation, making style wrappers transparent.
        """
        from .path import compute_index_path
        return compute_index_path(self)

    @property
    def tag_path(self) -> "TagPath":
        """
        Get the tag path for this block.

        Returns a TagPath representing the block's semantic position
        via tags (e.g., "response.thinking").

        Collects the first tag from each block in the path.
        """
        from .path import compute_tag_path
        return compute_tag_path(self)

    # =========================================================================
    # Mutator Structure Properties
    # =========================================================================
    @property
    def content(self) -> str:
        """
        Get text from unstyled chunks only.

        Returns text from chunks that have style=None, excluding any
        styled chunks (e.g., prefixes, postfixes, formatting markers).
        This gives you the "pure content" without style decorations.
        """
        return self.mutator.content


    @property
    def head(self) -> "Block":
        """
        Get the head block via mutator.

        For simple blocks, returns self.
        For structured blocks (e.g., XML), returns the opening tag block.
        """
        return self.mutator.head

    @property
    def body(self) -> BlockChildren:
        """
        Get the body blocks (content children) via mutator.

        For simple blocks, returns all children.
        For structured blocks (e.g., XML), returns children between head and tail.
        """
        return self.mutator.body

    @property
    def tail(self) -> "Block":
        """
        Get the tail block via mutator.

        Returns the last block in the entire block tree (not just the body).
        For structured blocks (e.g., XML), returns the closing tag block after commit.
        """
        return self.mutator.tail
    
    
    def content_chunks(self) -> list[BlockChunk]:
        return self.mutator.content_chunks()

    # =========================================================================
    # Public API (delegates to mutator)
    # =========================================================================

    def append(
        self,
        content: ContentType,
        style: str | None = None,
        logprob: float | None = None,
        use_mutator_style: bool = False,
    ) -> list[Block | BlockChunk]:
        """
        Append content to this block.

        Delegates to mutator for style-aware placement.
        """
        if use_mutator_style:
            style = self.mutator.styles[0]
        if isinstance(content, Block):
            content = content.text
        elif not isinstance(content, str):
            content = str(content)
        events = []
        if self._should_use_mutator("on_append"):
            for event in self.mutator.on_append(content):
                events.append(event)
        event = self._raw_append(content, style=style, logprob=logprob)
        events.append(event)
        return events


    def prepend(
        self,
        content: ContentType,
        style: str | None = None,
        logprob: float | None = None,
    ) -> BlockChunk:
        """
        Prepend content to this block.

        Delegates to mutator for style-aware placement.
        """
        if isinstance(content, Block):
            content = content.text
        elif not isinstance(content, str):
            content = str(content)
        return self._raw_prepend(content, style=style, logprob=logprob)


    def _should_use_mutator(self, method: str) -> bool:
        from .mutator import Mutator
        if type(self.mutator) is Mutator:
            return False
        if self.mutator.is_streaming:
            return False
        if is_overridden(self.mutator.__class__, method, Mutator):
            return True
        return False
    
    def _apply_child_stylizers(self, block: Block) -> list[Block | BlockChunk]:
        from .mutator import Stylizer
        events = []
        for stylizer in self.stylizers:
            if self.mutator.is_streaming:
                continue
            if is_overridden(stylizer.__class__, "on_append_child", Stylizer):
                for event in stylizer.on_append_child(block):
                    events.append(event)
        return events

    def append_child(
        self,
        child: Block | ContentType = None,
        to_body: bool = True,
        **kwargs
    ) -> Block:
        """
        Append a child block to the body.

        Delegates to mutator for style-aware placement.

        Args:
            child: Block or content to append
            **kwargs: Passed to Block() if creating new block
        """
        from .transform import is_transforming
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)

        child = self._raw_append_child(child, to_body=to_body)
        if is_transforming():
            prev = child.prev()
            if not prev.is_wrapper and not prev.has_newline():
                prev.append("\n")    
        else:
            if to_body and self._should_use_mutator("on_append_child"):
                for event in self.mutator.on_append_child(child=child):
                    pass  # Events handled by mutator
            if to_body:
                stylizers_events = self._apply_child_stylizers(child)
        return child

    def prepend_child(
        self,
        child: Block | ContentType = None,
        **kwargs
    ) -> Block:
        """
        Prepend a child block to the body.

        Delegates to mutator for style-aware placement.
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)
        return self._raw_prepend_child(child)

    def insert_child(
        self,
        index: int,
        child: Block | ContentType = None,
        **kwargs
    ) -> Block:
        """
        Insert a child block at the given index in the body.

        Delegates to mutator for style-aware placement.
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)
        return self._raw_insert_child(index, child)

    # =========================================================================
    # Operator Overloads
    # =========================================================================

    def __itruediv__(self, other: Block | ContentType | tuple) -> Self:
        """
        /= operator for appending children.

        Usage:
            block /= "content"           # Append child with content
            block /= child_block         # Append child block
            block /= ("a", "b", "c")     # Append tuple as single child
        """
        if isinstance(other, tuple):
            if len(other) == 0:
                return self
            # Join tuple elements into a single child
            first = other[0]
            if isinstance(first, Block):
                child = first.copy()
            else:
                child = Block(first)
            for item in other[1:]:
                if isinstance(item, Block):
                    child.append_child(item.copy())
                else:
                    child.append(item)
            self.append_child(child)
        elif isinstance(other, Block):
            self.append_child(other.copy())
        else:
            self.append_child(Block(other))

        return self

    def __truediv__(self, other: Block | ContentType) -> Block:
        """
        / operator for creating child and returning it.

        Usage:
            child = parent / "content"
        """
        if isinstance(other, Block):
            return self.append_child(other.copy())
        else:
            return self.append_child(Block(other))

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on text content.

        Supports comparison with:
        - Another Block (compares text content)
        - A string (compares to block's text)

        Usage:
            b1 == b2        # Compare two blocks
            b1 == "hello"   # Compare block to string
            "hello" == b1   # Also works
        """
        if isinstance(other, Block):
            return self._text == other._text
        elif isinstance(other, str):
            return self._text == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        """
        Inequality comparison based on text content.

        Usage:
            b1 != b2        # Compare two blocks
            b1 != "hello"   # Compare block to string
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        """
        Hash based on id (not content) to allow blocks in sets/dicts.

        Note: Blocks are mutable, so we hash by id rather than content.
        """
        return hash(self.id)

    def __bool__(self) -> bool:
        """
        Boolean evaluation based on text content.

        Empty blocks are falsy, non-empty blocks are truthy.

        Usage:
            if not block:
                print("empty")
            if block:
                print("has content")
        """
        return len(self._text) > 0

    def __add__(self, other: "Block | str") -> "Block":
        """
        Concatenation operator - creates a new Block with combined text.

        Usage:
            b3 = b1 + b2          # Combine two blocks
            b3 = b1 + "world"     # Combine block with string
        """
        if isinstance(other, Block):
            new_block = Block(self._text)
            # Copy chunks from self
            for chunk in self.chunks:
                new_block.chunks.append(chunk.copy())
            # Append other's text and chunks
            offset = len(new_block._text)
            new_block._text += other._text
            for chunk in other.chunks:
                new_chunk = chunk.copy()
                new_chunk.shift(offset)
                new_block.chunks.append(new_chunk)
            return new_block
        elif isinstance(other, str):
            new_block = Block(self._text)
            for chunk in self.chunks:
                new_block.chunks.append(chunk.copy())
            new_block._raw_append(other)
            return new_block
        return NotImplemented

    def __radd__(self, other: str) -> "Block":
        """
        Right-hand concatenation for string + Block.

        Usage:
            b2 = "Hello" + b1
        """
        if isinstance(other, str):
            new_block = Block(other)
            # Append self's text and chunks
            offset = len(new_block._text)
            new_block._text += self._text
            for chunk in self.chunks:
                new_chunk = chunk.copy()
                new_chunk.shift(offset)
                new_block.chunks.append(new_chunk)
            return new_block
        return NotImplemented

    def __iadd__(self, other: "Block | str") -> Self:
        """
        In-place concatenation operator.

        Usage:
            b1 += b2          # Append b2's text to b1
            b1 += "world"     # Append string to b1
        """
        if isinstance(other, Block):
            # Append other's text and chunks
            offset = len(self._text)
            self._text += other._text
            for chunk in other.chunks:
                new_chunk = chunk.copy()
                new_chunk.shift(offset)
                self.chunks.append(new_chunk)
            return self
        elif isinstance(other, str):
            self._raw_append(other)
            return self
        return NotImplemented

    # =========================================================================
    # Raw Text Operations (used by mutators)
    # =========================================================================

    def _raw_append(
        self,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> BlockChunk:
        """
        Low-level append to this block's local text.

        Appends content at the end of this block's text.
        Creates chunk metadata with relative position.

        Returns a Chunk with the content and metadata.
        """
        target = self.head

        # Create chunk metadata with relative position
        rel_start = len(target._text)
        rel_end = rel_start + len(content)
        chunk_meta = ChunkMeta(start=rel_start, end=rel_end, logprob=logprob, style=style)
        target.chunks.append(chunk_meta)

        # Append to local text
        target._text += content

        return BlockChunk(content=content, meta=chunk_meta)

    def _raw_prepend(
        self,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> BlockChunk:
        """
        Low-level prepend to this block's local text.

        Prepends content at the start of this block's text.

        Returns a Chunk with the content and metadata.
        """
        target = self.head

        if not content:
            chunk_meta = ChunkMeta(start=0, end=0, logprob=logprob, style=style)
            target.chunks.insert(0, chunk_meta)
            return BlockChunk(content="", meta=chunk_meta)

        # Create chunk metadata at start (relative position 0)
        chunk_meta = ChunkMeta(start=0, end=len(content), logprob=logprob, style=style)

        # Shift existing chunks in this block
        delta = len(content)
        for existing_chunk in target.chunks:
            existing_chunk.shift(delta)

        target.chunks.insert(0, chunk_meta)

        # Prepend to local text
        target._text = content + target._text

        return BlockChunk(content=content, meta=chunk_meta)

    def _raw_insert(
        self,
        rel_position: int,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> BlockChunk:
        """
        Low-level insert at a relative position within this block.

        Args:
            rel_position: Position relative to start of local text
            content: Text to insert
            logprob: Optional log probability
            style: Optional chunk style

        Returns a Chunk with the content and metadata.
        """
        target = self.head

        if not content:
            chunk_meta = ChunkMeta(start=rel_position, end=rel_position, logprob=logprob, style=style)
            target._insert_chunk_sorted(chunk_meta)
            return BlockChunk(content="", meta=chunk_meta)

        delta = len(content)

        # Create chunk metadata
        chunk_meta = ChunkMeta(start=rel_position, end=rel_position + delta, logprob=logprob, style=style)

        # Shift existing chunks in this block that are at or after insertion point
        for existing_chunk in target.chunks:
            if existing_chunk.start >= rel_position:
                existing_chunk.shift(delta)
            elif existing_chunk.end > rel_position:
                # Chunk spans insertion point - extend its end
                existing_chunk.end += delta

        target._insert_chunk_sorted(chunk_meta)

        # Insert into local text
        target._text = target._text[:rel_position] + content + target._text[rel_position:]

        return BlockChunk(content=content, meta=chunk_meta)

    def _insert_chunk_sorted(self, chunk: ChunkMeta) -> None:
        """Insert chunk maintaining sorted order by start position."""
        for i, existing in enumerate(self.chunks):
            if existing.start > chunk.start:
                self.chunks.insert(i, chunk)
                return
        self.chunks.append(chunk)

    # =========================================================================
    # Raw Tree Operations (used by mutators)
    # =========================================================================

    def _connect_block(self, child: Block, parent: Block) -> None:
        """
        Connect a block to a parent, setting parent reference and inheriting role.

        Sets the parent for the child block and inherits the role for the child
        and all its descendants if their role is not explicitly set.

        Only inherits role if the parent has an explicitly set role (_role is not None).
        This prevents the default "user" role from being applied prematurely.

        Args:
            child: Block to connect
            parent: Parent block to connect to
        """
        child.parent = parent
        if parent is not None and parent._role is not None:
            parent_role = parent._role
            for block in child.iter_depth_first():
                if block._role is None:
                    block._role = parent_role

    def _raw_append_child(self, child: Block | None = None, content: str | None = None, to_body: bool = True) -> Block:
        """
        Low-level append child.

        Args:
            child: Block to append, or None to create new block
            content: Content for new block if child is None
            to_body: If True, append to body. If False, append to children.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        target = self.body if to_body else self.children
        self._connect_block(child, target.parent)
        target.append(child)

        return child

    def _raw_prepend_child(self, child: Block | None = None, content: str | None = None, to_body: bool = True) -> Block:
        """
        Low-level prepend child.

        Args:
            child: Block to prepend, or None to create new block
            content: Content for new block if child is None
            to_body: If True, prepend to body. If False, prepend to children.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        target = self.body if to_body else self.children
        self._connect_block(child, target.parent)
        target.insert(0, child)

        return child

    def _raw_insert_child(self, index: int, child: Block | None = None, content: str | None = None, to_body: bool = True) -> Block:
        """
        Low-level insert child at index.

        Args:
            index: Index in body (if to_body=True) or children (if to_body=False)
            child: Block to insert, or None to create new block
            content: Content for new block if child is None
            to_body: If True, index is relative to body. If False, index is relative to children.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        target = self.body if to_body else self.children

        if index <= 0:
            return self._raw_prepend_child(child, to_body=to_body)
        if index >= len(target):
            return self._raw_append_child(child, to_body=to_body)

        self._connect_block(child, target.parent)
        target.insert(index, child)

        return child

    def remove_child(self, child: Block) -> Block:
        """
        Remove a child block from the tree.
        """
        if child not in self.children:
            raise ValueError("Block is not a child of this block")

        child.parent = None
        self.children.remove(child)
        return child

    # =========================================================================
    # Chunk Queries
    # =========================================================================

    def get_chunk_text(self, chunk: ChunkMeta) -> str:
        """Get the text for a specific chunk."""
        return self._text[chunk.start:chunk.end]

    def get_chunk_at_position(self, rel_position: int) -> ChunkMeta | None:
        """Get the chunk containing the given relative position."""
        for chunk in self.chunks:
            if chunk.contains(rel_position):
                return chunk
        return None

    def get_chunks_by_style(self, style: str) -> list[ChunkMeta]:
        """Get all chunks with the given style."""
        return [c for c in self.chunks if c.style == style]

    def get_chunks_in_range(self, start: int, end: int) -> list[ChunkMeta]:
        """Get all chunks that overlap with the given range."""
        return [c for c in self.chunks if c.overlaps(start, end)]

    def get_logprob_at_position(self, rel_position: int) -> float | None:
        """Get the logprob at a relative position, if available."""
        chunk = self.get_chunk_at_position(rel_position)
        return chunk.logprob if chunk else None

    def get_region_text(self, style: str) -> str:
        """Get concatenated text of all chunks with the given style."""
        chunks = self.get_chunks_by_style(style)
        if not chunks:
            return ""
        parts = [self._text[c.start:c.end] for c in chunks]
        return "".join(parts)

    def get_chunks(self) -> list[BlockChunk]:
        """
        Get all chunks as Chunk objects (with content).

        Returns:
            List of Chunk objects containing both content and metadata
        """
        return [BlockChunk.from_meta(meta, self._text) for meta in self.chunks]

    # =========================================================================
    # Chunk Manipulation
    # =========================================================================

    def contains(self, s: str) -> bool:
        """
        Check if this block's text contains the given string.

        Args:
            s: String to search for

        Returns:
            True if string is found in the block's text
        """
        return s in self._text

    def chunks_contain(self, s: str) -> bool:
        """
        Check if this block's chunks contain the given string.

        The string may span multiple chunks.

        Args:
            s: String to search for

        Returns:
            True if string is found
        """
        return chunks_contain(self.get_chunks(), s)

    def _create_block_from_chunks(
        self,
        chunks: list[BlockChunk],
        inherit: bool = False,
    ) -> "Block":
        """
        Create a new Block from a list of chunks.

        Args:
            chunks: List of Chunk objects to populate the block
            inherit: If True, copy role, tags, style, attrs from this block

        Returns:
            New Block with the given chunks
        """
        if inherit:
            block = Block(
                role=self._role,
                tags=list(self.tags),
                style=list(self.style),
                attrs=dict(self.attrs),
            )
        else:
            block = Block()

        for chunk in chunks:
            block._raw_append(chunk.content, logprob=chunk.logprob, style=chunk.style)

        return block

    def split(
        self,
        sep: str,
        inherit: bool = False,
    ) -> tuple["Block", "Block", "Block"]:
        """
        Split this block's chunks on a separator.

        Returns three new Blocks: (before, separator, after).
        The separator may span multiple chunks.

        Args:
            sep: Separator string to split on
            inherit: If True, copy role, tags, style, attrs to new blocks

        Returns:
            Tuple of (before, separator, after) Blocks
        """
        chunks = self.get_chunks()
        before_chunks, sep_chunks, after_chunks = split_chunks(chunks, sep)

        before_block = self._create_block_from_chunks(before_chunks, inherit=inherit)
        sep_block = self._create_block_from_chunks(sep_chunks, inherit=inherit)
        after_block = self._create_block_from_chunks(after_chunks, inherit=inherit)

        return before_block, sep_block, after_block

    def split_prefix(
        self,
        sep: str,
        inherit: bool = False,
        create_on_empty: bool = False,
    ) -> tuple["Block", "Block"]:
        """
        Split this block's chunks, returning (before + separator, after).

        Args:
            sep: Separator string to split on
            inherit: If True, copy role, tags, style, attrs to new blocks
            create_on_empty: If True and separator not found, create a block with the separator

        Returns:
            Tuple of (prefix_block, remainder_block)
        """
        chunks = self.get_chunks()
        before_chunks, sep_chunks, after_chunks = split_chunks(chunks, sep)

        if not sep_chunks:
            if create_on_empty:
                prefix_block = Block(sep)
                remainder_block = self._create_block_from_chunks(chunks, inherit=inherit)
            else:
                prefix_block = Block()
                remainder_block = self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            prefix_block = self._create_block_from_chunks(before_chunks + sep_chunks, inherit=inherit)
            remainder_block = self._create_block_from_chunks(after_chunks, inherit=inherit)

        return prefix_block, remainder_block

    def split_postfix(
        self,
        sep: str,
        inherit: bool = False,
        create_on_empty: bool = False,
    ) -> tuple["Block", "Block"]:
        """
        Split this block's chunks, returning (before, separator + after).

        Args:
            sep: Separator string to split on
            inherit: If True, copy role, tags, style, attrs to new blocks
            create_on_empty: If True and separator not found, create a block with the separator

        Returns:
            Tuple of (content_block, postfix_block)
        """
        chunks = self.get_chunks()
        before_chunks, sep_chunks, after_chunks = split_chunks(chunks, sep)

        if not sep_chunks:
            if create_on_empty:
                content_block = self._create_block_from_chunks(chunks, inherit=inherit)
                postfix_block = Block(sep)
            else:
                content_block = self._create_block_from_chunks(chunks, inherit=inherit)
                postfix_block = Block()
        else:
            content_block = self._create_block_from_chunks(before_chunks, inherit=inherit)
            postfix_block = self._create_block_from_chunks(sep_chunks + after_chunks, inherit=inherit)

        return content_block, postfix_block

    def filter_chunks(
        self,
        styles: set[str] | str | None = None,
        inherit: bool = False,
    ) -> "Block":
        """
        Create a new Block containing only chunks with matching styles.

        Args:
            styles: Style(s) to filter by. If None, keeps all chunks.
            inherit: If True, copy role, tags, style, attrs to new block

        Returns:
            New Block with filtered chunks
        """
        if styles is None:
            return self._create_block_from_chunks(self.get_chunks(), inherit=inherit)

        if isinstance(styles, str):
            styles = {styles}

        filtered_chunks = [c for c in self.get_chunks() if c.style in styles]
        return self._create_block_from_chunks(filtered_chunks, inherit=inherit)

    # =========================================================================
    # Text Transformations
    # =========================================================================

    def lower(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Convert text to lowercase.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = [BlockChunk(c.content.lower(), logprob=c.logprob, style=c.style) for c in self.get_chunks()]
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.lower()
            return self

    def upper(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Convert text to uppercase.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = [BlockChunk(c.content.upper(), logprob=c.logprob, style=c.style) for c in self.get_chunks()]
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.upper()
            return self

    def title(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Convert text to title case.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = [BlockChunk(c.content.title(), logprob=c.logprob, style=c.style) for c in self.get_chunks()]
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.title()
            return self

    def capitalize(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Capitalize the first character.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = self.get_chunks()
            if chunks:
                first = chunks[0]
                chunks[0] = BlockChunk(first.content.capitalize(), logprob=first.logprob, style=first.style)
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.capitalize()
            return self

    def swapcase(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Swap case of all characters.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = [BlockChunk(c.content.swapcase(), logprob=c.logprob, style=c.style) for c in self.get_chunks()]
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.swapcase()
            return self

    def snake_case(self, new_block: bool = False, inherit: bool = True) -> "Block":
        """
        Convert text to snake_case.

        Replaces spaces with underscores and converts to lowercase.

        Args:
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = []
            for c in self.get_chunks():
                if c.is_whitespace:
                    chunks.append(BlockChunk("_", logprob=c.logprob, style=c.style))
                else:
                    chunks.append(BlockChunk(c.content.lower().replace(" ", "_"), logprob=c.logprob, style=c.style))
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.lower().replace(" ", "_")
            return self

    def replace(
        self,
        old: str,
        new: str,
        new_block: bool = False,
        inherit: bool = True,
    ) -> "Block":
        """
        Replace occurrences of old with new.

        Args:
            old: String to replace
            new: Replacement string
            new_block: If True, return a new block; otherwise modify in place
            inherit: If new_block is True, copy role, tags, style, attrs

        Returns:
            Self (modified) or new Block
        """
        if new_block:
            chunks = [BlockChunk(c.content.replace(old, new), logprob=c.logprob, style=c.style) for c in self.get_chunks()]
            return self._create_block_from_chunks(chunks, inherit=inherit)
        else:
            self._text = self._text.replace(old, new)
            return self

    # =========================================================================
    # Tree Traversal
    # =========================================================================

    def _iter_all_blocks(self) -> Iterator[Block]:
        """Iterate all blocks in tree (depth-first)."""
        yield self
        for child in self.children:
            yield from child._iter_all_blocks()

    def iter_depth_first(self) -> Iterator[Block]:
        """Iterate this subtree in depth-first order."""
        yield self
        for child in self.children:
            yield from child.iter_depth_first()

    def iter_ancestors(self) -> Iterator[Block]:
        """Iterate from this block up to root (inclusive)."""
        node = self
        while node is not None:
            yield node
            node = node.parent

    def next_sibling(self) -> Block | None:
        """Get next sibling or None."""
        if self.parent is None:
            return None
        siblings = self.parent.body
        for i, sibling in enumerate(siblings):
            if sibling is self:  # Use identity, not equality
                return siblings[i + 1] if i + 1 < len(siblings) else None
        return None

    def prev_sibling(self) -> Block | None:
        """Get previous sibling or None."""
        if self.parent is None:
            return None
        siblings = self.parent.body
        for i, sibling in enumerate(siblings):
            if sibling is self:  # Use identity, not equality
                return siblings[i - 1] if i > 0 else None
        return None

    def rightmost_descendant(self) -> Block:
        """
        Get the rightmost (deepest last) descendant of this block.

        If this block has no children, returns self.
        """
        if not self.children:
            return self
        return self.children[-1].rightmost_descendant()

    def prev_or_none(self) -> Block | None:
        """Get previous block in tree order, or None."""
        prev_sib = self.prev_sibling()
        if prev_sib is not None:
            return prev_sib.rightmost_descendant()
        return self.parent

    def prev(self) -> Block:
        """Get previous block in tree order."""
        result = self.prev_or_none()
        if result is None:
            raise ValueError("No previous block: this is the root")
        return result

    def next_or_none(self) -> Block | None:
        """Get next block in tree order, or None."""
        if self.children:
            return self.children[0]

        next_sib = self.next_sibling()
        if next_sib is not None:
            return next_sib

        node = self.parent
        while node is not None:
            next_sib = node.next_sibling()
            if next_sib is not None:
                return next_sib
            node = node.parent

        return None

    def next(self) -> Block:
        """Get next block in tree order."""
        result = self.next_or_none()
        if result is None:
            raise ValueError("No next block: at the end of the tree")
        return result

    def has_newline(self) -> bool:
        """Check if this block's text ends with a newline."""
        return self._text.endswith("\n")

    def add_newline(self, style: str | None = None) -> BlockChunk:
        """Add a newline to the end of this block."""
        return self._raw_append("\n", style=style)

    # =========================================================================
    # String Operations
    # =========================================================================

    def find(self, pattern: str) -> int:
        """Find pattern in this block's text. Returns position or -1."""
        return self._text.find(pattern)

    def find_all(self, pattern: str) -> list[int]:
        """Find all occurrences of pattern. Returns positions."""
        positions = []
        start = 0
        while True:
            pos = self._text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def regex_search(self, pattern: str) -> re.Match | None:
        """Search for regex pattern. Returns first match or None."""
        return re.search(pattern, self._text)

    def regex_find_all(self, pattern: str) -> list[re.Match]:
        """Find all regex matches in this block's text."""
        return list(re.finditer(pattern, self._text))

    def indent(self, spaces: int = 2, style: str | None = None):
        if not self.is_wrapper:
            self.prepend(" " * spaces, style=style or "tab")
        for child in self.children:
            child.indent(spaces, style=style)
        return self
    
    
    def apply_style(self, style: str, only_views: bool = False, copy: bool = True, recursive: bool = True):
        from .schema import BlockSchema

        block_copy = self.copy(copy) if copy else self
        styles = _parse_style(style)
        
        if recursive:
            for block in block_copy.iter_depth_first():
                if only_views and not isinstance(block, BlockSchema):
                    continue
                if block.is_leaf():
                    continue
                block.style.extend(styles)
        else:
            block_copy.style.extend(styles)
        return block_copy


    # =========================================================================
    # Tag-based Search
    # =========================================================================
    
    def __iter__(self):
        return iter(self.body)

    def __getitem__(self, key: str | int | tuple[int,...]) -> Block:
        if isinstance(key, int):
            return self.body[key]
        elif isinstance(key, str):
            return self.get(key)
        elif isinstance(key, tuple):
            return self.get_index(key)
        else:
            raise ValueError(f"Invalid key: {key}")

    def get(self, tag: str) -> Block:
        """Get first descendant with the given tag."""
        for block in self.iter_depth_first():
            if tag in block.tags:
                return block
        raise ValueError(f"Block with tag {tag} not found")
    
    
    def get_index(self, index: int | tuple[int,...]) -> Block:
        if isinstance(index, int):
            return self.body[index]
        else:
            return self.body[index[0]].get_index(index[1:] if len(index) > 2 else index[1])
    
    def get_or_none(self, tag: str) -> "Block | None":
        for block in self.iter_depth_first():
            if tag in block.tags:
                return block
        return None
    
    def get_schema(self, tag: str) -> "BlockSchema":
        from .schema import BlockSchema
        res = self.get(tag)
        if not isinstance(res, BlockSchema):
            raise ValueError(f"Block {res} is not a BlockSchema")
        return res
    
    def get_list(self, tag: str) -> "BlockList":
        from .schema import BlockList
        res = self.get_or_none(tag)        
        if res is None:
            return BlockList()        
        if not isinstance(res, BlockList):
            raise ValueError(f"Block {res} is not a BlockList")        
        return res

    def get_all(self, tag: str) -> list[Block]:
        """Get all descendants with the given tag."""
        return [b for b in self.iter_depth_first() if tag in b.tags]
    
    
    def get_all_schemas(self, tag: str) -> list["BlockSchema"]:
        from .schema import BlockSchema
        res = self.get_all(tag)
        if not all(isinstance(b, BlockSchema) for b in res):
            raise ValueError(f"Blocks {res} are not all BlockSchemas")
        return res

    def get_children_by_tag(self, tag: str) -> list[Block]:
        """Get direct children with the given tag."""
        return [c for c in self.children if tag in c.tags]

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def __call__(
        self,
        content: ContentType = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        **attrs
    ) -> Block:
        """
        Create and append a child block.

        Usage:
            with root("Hello") as child:
                child("Nested")
        """
        child = Block(
            content,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
        )
        # return self._raw_append_child(child)
        return self.append_child(child)
    
    

    def view(
        self,
        name: str,
        type: Type | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        is_required: bool = True,
    ) -> "BlockSchema":
        from .schema import BlockSchema
        schema = BlockSchema(name, type=type, role=role, tags=tags, style=style, attrs=attrs, is_required=is_required)
        # self._raw_append_child(schema)
        return self.append_child(schema)
        return schema
    
    @classmethod
    def schema_view(
        cls, 
        name: str | None = None, 
        type: Type | None = None, 
        tags: list[str] | None = None, 
        style: str | None = None, 
        attrs: dict[str, Any] | None = None,
        is_required: bool = True,
    ) -> "BlockSchema":
        from .schema import BlockSchema
        schema_block = BlockSchema(
            name,
            type=type,
            tags=tags,
            attrs=attrs,
            # style=["xml"] if style is None and name is not None else parse_style(style),
            style=style, 
            is_required=is_required,
        )
        return schema_block
    
    
    def view_list(
        self,
        item_name: str,
        key: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,        
        style: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "BlockListSchema":
        from .schema import BlockListSchema
        schema_block = BlockListSchema(
            item_name=item_name,
            key=key,
            name=name,
            tags=tags,
            attrs=attrs,
            style=style,
        )
        return self.append_child(schema_block)
        # self._raw_append_child(schema_block)
        return schema_block


    # =========================================================================
    # Copy Operations
    # =========================================================================
    
    def extract(self) -> Block:        
        ex_block = self.mutator.extract()        
        for child in self.body:
            ex_child = child.extract()
            ex_block.append_child(ex_child)
        return ex_block
        

    def copy(self, deep: bool = True) -> Block:
        """
        Create a copy of this block.

        Args:
            deep: If True, copy entire subtree. If False, just this block.

        Returns:
            New block with copied content
        """
        new_block = Block(
            role=self._role,
            tags=list(self.tags),
            style=list(self.style),
            attrs=dict(self.attrs),
        )
        new_block._text = self._text
        new_block.chunks = [c.copy() for c in self.chunks]

        if deep:
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_block._raw_append_child(child_copy, to_body=False)

        return new_block

    
    def copy_head(self) -> Block:
        return self.copy(deep=False)
    # =========================================================================
    # Serialization
    # =========================================================================

    def model_dump(self) -> dict[str, Any]:
        """Serialize block to dictionary."""
        return {
            "id": self.id,
            "path": str(self.path),
            "text": self._text,
            "role": self._role,
            "tags": self.tags,
            "style": self.style,
            "attrs": self.attrs,
            "chunks": [
                {
                    "id": c.id,
                    "start": c.start,
                    "end": c.end,
                    "logprob": c.logprob,
                    "style": c.style,
                }
                for c in self.chunks
            ],
            "children": [c.model_dump() for c in self.children],
        }

    @classmethod
    def model_load(cls, data: dict[str, Any]) -> Block:
        """Deserialize block from dictionary."""
        block = cls(
            role=data.get("role"),
            tags=data.get("tags", []),
            style=data.get("style", []),
            attrs=data.get("attrs", {}),
        )
        block.id = data.get("id", _generate_id())
        block._text = data.get("text", "")
        block.chunks = [
            ChunkMeta(
                id=c.get("id", _generate_id()),
                start=c["start"],
                end=c["end"],
                logprob=c.get("logprob"),
                style=c.get("style"),
            )
            for c in data.get("chunks", [])
        ]

        for child_data in data.get("children", []):
            child = cls.model_load(child_data)
            child.parent = block
            block.children.append(child)

        return block

    # =========================================================================
    # Rendering
    # =========================================================================

    def commit(self, postfix: Block | None = None) -> Block | None:
        post_block = self.mutator.call_commit(postfix)
        return post_block

    def transform(self) -> Block:
        from .transform import transform
        return transform(self)

    def render(self) -> str:
        """Render this block tree to a string by concatenating all text depth-first."""
        parts = []
        tran_block = self.transform()
        for block in tran_block.iter_depth_first():
            parts.append(block._text)
        return "".join(parts)

    def print(self) -> None:
        print(self.render())

    # =========================================================================
    # Schema Extraction
    # =========================================================================

    def extract_schema(
        self,
        style: str | list[str] | None = None,
        root: str | None = None,
        role: str | None = None,
    ) -> "BlockSchema | None":
        """
        Extract a new BlockSchema tree containing only BlockSchema nodes.

        Traverses this block's subtree and creates a new BlockSchema with
        only BlockSchema children, preserving the schema hierarchy while
        filtering out regular Block nodes.

        Args:
            style: Optional style(s) to apply to the extracted schema tree
            root: Optional root tag name. Only used when extracting from a
                  regular Block with multiple schema children. If not provided
                  and multiple schemas exist at root level, returns None.
            role: Optional role to set on each extracted schema block

        Returns:
            A new BlockSchema tree with only schema nodes, or None if no schemas found
        """
        from .schema import BlockSchema, BlockListSchema

        # Create a schema from this block
        if isinstance(self, BlockSchema):
            # Already a schema - copy without children (this IS the root)
            new_schema = self.copy(deep=False)

            # Apply style if provided
            if style:
                styles = _parse_style(style)
                for s in styles:
                    if s not in new_schema.style:
                        new_schema.style.append(s)

            # Apply role if provided
            if role is not None:
                new_schema.role = role
        else:
            # Regular block - don't create a schema, just collect children
            new_schema = None

        # Collect schema children
        schema_children = []
        for child in self.children:
            if isinstance(child, (BlockSchema, BlockListSchema)):
                child_schema = child.extract_schema(style=style, role=role)
                if child_schema is not None:
                    schema_children.append(child_schema)
            else:
                # For regular blocks, search their children for nested schemas
                self._collect_nested_schemas(child, schema_children, style=style, role=role)

        if new_schema is not None:
            # Already have a root schema - add collected children
            for child_schema in schema_children:
                new_schema._raw_append_child(child_schema)
            return new_schema
        else:
            # No parent schema - return based on number of children found
            if len(schema_children) == 0:
                return None
            elif len(schema_children) == 1:
                # Single schema child becomes the root
                return schema_children[0]
            else:
                # Multiple schemas at root level - need a wrapper
                if root is None:
                    # No root name provided - can't create proper wrapper
                    return None
                wrapper_schema = BlockSchema(
                    name=root,
                    style="block",
                    is_root=True,
                    role=role,
                    # style=_parse_style(style) if style else [],
                )
                for child_schema in schema_children:
                    wrapper_schema._raw_append_child(child_schema)
                return wrapper_schema

    def _collect_nested_schemas(
        self,
        block: "Block",
        result: list,
        style: str | list[str] | None = None,
        role: str | None = None,
    ) -> None:
        """
        Recursively search a Block's children for BlockSchema nodes.

        Collects found BlockSchema nodes into the result list.

        Args:
            block: Block to search
            result: List to append found schemas to
            style: Style to apply to extracted schemas
            role: Role to set on extracted schemas
        """
        from .schema import BlockSchema, BlockListSchema

        for child in block.children:
            if isinstance(child, (BlockSchema, BlockListSchema)):
                child_schema = child.extract_schema(style=style, role=role)
                if child_schema is not None:
                    result.append(child_schema)
            else:
                # Keep searching deeper
                self._collect_nested_schemas(child, result, style=style, role=role)
                
                
    # =========================================================================
    # pydantic support
    # =========================================================================

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, Block):
            return v
        elif isinstance(v, dict):
            # if "_type" in v and v["_type"] == "Block":
            #     return Block.model_validate(v)
            return Block.model_load(v)
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")

    # =========================================================================
    # Debug
    # =========================================================================

    def debug_tree(self, indent: int = 0) -> str:
        """Generate debug representation of block tree."""
        prefix = "  " * indent
        cls_name = self.__class__.__name__
        parts = [f"{prefix}{cls_name}[{self.path}]("]
        
        
        content_preview = self._text[:30]
        if len(self._text) > 30:
            content_preview += "..."
            
        if len(self._text) > 0:        
            parts.append(f"{content_preview!r}")
            
            
        # parts.append(f", path={self.path}")

        if self.tags:
            parts.append(f", tags={self.tags}")
        if self.role:
            parts.append(f", role={self.role!r}")
        if self.style:
            parts.append(f", style={self.style}")



        if self.chunks:
            chunk_styles = [c.style if c.style else 'txt' for c in self.chunks]
            if chunk_styles:
                parts.append(f", chunk_styles={chunk_styles}")
                
        parts.append(")")

        lines = ["".join(parts)]

        for child in self.children:
            lines.append(child.debug_tree(indent + 1))

        return "\n".join(lines)

    def print_debug(self) -> None:
        """Print debug tree."""
        print(self.debug_tree())

    def __repr__(self) -> str:
        content_preview = self._text[:20] if self._text else ""
        if len(self._text) > 20:
            content_preview += "..."
        
        block_meta = ""
        if self.tags:
            block_meta += f", tags={self.tags}"
        if self.role:
            block_meta += f", role={self.role}"
        if self.style:
            block_meta += f", style={self.style}"
        if self.attrs:
            block_meta += f", attrs={self.attrs}"
        return f"Block({content_preview!r}{block_meta}, children={len(self.children)})"


def _parse_style(style: str | list[str] | None) -> list[str]:
    """Parse style into list of style strings."""
    if style is None:
        return []
    if isinstance(style, str):
        return style.split() if " " in style else [style]
    return list(style)
