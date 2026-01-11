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
import re

from promptview.utils.function_utils import is_overridden

from .chunk import ChunkMeta, Chunk

if TYPE_CHECKING:
    from .mutator import Mutator
    from .schema import BlockSchema


def _generate_id() -> str:
    """Generate a short unique ID."""
    from uuid import uuid4
    return uuid4().hex[:8]


# Type for content that can be passed to Block
ContentType = Union[str, int, float, bool, "Block", None]


class BlockChildren(UserList["Block"]):

    def __init__(self, parent: Block, items: list[Block] | None = None):
        self.parent: Block = parent
        UserList.__init__(self, items)

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
        "role", "tags", "style", "attrs", "_text", "id", "mutator"
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
        from .mutator import Mutator
        # Tree structure
        self.parent: Block | None = None
        self.children: BlockChildren = BlockChildren(parent=self)

        # Chunk metadata (relative to local _text)
        self.chunks: list[ChunkMeta] = []

        # Block metadata
        self.role = role
        self.tags = tags or []
        self.style = _parse_style(style)
        self.attrs = attrs or {}

        # Local text for this block
        self._text: str = ""

        # ID
        self.id: str = _generate_id()

        # Mutator (lazy initialized)
        self.mutator: Mutator = Mutator(self)

        # Handle initial content
        if content is not None:
            if isinstance(content, Block):
                # Copy content from another block
                self._raw_append(content.text)
            elif isinstance(content, (str, int, float, bool)):
                self._raw_append(str(content))

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
    def content(self) -> str:
        """
        Get text from unstyled chunks only.

        Returns text from chunks that have style=None, excluding any
        styled chunks (e.g., prefixes, postfixes, formatting markers).
        This gives you the "pure content" without style decorations.
        """
        if not self.chunks:
            return self._text

        content_parts = []
        for chunk in self.chunks:
            if chunk.style is None:
                content_parts.append(self._text[chunk.start:chunk.end])
        return "".join(content_parts)

    @property
    def is_root(self) -> bool:
        """True if this is the root block."""
        return self.parent is None

    @property
    def is_wrapper(self) -> bool:
        """True if this block is a wrapper (has no local text)."""
        return self.is_empty

    @property
    def is_rendered(self) -> bool:
        """True if this block has been rendered."""
        from .mutator import Mutator
        return self.mutator is not None and not type(self.mutator) is Mutator

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

    # =========================================================================
    # Mutator Structure Properties
    # =========================================================================

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

    # =========================================================================
    # Public API (delegates to mutator)
    # =========================================================================

    def append(
        self,
        content: ContentType,
        style: str | None = None,
        logprob: float | None = None,
    ) -> list[Block | Chunk]:
        """
        Append content to this block.

        Delegates to mutator for style-aware placement.
        """
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
    ) -> Chunk:
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
        if is_overridden(self.mutator.__class__, method, Mutator):
            return True
        return False

    def append_child(
        self,
        child: Block | ContentType = None,
        **kwargs
    ) -> Block:
        """
        Append a child block to the body.

        Delegates to mutator for style-aware placement.

        Args:
            child: Block or content to append
            **kwargs: Passed to Block() if creating new block
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)

        child = self._raw_append_child(child)
        if self._should_use_mutator("on_append_child"):
            for event in self.mutator.on_append_child(child=child):
                pass  # Events handled by mutator

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

    # =========================================================================
    # Raw Text Operations (used by mutators)
    # =========================================================================

    def _raw_append(
        self,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> Chunk:
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

        return Chunk(content=content, meta=chunk_meta)

    def _raw_prepend(
        self,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> Chunk:
        """
        Low-level prepend to this block's local text.

        Prepends content at the start of this block's text.

        Returns a Chunk with the content and metadata.
        """
        target = self.head

        if not content:
            chunk_meta = ChunkMeta(start=0, end=0, logprob=logprob, style=style)
            target.chunks.insert(0, chunk_meta)
            return Chunk(content="", meta=chunk_meta)

        # Create chunk metadata at start (relative position 0)
        chunk_meta = ChunkMeta(start=0, end=len(content), logprob=logprob, style=style)

        # Shift existing chunks in this block
        delta = len(content)
        for existing_chunk in target.chunks:
            existing_chunk.shift(delta)

        target.chunks.insert(0, chunk_meta)

        # Prepend to local text
        target._text = content + target._text

        return Chunk(content=content, meta=chunk_meta)

    def _raw_insert(
        self,
        rel_position: int,
        content: str,
        style: str | None = None,
        logprob: float | None = None,
    ) -> Chunk:
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
            return Chunk(content="", meta=chunk_meta)

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

        return Chunk(content=content, meta=chunk_meta)

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
        child.parent = target.parent
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
        child.parent = target.parent
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

        child.parent = target.parent
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
        try:
            idx = siblings.index(self)
            return siblings[idx + 1] if idx + 1 < len(siblings) else None
        except ValueError:
            return None

    def prev_sibling(self) -> Block | None:
        """Get previous sibling or None."""
        if self.parent is None:
            return None
        siblings = self.parent.body
        try:
            idx = siblings.index(self)
            return siblings[idx - 1] if idx > 0 else None
        except ValueError:
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

    def add_newline(self, style: str | None = None) -> Chunk:
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

    # =========================================================================
    # Tag-based Search
    # =========================================================================

    def __getitem__(self, key: str | int) -> Block:
        if isinstance(key, int):
            return self.body[key]
        elif isinstance(key, str):
            return self.get_by_tag(key)
        else:
            raise ValueError(f"Invalid key: {key}")

    def get_by_tag(self, tag: str) -> Block:
        """Get first descendant with the given tag."""
        for block in self.iter_depth_first():
            if tag in block.tags:
                return block
        raise ValueError(f"Block with tag {tag} not found")

    def get_all_by_tag(self, tag: str) -> list[Block]:
        """Get all descendants with the given tag."""
        return [b for b in self.iter_depth_first() if tag in b.tags]

    def get_children_by_tag(self, tag: str) -> list[Block]:
        """Get direct children with the given tag."""
        return [c for c in self.children if tag in c.tags]

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> Block:
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
        return self._raw_append_child(child)

    def view(
        self,
        name: str,
        type: Type | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "BlockSchema":
        from .schema import BlockSchema
        schema = BlockSchema(name, type=type, role=role, tags=tags, style=style, attrs=attrs)
        self._raw_append_child(schema)
        return schema


    # =========================================================================
    # Copy Operations
    # =========================================================================

    def copy(self, deep: bool = True) -> Block:
        """
        Create a copy of this block.

        Args:
            deep: If True, copy entire subtree. If False, just this block.

        Returns:
            New block with copied content
        """
        new_block = Block(
            role=self.role,
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

    # =========================================================================
    # Serialization
    # =========================================================================

    def model_dump(self) -> dict[str, Any]:
        """Serialize block to dictionary."""
        return {
            "id": self.id,
            "text": self._text,
            "role": self.role,
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

    def commit(self) -> Block | None:
        post_block = self.mutator.commit(self)
        self.mutator._committed = True
        return post_block

    def transform(self) -> Block:
        from .transform import transform
        return transform(self)

    def render(self) -> str:
        """Render this block tree to a string by concatenating all text depth-first."""
        parts = []
        for block in self.iter_depth_first():
            parts.append(block._text)
        return "".join(parts)

    def print(self) -> None:
        print(self.transform().render())

    # =========================================================================
    # Debug
    # =========================================================================

    def debug_tree(self, indent: int = 0) -> str:
        """Generate debug representation of block tree."""
        prefix = "  " * indent
        parts = [f"{prefix}Block"]

        if self.tags:
            parts.append(f" tags={self.tags}")
        if self.role:
            parts.append(f" role={self.role!r}")
        if self.style:
            parts.append(f" style={self.style}")

        content_preview = self._text[:30]
        if len(self._text) > 30:
            content_preview += "..."
        parts.append(f" text={content_preview!r}")

        if self.chunks:
            chunk_styles = [c.style if c.style else 'txt' for c in self.chunks]
            if chunk_styles:
                parts.append(f" chunk_styles={chunk_styles}")

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
        return f"Block({content_preview!r}, children={len(self.children)})"


def _parse_style(style: str | list[str] | None) -> list[str]:
    """Parse style into list of style strings."""
    if style is None:
        return []
    if isinstance(style, str):
        return style.split() if " " in style else [style]
    return list(style)
