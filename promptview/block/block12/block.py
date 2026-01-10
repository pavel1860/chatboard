"""
Block - Unified tree node with text positions and chunk metadata.

Block12 simplifies the block system by:
1. Storing the entire document as a single string (on root)
2. Using the block tree for structure (no separate span chain)
3. Using relative chunk positions (no shifting on block moves)
4. Delegating style-aware operations to Mutators

Usage:
    with Block("Header") as b:
        b /= "Hello"
        b /= "World"
"""

from __future__ import annotations
from typing import Any, Iterator, TYPE_CHECKING, Union
import re

from promptview.utils.function_utils import is_overridden

from .chunk import ChunkMeta, Chunk

if TYPE_CHECKING:
    from .mutator import Mutator


def _generate_id() -> str:
    """Generate a short unique ID."""
    from uuid import uuid4
    return uuid4().hex[:8]


# Type for content that can be passed to Block
ContentType = Union[str, int, float, bool, "Block", None]


class Block:
    """
    Tree node with text positions and chunk metadata.

    The root block owns the shared string (_text). All other blocks
    reference positions within that string via start/end.

    Chunk metadata uses positions relative to the block's start,
    so chunks don't need updating when the block moves.

    Public operations (append, append_child) delegate to the mutator
    for style-aware behavior. Raw operations (_raw_*) bypass the mutator.

    Usage:
        with Block("Header") as b:
            b /= "First child"
            b /= "Second child"

    Attributes:
        parent: Parent block (None for root)
        children: Child blocks
        start: Start position in root._text (absolute)
        end: End position in root._text (absolute)
        chunks: Chunk metadata (positions relative to self.start)
        role: Role identifier (e.g., "user", "assistant")
        tags: Tag list for querying
        style: Style list for rendering/mutator selection
        attrs: Arbitrary attributes
        _text: The shared string (only meaningful on root)
    """

    __slots__ = [
        "parent", "children", "start", "end", "chunks",
        "role", "tags", "style", "attrs", "_text", "id", "_mutator"
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
        # Tree structure
        self.parent: Block | None = None
        self.children: list[Block] = []

        # Position in shared string (absolute)
        self.start: int = 0
        self.end: int = 0

        # Chunk metadata (relative positions)
        self.chunks: list[ChunkMeta] = []

        # Block metadata
        self.role = role
        self.tags = tags or []
        self.style = _parse_style(style)
        self.attrs = attrs or {}

        # Shared string (only root)
        self._text: str = ""

        # ID
        self.id: str = _generate_id()

        # Mutator (lazy initialized)
        self._mutator: Mutator | None = None

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
        """Get this block's text content."""
        return self.root._text[self.start:self.end]

    @property
    def is_root(self) -> bool:
        """True if this is the root block."""
        return self.parent is None

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
        """Length of this block's text."""
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        """True if block has no text content."""
        return self.start == self.end

    # =========================================================================
    # Mutator Access
    # =========================================================================

    @property
    def mutator(self) -> Mutator:
        """Get mutator for this block's style."""
        if self._mutator is None:
            from .mutator import Mutator, MutatorMeta
            style = self.style[0] if self.style else "default"
            mutator_cls = MutatorMeta.get_mutator(style)
            self._mutator = mutator_cls(self)
        return self._mutator

    # =========================================================================
    # Public API (delegates to mutator)
    # =========================================================================

    def append(
        self,
        content: ContentType,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Append content to this block.

        Delegates to mutator for style-aware placement.
        """
        if isinstance(content, Block):
            content = content.text
        elif not isinstance(content, str):
            content = str(content)
        return self.mutator.append(content, logprob=logprob, style=style)

    def prepend(
        self,
        content: ContentType,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Prepend content to this block.

        Delegates to mutator for style-aware placement.
        """
        if isinstance(content, Block):
            content = content.text
        elif not isinstance(content, str):
            content = str(content)
        return self.mutator.prepend(content, logprob=logprob, style=style)
    
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
        Append a child block.

        Delegates to mutator for style-aware placement.

        Args:
            child: Block or content to append
            **kwargs: Passed to Block() if creating new block
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)
        events = []
        # if self.mutator and is_overridden(self.mutator.__class__, "on_append", Mutator):
        child = self._raw_append_child(child)
        if self._should_use_mutator("on_append_child"):
            for event in self.mutator.on_append_child(child=child):
                events.append(event)        
        
        return child

    def prepend_child(
        self,
        child: Block | ContentType = None,
        **kwargs
    ) -> Block:
        """
        Prepend a child block.

        Delegates to mutator for style-aware placement.
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)
        return self.mutator.prepend_child(child=child)

    def insert_child(
        self,
        index: int,
        child: Block | ContentType = None,
        **kwargs
    ) -> Block:
        """
        Insert a child block at the given index.

        Delegates to mutator for style-aware placement.
        """
        if child is None:
            child = Block(**kwargs)
        elif not isinstance(child, Block):
            child = Block(child, **kwargs)
        return self.mutator.insert_child(index, child=child)

    # =========================================================================
    # Operator Overloads
    # =========================================================================

    def __itruediv__(self, other: Block | ContentType | tuple) -> Block:
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
        logprob: float | None = None,
        style: str | None = None
    ) -> Chunk:
        """
        Low-level append without mutator interception.

        Appends content at the end of this block's text region.
        Creates chunk metadata with relative position.
        Shifts positions of subsequent blocks.

        Returns a Chunk with the content and metadata.
        """
        if not content:
            # Empty content - create empty chunk for metadata tracking
            rel_start = self.end - self.start
            chunk_meta = ChunkMeta(start=rel_start, end=rel_start, logprob=logprob, style=style)
            self.chunks.append(chunk_meta)
            return Chunk(content="", meta=chunk_meta)

        root = self.root

        # Create chunk metadata with relative position
        rel_start = self.end - self.start
        rel_end = rel_start + len(content)
        chunk_meta = ChunkMeta(start=rel_start, end=rel_end, logprob=logprob, style=style)
        self.chunks.append(chunk_meta)

        # Insert into shared string
        insert_pos = self.end
        root._text = root._text[:insert_pos] + content + root._text[insert_pos:]

        # Update this block's end
        delta = len(content)
        self.end += delta

        # Shift all blocks after insertion point
        self._shift_positions_after(insert_pos, delta)

        return Chunk(content=content, meta=chunk_meta)

    def _raw_prepend(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> Chunk:
        """
        Low-level prepend without mutator interception.

        Prepends content at the start of this block's text region.

        Returns a Chunk with the content and metadata.
        """
        if not content:
            chunk_meta = ChunkMeta(start=0, end=0, logprob=logprob, style=style)
            self.chunks.insert(0, chunk_meta)
            return Chunk(content="", meta=chunk_meta)

        root = self.root

        # Create chunk metadata at start (relative position 0)
        chunk_meta = ChunkMeta(start=0, end=len(content), logprob=logprob, style=style)

        # Shift existing chunks in this block
        delta = len(content)
        for existing_chunk in self.chunks:
            existing_chunk.shift(delta)

        self.chunks.insert(0, chunk_meta)

        # Insert into shared string
        insert_pos = self.start
        root._text = root._text[:insert_pos] + content + root._text[insert_pos:]

        # Update this block's end (start stays same, content inserted at start)
        self.end += delta

        # Shift all blocks after insertion point
        self._shift_positions_after(insert_pos, delta)

        return Chunk(content=content, meta=chunk_meta)

    def _raw_insert(
        self,
        rel_position: int,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> Chunk:
        """
        Low-level insert at a relative position within this block.

        Used by mutators for style-aware insertion (e.g., before postfix).

        Args:
            rel_position: Position relative to self.start
            content: Text to insert
            logprob: Optional log probability
            style: Optional chunk style

        Returns a Chunk with the content and metadata.
        """
        if not content:
            chunk_meta = ChunkMeta(start=rel_position, end=rel_position, logprob=logprob, style=style)
            self._insert_chunk_sorted(chunk_meta)
            return Chunk(content="", meta=chunk_meta)

        root = self.root
        abs_position = self.start + rel_position
        delta = len(content)

        # Create chunk metadata
        chunk_meta = ChunkMeta(start=rel_position, end=rel_position + delta, logprob=logprob, style=style)

        # Shift existing chunks in this block that are at or after insertion point
        for existing_chunk in self.chunks:
            if existing_chunk.start >= rel_position:
                existing_chunk.shift(delta)
            elif existing_chunk.end > rel_position:
                # Chunk spans insertion point - extend its end
                existing_chunk.end += delta

        self._insert_chunk_sorted(chunk_meta)

        # Insert into shared string
        root._text = root._text[:abs_position] + content + root._text[abs_position:]

        # Update this block's end
        self.end += delta

        # Shift all blocks after insertion point
        self._shift_positions_after(abs_position, delta)

        return Chunk(content=content, meta=chunk_meta)

    def _insert_chunk_sorted(self, chunk: ChunkMeta) -> None:
        """Insert chunk maintaining sorted order by start position."""
        for i, existing in enumerate(self.chunks):
            if existing.start > chunk.start:
                self.chunks.insert(i, chunk)
                return
        self.chunks.append(chunk)

    # =========================================================================
    # Position Shifting
    # =========================================================================

    def _shift_positions_after(self, position: int, delta: int) -> None:
        """
        Shift start/end of all blocks with positions after position.

        Called after inserting text to update subsequent blocks.
        Does NOT update ancestors (caller handles that separately).
        Does NOT update self.
        """
        for block in self.root._iter_all_blocks():
            if block is self:
                continue
            # Skip ancestors (handled separately)
            if self._is_ancestor(block):
                continue

            if block.start >= position:
                block.start += delta
            if block.end > position:
                block.end += delta

    def _extend_ancestors(self, delta: int) -> None:
        """Extend all ancestor blocks' end positions."""
        node = self.parent
        while node is not None:
            node.end += delta
            node = node.parent

    def _contract_ancestors(self, delta: int) -> None:
        """Contract all ancestor blocks' end positions."""
        self._extend_ancestors(-delta)

    def _is_ancestor(self, block: Block) -> bool:
        """Check if block is an ancestor of self."""
        node = self.parent
        while node is not None:
            if node is block:
                return True
            node = node.parent
        return False

    # =========================================================================
    # Raw Tree Operations (used by mutators)
    # =========================================================================

    def _raw_append_child(self, child: Block | None = None, content: str | None = None) -> Block:
        """
        Low-level append child without mutator interception.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        # Determine insertion position
        if self.children:
            insert_pos = self.children[-1].end
        else:
            insert_pos = self.end

        # Merge child's text if it has any
        if child._text:
            self._merge_child_text(child, insert_pos)
        else:
            # Child has no text yet, just set its position
            child.start = insert_pos
            child.end = insert_pos

        # Set up tree relationship
        child.parent = self
        self.children.append(child)

        return child

    def _raw_prepend_child(self, child: Block | None = None, content: str | None = None) -> Block:
        """
        Low-level prepend child without mutator interception.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        # Insertion position is at this block's start
        insert_pos = self.start

        # Merge child's text if it has any
        if child._text:
            self._merge_child_text(child, insert_pos)
        else:
            child.start = insert_pos
            child.end = insert_pos

        # Set up tree relationship
        child.parent = self
        self.children.insert(0, child)

        return child

    def _raw_insert_child(self, index: int, child: Block | None = None, content: str | None = None) -> Block:
        """
        Low-level insert child at index without mutator interception.
        """
        if child is None:
            child = Block()
            if content:
                child._raw_append(content)

        # Determine insertion position
        if index <= 0:
            return self._raw_prepend_child(child)
        elif index >= len(self.children):
            return self._raw_append_child(child)
        else:
            insert_pos = self.children[index - 1].end

        # Merge child's text if it has any
        if child._text:
            self._merge_child_text(child, insert_pos)
        else:
            child.start = insert_pos
            child.end = insert_pos

        # Set up tree relationship
        child.parent = self
        self.children.insert(index, child)

        return child

    def _merge_child_text(self, child: Block, insert_pos: int) -> None:
        """Merge a child's text into the root's shared string."""
        root = self.root
        child_text = child._text
        delta = len(child_text)

        if not child_text:
            return

        # Insert child's text into shared string
        root._text = root._text[:insert_pos] + child_text + root._text[insert_pos:]

        # Calculate offset for remapping child positions
        offset = insert_pos - child.start

        # Remap child and all its descendants
        self._remap_subtree(child, offset)

        # Clear child's local text (now using root's)
        child._text = ""

        # Shift blocks after insertion
        self._shift_positions_after(insert_pos, delta)

    def _remap_subtree(self, block: Block, offset: int) -> None:
        """Remap positions in a subtree by offset."""
        block.start += offset
        block.end += offset
        for child in block.children:
            self._remap_subtree(child, offset)

    def remove_child(self, child: Block) -> Block:
        """
        Remove a child block from the tree.

        Note: This removes the block from the tree but does NOT remove
        its text from the shared string (text becomes orphaned).
        """
        if child not in self.children:
            raise ValueError("Block is not a child of this block")

        child.parent = None
        self.children.remove(child)
        return child

    def remove_child_with_text(self, child: Block) -> Block:
        """
        Remove a child block and its text from the shared string.
        """
        if child not in self.children:
            raise ValueError("Block is not a child of this block")

        root = self.root
        child_text = root._text[child.start:child.end]
        delta = len(child_text)

        if delta > 0:
            # Remove text from shared string
            root._text = root._text[:child.start] + root._text[child.end:]

            # Shift blocks after removal
            self._shift_positions_after(child.start, -delta)

        # Store text locally on child
        child._text = child_text
        child.start = 0
        child.end = len(child_text)

        # Remap child's chunks (they're already relative, but subtree positions need reset)
        for descendant in child._iter_all_blocks():
            if descendant is not child:
                descendant.start -= child.start
                descendant.end -= child.start

        # Remove from tree
        child.parent = None
        self.children.remove(child)

        return child

    # =========================================================================
    # Chunk Queries
    # =========================================================================

    def get_chunk_text(self, chunk: ChunkMeta) -> str:
        """Get the text for a specific chunk."""
        return self.text[chunk.start:chunk.end]

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
        # Chunks should be contiguous for a region, but handle gaps
        parts = [self.text[c.start:c.end] for c in chunks]
        return "".join(parts)

    # =========================================================================
    # Tree Traversal
    # =========================================================================

    def _iter_all_blocks(self) -> Iterator[Block]:
        """Iterate all blocks in tree from root (depth-first)."""
        yield self
        for child in self.children:
            yield from child._iter_all_blocks()

    def iter_depth_first(self) -> Iterator[Block]:
        """Iterate this subtree in depth-first order (text order)."""
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
        siblings = self.parent.children
        try:
            idx = siblings.index(self)
            return siblings[idx + 1] if idx + 1 < len(siblings) else None
        except ValueError:
            return None

    def prev_sibling(self) -> Block | None:
        """Get previous sibling or None."""
        if self.parent is None:
            return None
        siblings = self.parent.children
        try:
            idx = siblings.index(self)
            return siblings[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    def prev_or_none(self) -> Block | None:
        """
        Get previous block in tree order, or None.

        Returns previous sibling if exists, otherwise returns parent.
        Returns None if this is the root.
        """
        prev_sib = self.prev_sibling()
        if prev_sib is not None:
            return prev_sib
        return self.parent

    def prev(self) -> Block:
        """
        Get previous block in tree order.

        Returns previous sibling if exists, otherwise returns parent.
        Raises ValueError if this is the root (no previous block).
        """
        result = self.prev_or_none()
        if result is None:
            raise ValueError("No previous block: this is the root")
        return result

    def next_or_none(self) -> Block | None:
        """
        Get next block in tree order, or None.

        Returns next sibling if exists, otherwise walks up ancestors
        to find the next block in the lineage.
        Returns None if at the end of the tree.
        """
        # First check for next sibling
        next_sib = self.next_sibling()
        if next_sib is not None:
            return next_sib

        # Walk up to find next in lineage
        node = self.parent
        while node is not None:
            next_sib = node.next_sibling()
            if next_sib is not None:
                return next_sib
            node = node.parent

        return None

    def next(self) -> Block:
        """
        Get next block in tree order.

        Returns next sibling if exists, otherwise walks up ancestors
        to find the next block in the lineage.
        Raises ValueError if at the end of the tree.
        """
        result = self.next_or_none()
        if result is None:
            raise ValueError("No next block: at the end of the tree")
        return result

    def has_newline(self) -> bool:
        """Check if this block's text ends with a newline."""
        return self.text.endswith("\n")

    def add_newline(self, style: str | None = None) -> Chunk:
        """
        Add a newline to the end of this block.

        Returns a Chunk for the added newline.
        """
        return self._raw_append("\n", style=style)

    # =========================================================================
    # String Operations
    # =========================================================================

    def find(self, pattern: str) -> int:
        """Find pattern in this block's text. Returns relative position or -1."""
        return self.text.find(pattern)

    def find_all(self, pattern: str) -> list[int]:
        """Find all occurrences of pattern. Returns relative positions."""
        text = self.text
        positions = []
        start = 0
        while True:
            pos = text.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def regex_search(self, pattern: str) -> re.Match | None:
        """Search for regex pattern. Returns first match or None."""
        return re.search(pattern, self.text)

    def regex_find_all(self, pattern: str) -> list[re.Match]:
        """Find all regex matches in this block's text."""
        return list(re.finditer(pattern, self.text))

    # =========================================================================
    # Tag-based Search
    # =========================================================================

    def get_by_tag(self, tag: str) -> Block | None:
        """Get first descendant with the given tag."""
        for block in self.iter_depth_first():
            if tag in block.tags:
                return block
        return None

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

    # =========================================================================
    # Copy Operations
    # =========================================================================

    def copy(self, deep: bool = True) -> Block:
        """
        Create a copy of this block.

        Args:
            deep: If True, copy entire subtree. If False, just this block.

        Returns:
            New block with copied text stored locally
        """
        new_block = Block(
            role=self.role,
            tags=list(self.tags),
            style=list(self.style),
            attrs=dict(self.attrs),
        )
        new_block._text = self.text
        new_block.start = 0
        new_block.end = len(self.text)
        new_block.chunks = [c.copy() for c in self.chunks]

        if deep:
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_block._raw_append_child(child_copy)

        return new_block

    # =========================================================================
    # Serialization
    # =========================================================================

    def model_dump(self, include_text: bool = True) -> dict[str, Any]:
        """Serialize block to dictionary."""
        data = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
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
            "children": [c.model_dump(include_text=False) for c in self.children],
        }
        if include_text and self.is_root:
            data["_text"] = self._text
        return data

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
        block.start = data.get("start", 0)
        block.end = data.get("end", 0)
        block._text = data.get("_text", "")
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
    
    
    def commit(self, content: str | None = None) -> Block | None:
        post_block = self.mutator.commit(self, content)
        return post_block
        
    
    def transform(self) -> Block:
        from .transform import transform
        return transform(self)
    
    def render(self) -> str:
        tran_block = self.transform()
        return tran_block._text
    
    def print(self) -> None:
        print(self.render())

    # =========================================================================
    # Debug
    # =========================================================================

    def debug_tree(self, indent: int = 0) -> str:
        """Generate debug representation of block tree."""
        prefix = "  " * indent
        parts = [f"{prefix}Block[{self.start}:{self.end}]"]

        if self.tags:
            parts.append(f" tags={self.tags}")
        if self.role:
            parts.append(f" role={self.role!r}")
        if self.style:
            parts.append(f" style={self.style}")

        content_preview = self.text[:30]
        if len(self.text) > 30:
            content_preview += "..."
        parts.append(f" text={content_preview!r}")

        if self.chunks:
            chunk_styles = [c.style for c in self.chunks if c.style]
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
        content_preview = self.text[:20] if self.text else ""
        if len(self.text) > 20:
            content_preview += "..."
        return f"Block[{self.start}:{self.end}]({content_preview!r}, children={len(self.children)})"


def _parse_style(style: str | list[str] | None) -> list[str]:
    """Parse style into list of style strings."""
    if style is None:
        return []
    if isinstance(style, str):
        return style.split() if " " in style else [style]
    return list(style)
