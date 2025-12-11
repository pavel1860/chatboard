"""
Chunk and BlockText - Core storage layer for block10.

Chunk is the atomic unit of text, potentially from LLM output with logprob metadata.
BlockText is a linked-list container that owns all chunks and provides O(1) insertion.

Design Principles:
- Chunks are linked for O(1) insertion anywhere
- Chunk content is mutable for editing
- BlockText is the single source of truth for chunk storage
- Spans reference chunks directly (not by index) for stability during edits
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Any, TYPE_CHECKING
from uuid import uuid4
if TYPE_CHECKING:
    from .span import Span, SpanAnchor


def _generate_id() -> str:
    """Generate a short unique ID for chunks."""
    return uuid4().hex[:8]



@dataclass
class Chunk:
    """
    Atomic unit of text in the block system.

    Chunks form a doubly-linked list within a BlockText. Each chunk has:
    - Unique ID for stable references
    - Text content (mutable for editing)
    - Optional logprob from LLM
    - Prev/next pointers for linked list

    Example:
        chunk = Chunk(content="Hello")
        chunk.content += " world"  # Mutable
    """

    content: str = ""
    logprob: float | None = None
    id: str = field(default_factory=_generate_id)

    # Linked list pointers (not part of equality/hash)
    prev: Chunk | None = field(default=None, repr=False, compare=False)
    next: Chunk | None = field(default=None, repr=False, compare=False)

    # Back-reference to owning BlockText (set when added)
    _owner: BlockText | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        """Return length of content."""
        return len(self.content)

    def __eq__(self, other: object) -> bool:
        """Chunks are equal if they have the same ID."""
        if isinstance(other, Chunk):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        """Hash by ID for use in sets/dicts."""
        return hash(self.id)

    def copy(self, include_links: bool = False) -> Chunk:
        """
        Create a copy of this chunk.

        Args:
            include_links: If True, copy prev/next references (usually False)

        Returns:
            New Chunk with same content and logprob, new ID
        """
        return Chunk(
            content=self.content,
            logprob=self.logprob,
            prev=self.prev if include_links else None,
            next=self.next if include_links else None,
        )

    def split(self, offset: int) -> tuple[Chunk, Chunk]:
        """
        Split this chunk at the given offset.

        Creates two new chunks. Does NOT modify the linked list -
        caller is responsible for updating links.

        Args:
            offset: Byte offset to split at (0 to len(content))

        Returns:
            Tuple of (left_chunk, right_chunk)

        Raises:
            ValueError: If offset is out of bounds

        Example:
            chunk = Chunk(content="Hello world")
            left, right = chunk.split(5)
            # left.content == "Hello"
            # right.content == " world"
        """
        if offset < 0 or offset > len(self.content):
            raise ValueError(f"Split offset {offset} out of bounds [0, {len(self.content)}]")

        left = Chunk(
            content=self.content[:offset],
            logprob=self.logprob,  # Logprob is preserved on both (debatable)
        )
        right = Chunk(
            content=self.content[offset:],
            logprob=self.logprob,
        )
        return left, right

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "logprob": self.logprob,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Chunk:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", _generate_id()),
            content=data.get("content", ""),
            logprob=data.get("logprob"),
        )
        
        
    @property
    def is_line_end(self) -> bool:
        return self.content.endswith("\n")



class BlockText:
    """
    Linked-list storage for chunks.

    BlockText owns all chunks and provides:
    - O(1) append to end
    - O(1) insert after any chunk
    - O(1) chunk lookup by ID
    - Sequential iteration
    - Split and merge operations

    Design:
    - Doubly-linked list for bidirectional traversal
    - ID index for fast lookup
    - No random access by position (use spans for that)

    Example:
        text = BlockText()
        text.append(Chunk("Hello "))
        text.append(Chunk("world"))

        for chunk in text:
            print(chunk.content)

        # Insert in middle
        hello = text.head
        text.insert_after(hello, Chunk("beautiful "))
    """

    def __init__(self, chunks: list[Chunk] | None = None):
        """
        Initialize BlockText, optionally with initial chunks.

        Args:
            chunks: Optional list of chunks to add in order
        """
        self.head: Chunk | None = None
        self.tail: Chunk | None = None
        self._by_id: dict[str, Chunk] = {}
        self._length: int = 0

        if chunks:
            for chunk in chunks:
                self.append(chunk)

    def __len__(self) -> int:
        """Return number of chunks."""
        return self._length

    def __iter__(self) -> Iterator[Chunk]:
        """Iterate over chunks from head to tail."""
        current = self.head
        while current is not None:
            yield current
            current = current.next

    def __reversed__(self) -> Iterator[Chunk]:
        """Iterate over chunks from tail to head."""
        current = self.tail
        while current is not None:
            yield current
            current = current.prev

    def __contains__(self, chunk: Chunk) -> bool:
        """Check if chunk is in this BlockText."""
        return chunk.id in self._by_id

    def __bool__(self) -> bool:
        """BlockText is truthy if it has any chunks."""
        return self._length > 0

    @property
    def is_empty(self) -> bool:
        """Check if BlockText has no chunks."""
        return self._length == 0

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """
        Get chunk by ID.

        Args:
            chunk_id: The chunk's unique ID

        Returns:
            The chunk, or None if not found
        """
        return self._by_id.get(chunk_id)

    def append(self, chunk: Chunk) -> Chunk:
        """
        Append chunk to the end.

        Args:
            chunk: Chunk to append

        Returns:
            The appended chunk

        Raises:
            ValueError: If chunk is already in a BlockText
        """
        if chunk._owner is not None:
            raise ValueError(f"Chunk {chunk.id} already belongs to a BlockText")

        chunk._owner = self
        chunk.prev = self.tail
        chunk.next = None

        if self.tail is not None:
            self.tail.next = chunk
        self.tail = chunk

        if self.head is None:
            self.head = chunk

        self._by_id[chunk.id] = chunk
        self._length += 1

        return chunk
    
    
    def extend(self, chunks: list[Chunk], after: Chunk | None = None):
        result = []
        if after is None:
            for chunk in chunks:
                result.append(self.append(chunk))
        else:
            for chunk in chunks:
                after = self.insert_after(after, chunk)
                result.append(after)
        return result
    
    
    def left_extend(self, chunks: list[Chunk], before: Chunk | None = None):
        result = []
        if before is None:
            for chunk in reversed(chunks):
                result.append(self.prepend(chunk))
        else:
            for chunk in reversed(chunks):
                before = self.insert_before(before, chunk)
                result.append(before)
        return list(reversed(result))

    def prepend(self, chunk: Chunk) -> Chunk:
        """
        Prepend chunk to the beginning.

        Args:
            chunk: Chunk to prepend

        Returns:
            The prepended chunk

        Raises:
            ValueError: If chunk is already in a BlockText
        """
        if chunk._owner is not None:
            raise ValueError(f"Chunk {chunk.id} already belongs to a BlockText")

        chunk._owner = self
        chunk.prev = None
        chunk.next = self.head

        if self.head is not None:
            self.head.prev = chunk
        self.head = chunk

        if self.tail is None:
            self.tail = chunk

        self._by_id[chunk.id] = chunk
        self._length += 1

        return chunk

    def insert_after(self, after: Chunk, chunk: Chunk) -> Chunk:
        """
        Insert chunk after another chunk.

        Args:
            after: Existing chunk to insert after
            chunk: New chunk to insert

        Returns:
            The inserted chunk

        Raises:
            ValueError: If 'after' is not in this BlockText
            ValueError: If chunk is already in a BlockText
        """
        if after._owner is not self:
            raise ValueError(f"Chunk {after.id} is not in this BlockText")
        if chunk._owner is not None:
            raise ValueError(f"Chunk {chunk.id} already belongs to a BlockText")

        chunk._owner = self
        chunk.prev = after
        chunk.next = after.next

        if after.next is not None:
            after.next.prev = chunk
        after.next = chunk

        if after is self.tail:
            self.tail = chunk

        self._by_id[chunk.id] = chunk
        self._length += 1

        return chunk

    def insert_before(self, before: Chunk, chunk: Chunk) -> Chunk:
        """
        Insert chunk before another chunk.

        Args:
            before: Existing chunk to insert before
            chunk: New chunk to insert

        Returns:
            The inserted chunk

        Raises:
            ValueError: If 'before' is not in this BlockText
            ValueError: If chunk is already in a BlockText
        """
        if before._owner is not self:
            raise ValueError(f"Chunk {before.id} is not in this BlockText")
        if chunk._owner is not None:
            raise ValueError(f"Chunk {chunk.id} already belongs to a BlockText")

        chunk._owner = self
        chunk.prev = before.prev
        chunk.next = before

        if before.prev is not None:
            before.prev.next = chunk
        before.prev = chunk

        if before is self.head:
            self.head = chunk

        self._by_id[chunk.id] = chunk
        self._length += 1

        return chunk

    def remove(self, chunk: Chunk) -> Chunk:
        """
        Remove chunk from BlockText.

        Args:
            chunk: Chunk to remove

        Returns:
            The removed chunk

        Raises:
            ValueError: If chunk is not in this BlockText
        """
        if chunk._owner is not self:
            raise ValueError(f"Chunk {chunk.id} is not in this BlockText")

        # Update links
        if chunk.prev is not None:
            chunk.prev.next = chunk.next
        if chunk.next is not None:
            chunk.next.prev = chunk.prev

        # Update head/tail
        if chunk is self.head:
            self.head = chunk.next
        if chunk is self.tail:
            self.tail = chunk.prev

        # Clean up chunk
        chunk.prev = None
        chunk.next = None
        chunk._owner = None

        del self._by_id[chunk.id]
        self._length -= 1

        return chunk

    def split_chunk(self, chunk: Chunk, offset: int) -> tuple[Chunk, Chunk]:
        """
        Split a chunk at the given offset, replacing it with two chunks.

        This modifies the linked list in place: the original chunk is removed
        and replaced with left and right chunks.

        Args:
            chunk: Chunk to split
            offset: Byte offset within chunk to split at

        Returns:
            Tuple of (left_chunk, right_chunk)

        Raises:
            ValueError: If chunk is not in this BlockText
            ValueError: If offset is out of bounds
        """
        if chunk._owner is not self:
            raise ValueError(f"Chunk {chunk.id} is not in this BlockText")

        # Create split chunks
        left, right = chunk.split(offset)

        # Get neighbors before removing
        prev_chunk = chunk.prev
        next_chunk = chunk.next

        # Remove original (clears links)
        self.remove(chunk)

        # Insert left
        if prev_chunk is not None:
            self.insert_after(prev_chunk, left)
        else:
            self.prepend(left)

        # Insert right after left
        self.insert_after(left, right)

        return left, right

    def text(self) -> str:
        """
        Get full text content by concatenating all chunks.

        Returns:
            Concatenated content of all chunks
        """
        return "".join(chunk.content for chunk in self)

    def fork(
        self,
        start: Chunk | None = None,
        end: Chunk | None = None,
    ) -> BlockText:
        """
        Create an independent copy of this BlockText.

        All chunks are copied (new IDs), allowing independent editing.
        Optionally specify a range of chunks to copy.

        Args:
            start: First chunk to copy (inclusive). If None, starts from head.
            end: Last chunk to copy (inclusive). If None, copies to tail.

        Returns:
            New BlockText with copied chunks

        Example:
            # Copy entire BlockText
            new_text = text.fork()

            # Copy from chunk A to chunk B (inclusive)
            new_text = text.fork(start=chunk_a, end=chunk_b)

            # Copy from start to chunk B
            new_text = text.fork(end=chunk_b)

            # Copy from chunk A to end
            new_text = text.fork(start=chunk_a)
        """
        new_text = BlockText()

        # Determine start chunk
        current = start if start is not None else self.head

        # Iterate and copy chunks until we reach end (inclusive)
        while current is not None:
            new_text.append(current.copy())

            # Stop after copying the end chunk
            if current is end:
                break

            current = current.next

        return new_text

    def chunks_list(self) -> list[Chunk]:
        """
        Get chunks as a list (for serialization).

        Returns:
            List of chunks in order
        """
        return list(self)

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunks": [chunk.model_dump() for chunk in self],
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> BlockText:
        """Deserialize from dictionary."""
        chunks = [Chunk.model_validate(c) for c in data.get("chunks", [])]
        return cls(chunks)

    def __repr__(self) -> str:
        chunks_preview = []
        for i, chunk in enumerate(self):
            if i >= 3:
                chunks_preview.append("...")
                break
            content = chunk.content[:20] + "..." if len(chunk.content) > 20 else chunk.content
            chunks_preview.append(f'"{content}"')
        return f"BlockText([{', '.join(chunks_preview)}], len={self._length})"
    
    
    def print_chunk_linage(self):
        str_linage = "\n".join([f"{chunk.prev.id if chunk.prev else 'None'} -> [{chunk.id}] -> {chunk.next.id if chunk.next else 'None'} : {chunk.content}" for chunk in self])
        print(str_linage)
        # return str_linage
