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
class BlockChunk:
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
    prev: BlockChunk | None = field(default=None, repr=False, compare=False)
    next: BlockChunk | None = field(default=None, repr=False, compare=False)

    # Back-reference to owning BlockText (set when added)
    _owner: BlockText | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        """Return length of content."""
        return len(self.content)

    def __eq__(self, other: object) -> bool:
        """Chunks are equal if they have the same ID."""
        if isinstance(other, BlockChunk):
            return self.id == other.id
        return False

    def __hash__(self) -> int:
        """Hash by ID for use in sets/dicts."""
        return hash(self.id)

    def copy(self, include_links: bool = False) -> BlockChunk:
        """
        Create a copy of this chunk.

        Args:
            include_links: If True, copy prev/next references (usually False)

        Returns:
            New Chunk with same content and logprob, new ID
        """
        return BlockChunk(
            content=self.content,
            logprob=self.logprob,
            prev=self.prev if include_links else None,
            next=self.next if include_links else None,
        )

    def split(self, offset: int) -> tuple[BlockChunk, BlockChunk]:
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

        left = BlockChunk(
            content=self.content[:offset],
            logprob=self.logprob,  # Logprob is preserved on both (debatable)
        )
        right = BlockChunk(
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
    def model_validate(cls, data: dict[str, Any]) -> BlockChunk:
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

    def __init__(self, chunks: list[BlockChunk] | None = None):
        """
        Initialize BlockText, optionally with initial chunks.

        Args:
            chunks: Optional list of chunks to add in order
        """
        self.head: BlockChunk | None = None
        self.tail: BlockChunk | None = None
        self._by_id: dict[str, BlockChunk] = {}
        self._length: int = 0

        if chunks:
            for chunk in chunks:
                self.append(chunk)

    def __len__(self) -> int:
        """Return number of chunks."""
        return self._length

    def __iter__(self) -> Iterator[BlockChunk]:
        """Iterate over chunks from head to tail."""
        current = self.head
        while current is not None:
            yield current
            current = current.next

    def __reversed__(self) -> Iterator[BlockChunk]:
        """Iterate over chunks from tail to head."""
        current = self.tail
        while current is not None:
            yield current
            current = current.prev

    def __contains__(self, chunk: BlockChunk) -> bool:
        """Check if chunk is in this BlockText."""
        return chunk.id in self._by_id

    def __bool__(self) -> bool:
        """BlockText is truthy if it has any chunks."""
        return self._length > 0

    @property
    def is_empty(self) -> bool:
        """Check if BlockText has no chunks."""
        return self._length == 0

    def get_by_id(self, chunk_id: str) -> BlockChunk | None:
        """
        Get chunk by ID.

        Args:
            chunk_id: The chunk's unique ID

        Returns:
            The chunk, or None if not found
        """
        return self._by_id.get(chunk_id)

    def append(self, chunk: BlockChunk) -> BlockChunk:
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
    
    
    def extend(self, chunks: list[BlockChunk], after: BlockChunk | None = None):
        result = []
        if after is None:
            for chunk in chunks:
                result.append(self.append(chunk))
        else:
            for chunk in chunks:
                after = self.insert_after(after, chunk)
                result.append(after)
        return result
    
    
    def left_extend(self, chunks: list[BlockChunk], before: BlockChunk | None = None):
        result = []
        if before is None:
            for chunk in reversed(chunks):
                result.append(self.prepend(chunk))
        else:
            for chunk in reversed(chunks):
                before = self.insert_before(before, chunk)
                result.append(before)
        return list(reversed(result))

    def prepend(self, chunk: BlockChunk) -> BlockChunk:
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
    
    
    def insert_chunks_after(self, after: BlockChunk, chunks: list[BlockChunk]):
        result = []
        for chunk in chunks:
            result.append(self.insert_after(after, chunk))
            after = chunk
        return result
    
    def insert_chunks_before(self, before: BlockChunk, chunks: list[BlockChunk]):
        result = []
        for chunk in reversed(chunks):
            before = self.insert_before(before, chunk)
            result.append(before)
        return list(reversed(result))

    def extend_block_text(self, other: "BlockText", after: BlockChunk | None = None, copy: bool = True) -> list[BlockChunk]:
        """
        Extend this BlockText with chunks from another BlockText.

        Args:
            other: Source BlockText to extend from
            after: Insert after this chunk (None = append to end)
            copy: If True, copy chunks (other unchanged). If False, move chunks (other emptied).

        Returns:
            List of inserted chunks (new chunks if copy=True, moved chunks if copy=False)

        Example:
            text1 = BlockText([Chunk("Hello ")])
            text2 = BlockText([Chunk("world")])

            # Copy (text2 unchanged)
            text1.extend_block_text(text2, copy=True)

            # Move (text2 emptied)
            text1.extend_block_text(text2, copy=False)
        """
        if other.is_empty:
            return []

        if copy:
            chunks = [chunk.copy() for chunk in other]
            return self.extend(chunks, after=after)

        # Move mode: reconnect pointers directly (O(1) for linking)
        other_head = other.head
        other_tail = other.tail
        result = list(other)  # Snapshot for return value

        # Update ownership and index
        for chunk in result:
            chunk._owner = self
            self._by_id[chunk.id] = chunk

        # Connect to self
        if after is None:
            # Append to end
            if self.tail is not None:
                self.tail.next = other_head
                other_head.prev = self.tail
            else:
                self.head = other_head
            self.tail = other_tail
        else:
            # Insert after specific chunk
            next_chunk = after.next
            after.next = other_head
            other_head.prev = after
            other_tail.next = next_chunk
            if next_chunk is not None:
                next_chunk.prev = other_tail
            else:
                self.tail = other_tail

        self._length += other._length

        # Clear the source BlockText
        other.head = None
        other.tail = None
        other._by_id.clear()
        other._length = 0

        return result

    def left_extend_block_text(self, other: "BlockText", before: BlockChunk | None = None, copy: bool = True) -> list[BlockChunk]:
        """
        Prepend chunks from another BlockText to this BlockText.

        Args:
            other: Source BlockText to extend from
            before: Insert before this chunk (None = prepend to beginning)
            copy: If True, copy chunks (other unchanged). If False, move chunks (other emptied).

        Returns:
            List of inserted chunks in order

        Example:
            text1 = BlockText([Chunk("world")])
            text2 = BlockText([Chunk("Hello ")])

            # Prepend text2 to text1
            text1.left_extend_block_text(text2)
            # Result: "Hello world"
        """
        if other.is_empty:
            return []

        if copy:
            chunks = [chunk.copy() for chunk in other]
            return self.left_extend(chunks, before=before)

        # Move mode: reconnect pointers directly
        other_head = other.head
        other_tail = other.tail
        result = list(other)

        # Update ownership and index
        for chunk in result:
            chunk._owner = self
            self._by_id[chunk.id] = chunk

        # Connect to self
        if before is None:
            # Prepend to beginning
            if self.head is not None:
                self.head.prev = other_tail
                other_tail.next = self.head
            else:
                self.tail = other_tail
            self.head = other_head
        else:
            # Insert before specific chunk
            prev_chunk = before.prev
            before.prev = other_tail
            other_tail.next = before
            other_head.prev = prev_chunk
            if prev_chunk is not None:
                prev_chunk.next = other_head
            else:
                self.head = other_head

        self._length += other._length

        # Clear the source BlockText
        other.head = None
        other.tail = None
        other._by_id.clear()
        other._length = 0

        return result

    def replace(
        self,
        start: BlockChunk,
        end: BlockChunk,
        new_chunks: list[BlockChunk] | None = None,
    ) -> list[BlockChunk]:
        """
        Replace a range of chunks (inclusive) with new chunks.

        Removes all chunks from start to end (inclusive) and inserts
        new_chunks in their place. Returns the removed chunks.

        Args:
            start: First chunk to replace (inclusive)
            end: Last chunk to replace (inclusive)
            new_chunks: Chunks to insert in place of removed range (None or [] to just delete)

        Returns:
            List of removed chunks (detached from this BlockText)

        Raises:
            ValueError: If start or end is not in this BlockText
            ValueError: If end comes before start in the linked list

        Example:
            bt = BlockText([Chunk("a"), Chunk("b"), Chunk("c"), Chunk("d")])
            # Replace "b" and "c" with "X"
            removed = bt.replace(bt.head.next, bt.tail.prev, [Chunk("X")])
            # bt.text() == "aXd"
            # removed == [Chunk("b"), Chunk("c")]
        """
        if start._owner is not self:
            raise ValueError(f"Chunk {start.id} is not in this BlockText")
        if end._owner is not self:
            raise ValueError(f"Chunk {end.id} is not in this BlockText")

        # Get neighbors before modifying
        prev_chunk = start.prev
        next_chunk = end.next

        # Collect chunks to remove (and verify end comes after start)
        removed = []
        current = start
        while current is not None:
            removed.append(current)
            if current is end:
                break
            current = current.next
        else:
            # We reached the end of the list without finding 'end'
            raise ValueError(f"Chunk {end.id} does not come after {start.id} in the linked list")

        # Detach removed chunks
        for chunk in removed:
            chunk._owner = None
            chunk.prev = None
            chunk.next = None
            del self._by_id[chunk.id]
            self._length -= 1

        # Update head/tail if needed
        if start is self.head:
            self.head = next_chunk
        if end is self.tail:
            self.tail = prev_chunk

        # Connect the gap (before inserting new chunks)
        if prev_chunk is not None:
            prev_chunk.next = next_chunk
        if next_chunk is not None:
            next_chunk.prev = prev_chunk

        # Insert new chunks
        if new_chunks:
            if prev_chunk is not None:
                self.extend(new_chunks, after=prev_chunk)
            elif next_chunk is not None:
                self.left_extend(new_chunks, before=next_chunk)
            else:
                # BlockText is now empty, just extend
                self.extend(new_chunks)

        return removed

    def replace_block_text(
        self,
        start: BlockChunk,
        end: BlockChunk,
        other: "BlockText",
        copy: bool = True,
    ) -> tuple[list[BlockChunk], list[BlockChunk]]:
        """
        Replace a range of chunks with chunks from another BlockText.

        Combines replace() with extend_block_text() for convenience.

        Args:
            start: First chunk to replace (inclusive)
            end: Last chunk to replace (inclusive)
            other: Source BlockText for replacement chunks
            copy: If True, copy chunks from other. If False, move them.

        Returns:
            Tuple of (removed_chunks, inserted_chunks)

        Example:
            bt1 = BlockText([Chunk("a"), Chunk("old"), Chunk("d")])
            bt2 = BlockText([Chunk("new")])
            removed, inserted = bt1.replace_block_text(bt1.head.next, bt1.head.next, bt2)
            # bt1.text() == "anewd"
            # removed == [Chunk("old")]
            # inserted == [Chunk("new")]
        """
        if start._owner is not self:
            raise ValueError(f"Chunk {start.id} is not in this BlockText")
        if end._owner is not self:
            raise ValueError(f"Chunk {end.id} is not in this BlockText")

        # Get neighbors before modifying
        prev_chunk = start.prev
        next_chunk = end.next

        # Collect and remove chunks
        removed = []
        current = start
        while current is not None:
            removed.append(current)
            if current is end:
                break
            current = current.next
        else:
            raise ValueError(f"Chunk {end.id} does not come after {start.id} in the linked list")

        # Detach removed chunks
        for chunk in removed:
            chunk._owner = None
            chunk.prev = None
            chunk.next = None
            del self._by_id[chunk.id]
            self._length -= 1

        # Update head/tail
        if start is self.head:
            self.head = next_chunk
        if end is self.tail:
            self.tail = prev_chunk

        # Connect the gap
        if prev_chunk is not None:
            prev_chunk.next = next_chunk
        if next_chunk is not None:
            next_chunk.prev = prev_chunk

        # Insert from other BlockText
        inserted = []
        if not other.is_empty:
            if prev_chunk is not None:
                inserted = self.extend_block_text(other, after=prev_chunk, copy=copy)
            elif next_chunk is not None:
                inserted = self.left_extend_block_text(other, before=next_chunk, copy=copy)
            else:
                inserted = self.extend_block_text(other, copy=copy)

        return removed, inserted


    def insert_after(self, after: BlockChunk, chunk: BlockChunk) -> BlockChunk:
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

    def insert_before(self, before: BlockChunk, chunk: BlockChunk) -> BlockChunk:
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

    def remove(self, chunk: BlockChunk) -> BlockChunk:
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

    def split_chunk(self, chunk: BlockChunk, offset: int) -> tuple[BlockChunk, BlockChunk]:
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
        start: BlockChunk | None = None,
        end: BlockChunk | None = None,
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

    def chunks_list(self) -> list[BlockChunk]:
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
        chunks = [BlockChunk.model_validate(c) for c in data.get("chunks", [])]
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
