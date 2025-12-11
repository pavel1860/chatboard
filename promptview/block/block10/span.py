"""
Span and VirtualBlockText - View layer for block10.

SpanAnchor represents a position within a chunk.
Span represents a contiguous region of text across chunks.
VirtualBlockText provides a view into BlockText via spans.

Design Principles:
- Spans reference chunks directly (not by index) for stability during edits
- Spans can be discontiguous (multiple spans in VirtualBlockText)
- VirtualBlockText is a view, not a copy - changes to BlockText are reflected
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .chunk import Chunk, BlockText


@dataclass
class SpanAnchor:
    """
    A position within a chunk.

    Anchors are the building blocks of spans. They reference a chunk
    directly (not by index) so they remain valid when chunks are
    inserted or removed elsewhere in the BlockText.

    Attributes:
        chunk: The chunk this anchor points into
        offset: Byte offset within the chunk (0 to len(chunk.content))

    Example:
        chunk = Chunk(content="Hello world")
        start = SpanAnchor(chunk, 0)      # Points to 'H'
        end = SpanAnchor(chunk, 5)        # Points to ' ' (after "Hello")
    """

    chunk: "Chunk"
    offset: int

    def __post_init__(self):
        """Validate offset is within bounds."""
        if self.offset < 0:
            raise ValueError(f"Offset {self.offset} cannot be negative")
        if self.chunk and self.offset > len(self.chunk.content):
            raise ValueError(
                f"Offset {self.offset} exceeds chunk length {len(self.chunk.content)}"
            )
        if self.chunk is None and self.offset != 0:
            raise ValueError(f"Offset {self.offset} cannot be non-zero when chunk is None")

    def copy(self) -> SpanAnchor:
        """Create a copy of this anchor (same chunk reference)."""
        return SpanAnchor(chunk=self.chunk, offset=self.offset)

    def __eq__(self, other: object) -> bool:
        """Anchors are equal if they point to same position."""
        if isinstance(other, SpanAnchor):
            return self.chunk == other.chunk and self.offset == other.offset
        return False

    def __lt__(self, other: SpanAnchor) -> bool:
        """
        Compare anchors by position in BlockText.

        Note: This only works correctly if both anchors are in the same BlockText.
        """
        if self.chunk == other.chunk:
            return self.offset < other.offset
        # Walk forward from self to see if we reach other
        current = self.chunk.next
        while current is not None:
            if current == other.chunk:
                return True
            current = current.next
        return False

    def advance(self, n: int = 1) -> SpanAnchor:
        """
        Create new anchor advanced by n bytes.

        May move to next chunk(s) if needed.

        Args:
            n: Number of bytes to advance

        Returns:
            New SpanAnchor at advanced position

        Raises:
            ValueError: If advancing past end of BlockText
        """
        chunk = self.chunk
        offset = self.offset + n

        while offset > len(chunk.content):
            if chunk.next is None:
                # Allow pointing to end of last chunk
                if offset == len(chunk.content):
                    break
                raise ValueError("Cannot advance past end of BlockText")
            offset -= len(chunk.content)
            chunk = chunk.next

        return SpanAnchor(chunk, offset)

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk.id,
            "offset": self.offset,
        }


@dataclass
class Span:
    """
    A contiguous region of text across one or more chunks.

    Spans are defined by start and end anchors. They can span multiple
    chunks but represent a contiguous region of text.

    Attributes:
        start: Anchor at start of span (inclusive)
        end: Anchor at end of span (exclusive)

    Example:
        text = BlockText([Chunk("Hello "), Chunk("world")])
        # Span covering "lo wo"
        span = Span(
            start=SpanAnchor(text.head, 3),      # "lo " from first chunk
            end=SpanAnchor(text.tail, 2)         # "wo" from second chunk
        )
        print(span.text())  # "lo wo"
    """

    start: SpanAnchor
    end: SpanAnchor

    def __post_init__(self):
        """Validate span is well-formed."""
        # Start must not be after end
        if self.start.chunk == self.end.chunk:
            if self.start.offset > self.end.offset:
                raise ValueError("Span start offset cannot be after end offset")

    @property
    def is_empty(self) -> bool:
        """Check if span covers zero bytes."""
        return self.start.chunk == self.end.chunk and self.start.offset == self.end.offset

    @property
    def is_single_chunk(self) -> bool:
        """Check if span is entirely within one chunk."""
        return self.start.chunk == self.end.chunk
    
    
    @classmethod
    def from_chunks(cls, chunks: list["Chunk"]) -> Span:
        if len(chunks) == 0:
            return cls(start=SpanAnchor(chunk=None, offset=0), end=SpanAnchor(chunk=None, offset=0))
        if len(chunks) == 1:
            return cls(start=SpanAnchor(chunk=chunks[0], offset=0), end=SpanAnchor(chunk=chunks[0], offset=len(chunks[0].content)))
        start = SpanAnchor(chunk=chunks[0], offset=0)
        end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
        return cls(start=start, end=end)

    def text(self) -> str:
        """
        Extract text content covered by this span.

        Returns:
            The text content as a string
        """
        if self.is_single_chunk:
            return self.start.chunk.content[self.start.offset : self.end.offset]

        # Multi-chunk span
        result = []

        # First chunk (from start offset to end)
        result.append(self.start.chunk.content[self.start.offset :])

        # Middle chunks (full content)
        current = self.start.chunk.next
        while current is not None and current != self.end.chunk:
            result.append(current.content)
            current = current.next

        # Last chunk (from start to end offset)
        if self.end.chunk != self.start.chunk:
            result.append(self.end.chunk.content[: self.end.offset])

        return "".join(result)

    def chunks(self) -> Iterator["Chunk"]:
        """
        Iterate over chunks covered by this span.

        Yields:
            Each chunk that this span covers (may be partial coverage)
        """
        current = self.start.chunk
        while current is not None:
            yield current
            if current == self.end.chunk:
                break
            current = current.next

    def length(self) -> int:
        """
        Calculate the length of text covered by this span.

        Returns:
            Number of bytes in the span
        """
        if self.is_single_chunk:
            return self.end.offset - self.start.offset

        total = 0
        # First chunk
        total += len(self.start.chunk.content) - self.start.offset

        # Middle chunks
        current = self.start.chunk.next
        while current is not None and current != self.end.chunk:
            total += len(current.content)
            current = current.next

        # Last chunk
        if self.end.chunk != self.start.chunk:
            total += self.end.offset

        return total

    def copy(self) -> Span:
        """Create a copy of this span (same chunk references)."""
        return Span(start=self.start.copy(), end=self.end.copy())

    def contains_chunk(self, chunk: "Chunk") -> bool:
        """Check if this span covers any part of the given chunk."""
        for c in self.chunks():
            if c == chunk:
                return True
        return False

    def adjust_for_split(
        self,
        original_chunk: "Chunk",
        split_offset: int,
        left_chunk: "Chunk",
        right_chunk: "Chunk",
    ) -> None:
        """
        Adjust span anchors after a chunk is split.

        Called when a chunk that this span references is split into two.
        Updates anchors to point to the appropriate new chunk.

        Args:
            original_chunk: The chunk that was split
            split_offset: The offset at which it was split
            left_chunk: The new left chunk (content before split)
            right_chunk: The new right chunk (content after split)
        """
        # Adjust start anchor
        if self.start.chunk == original_chunk:
            if self.start.offset < split_offset:
                self.start.chunk = left_chunk
                # offset stays the same
            else:
                self.start.chunk = right_chunk
                self.start.offset -= split_offset

        # Adjust end anchor
        if self.end.chunk == original_chunk:
            if self.end.offset <= split_offset:
                self.end.chunk = left_chunk
                # offset stays the same
            else:
                self.end.chunk = right_chunk
                self.end.offset -= split_offset

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "start": self.start.model_dump(),
            "end": self.end.model_dump(),
        }


class VirtualBlockText:
    """
    A view into BlockText via one or more spans.

    VirtualBlockText doesn't own any chunks - it's a view into a BlockText.
    It can represent:
    - A single contiguous region (one span)
    - Multiple discontiguous regions (multiple spans)

    The spans are ordered and represent the logical sequence of content.

    Attributes:
        source: The BlockText being viewed
        spans: List of spans defining the view

    Example:
        text = BlockText([Chunk("Hello "), Chunk("beautiful "), Chunk("world")])

        # View of just "beautiful "
        view = VirtualBlockText(text, [
            Span(
                start=SpanAnchor(text.head.next, 0),
                end=SpanAnchor(text.head.next, 10)
            )
        ])

        print(view.render())  # "beautiful "
    """

    def __init__(self, source: "BlockText", spans: list[Span] | None = None):
        """
        Initialize VirtualBlockText.

        Args:
            source: The BlockText to view into
            spans: List of spans defining the view (empty means no content)
        """
        self.source = source
        self.spans: list[Span] = spans or []

    @classmethod
    def from_full_text(cls, source: "BlockText") -> VirtualBlockText:
        """
        Create a view covering the entire BlockText.

        Args:
            source: The BlockText to view

        Returns:
            VirtualBlockText covering all content
        """
        if source.is_empty:
            return cls(source, [])

        span = Span(
            start=SpanAnchor(source.head, 0),
            end=SpanAnchor(source.tail, len(source.tail.content)),
        )
        return cls(source, [span])

    @property
    def is_empty(self) -> bool:
        """Check if view has no content."""
        return len(self.spans) == 0 or all(s.is_empty for s in self.spans)

    def render(self) -> str:
        """
        Render the viewed content as a string.

        Returns:
            Concatenated text from all spans
        """
        return "".join(span.text() for span in self.spans)

    def __str__(self) -> str:
        """String representation is the rendered content."""
        return self.render()

    def length(self) -> int:
        """
        Calculate total length of viewed content.

        Returns:
            Total bytes across all spans
        """
        return sum(span.length() for span in self.spans)

    def __len__(self) -> int:
        """Length is the total content length."""
        return self.length()

    def chunks(self) -> Iterator["Chunk"]:
        """
        Iterate over all chunks covered by this view.

        Note: A chunk may be yielded multiple times if multiple spans
        reference it, or partially if spans only cover part of it.

        Yields:
            Chunks covered by spans (may have duplicates)
        """
        for span in self.spans:
            yield from span.chunks()

    def unique_chunks(self) -> list["Chunk"]:
        """
        Get unique chunks covered by this view, in order.

        Returns:
            List of unique chunks
        """
        seen = set()
        result = []
        for chunk in self.chunks():
            if chunk.id not in seen:
                seen.add(chunk.id)
                result.append(chunk)
        return result

    def find_position(self, offset: int) -> tuple["Chunk", int]:
        """
        Find the chunk and offset for a logical position in this view.

        Args:
            offset: Logical byte offset from start of view

        Returns:
            Tuple of (chunk, offset_within_chunk)

        Raises:
            ValueError: If offset is out of bounds
        """
        if offset < 0:
            raise ValueError(f"Offset {offset} cannot be negative")

        remaining = offset
        for span in self.spans:
            span_len = span.length()
            if remaining < span_len:
                # Position is within this span
                return self._find_position_in_span(span, remaining)
            remaining -= span_len

        raise ValueError(f"Offset {offset} exceeds view length {self.length()}")

    def _find_position_in_span(self, span: Span, offset: int) -> tuple["Chunk", int]:
        """Find position within a single span."""
        current_chunk = span.start.chunk
        current_offset = span.start.offset + offset

        while current_offset > len(current_chunk.content):
            if current_chunk == span.end.chunk:
                raise ValueError("Offset exceeds span")
            current_offset -= len(current_chunk.content)
            current_chunk = current_chunk.next
            if current_chunk is None:
                raise ValueError("Reached end of BlockText")

        return current_chunk, current_offset

    def append_span(self, span: Span) -> None:
        """
        Append a span to this view.

        Args:
            span: Span to append
        """
        self.spans.append(span)

    def prepend_span(self, span: Span) -> None:
        """
        Prepend a span to this view.

        Args:
            span: Span to prepend
        """
        self.spans.insert(0, span)

    def adjust_for_split(
        self,
        original_chunk: "Chunk",
        split_offset: int,
        left_chunk: "Chunk",
        right_chunk: "Chunk",
    ) -> None:
        """
        Adjust all spans after a chunk split.

        Args:
            original_chunk: The chunk that was split
            split_offset: The offset at which it was split
            left_chunk: The new left chunk
            right_chunk: The new right chunk
        """
        for span in self.spans:
            span.adjust_for_split(original_chunk, split_offset, left_chunk, right_chunk)

    def copy(self) -> VirtualBlockText:
        """
        Create a copy of this view (same source, copied spans).

        Returns:
            New VirtualBlockText with copied spans
        """
        return VirtualBlockText(
            source=self.source,
            spans=[span.copy() for span in self.spans],
        )

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "spans": [span.model_dump() for span in self.spans],
        }

    def __repr__(self) -> str:
        content = self.render()
        preview = content[:50] + "..." if len(content) > 50 else content
        return f'VirtualBlockText("{preview}", spans={len(self.spans)})'
