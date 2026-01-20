"""
Chunk and ChunkMeta - Text chunks with metadata.

ChunkMeta stores metadata (logprob, style) for a region of text within a block.
Positions are relative to the owning block's start position.

Chunk holds the actual content along with ChunkMeta, suitable for frontend consumption.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid4().hex[:8]


@dataclass
class ChunkMeta:
    """
    Metadata for a chunk of text within a block.

    Positions are RELATIVE to the owning block's start position.
    This means chunk positions don't need to shift when the block moves.

    Attributes:
        start: Start position relative to block.start (inclusive)
        end: End position relative to block.start (exclusive)
        logprob: LLM log probability if available
        style: Style label (e.g., "xml-tag", "content", "prefix")
        id: Unique identifier for this chunk

    Example:
        Block at position 100-150 in root text.
        Chunk with start=10, end=20 refers to root._text[110:120]
    """
    start: int
    end: int
    logprob: float | None = None
    style: str | None = None
    id: str = field(default_factory=_generate_id)

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def length(self) -> int:
        """Length of this chunk in characters."""
        return self.end - self.start

    @property
    def is_empty(self) -> bool:
        """True if chunk has zero length."""
        return self.start == self.end

    def shift(self, delta: int) -> None:
        """Shift positions by delta (in-place)."""
        self.start += delta
        self.end += delta

    def copy(self) -> ChunkMeta:
        """Create a copy of this chunk metadata."""
        return ChunkMeta(
            start=self.start,
            end=self.end,
            logprob=self.logprob,
            style=self.style,
            # New ID for copy
        )

    def contains(self, position: int) -> bool:
        """Check if relative position is within this chunk."""
        return self.start <= position < self.end

    def overlaps(self, start: int, end: int) -> bool:
        """Check if this chunk overlaps with the given range."""
        return self.start < end and start < self.end

    def __repr__(self) -> str:
        parts = [f"[{self.start}:{self.end}]"]
        if self.logprob is not None:
            parts.append(f"logprob={self.logprob:.3f}")
        if self.style:
            parts.append(f"style={self.style!r}")
        return f"ChunkMeta({', '.join(parts)})"


class BlockChunk:
    """
    A chunk of text with metadata.

    Combines content text with ChunkMeta for frontend consumption.

    Can be created simply with content and optional metadata:
        chunk = Chunk("Hello", logprob=0.7, style="xml")

    Or with an explicit ChunkMeta:
        chunk = Chunk("Hello", meta=existing_meta)

    Attributes:
        content: The actual text content
        meta: ChunkMeta with position and metadata info
    """

    def __init__(
        self,
        content: str,
        *,
        logprob: float | None = None,
        style: str | None = None,
        meta: ChunkMeta | None = None,
    ):
        """
        Create a Chunk.

        Args:
            content: The text content
            logprob: Optional log probability (ignored if meta is provided)
            style: Optional style label (ignored if meta is provided)
            meta: Optional ChunkMeta (if not provided, one is created with start=0, end=len(content))
        """
        self.content = content
        if meta is not None:
            self.meta = meta
        else:
            self.meta = ChunkMeta(
                start=0,
                end=len(content),
                logprob=logprob,
                style=style,
            )

    @property
    def start(self) -> int:
        """Start position (from meta)."""
        return self.meta.start

    @property
    def end(self) -> int:
        """End position (from meta)."""
        return self.meta.end

    @property
    def logprob(self) -> float | None:
        """Log probability (from meta)."""
        return self.meta.logprob

    @property
    def style(self) -> str | None:
        """Style label (from meta)."""
        return self.meta.style

    @property
    def id(self) -> str:
        """Unique ID (from meta)."""
        return self.meta.id

    @property
    def length(self) -> int:
        """Length of content."""
        return len(self.content)

    def is_empty(self) -> bool:
        """True if chunk has no content."""
        return len(self.content) == 0

    def is_newline(self) -> bool:
        """True if chunk is just a newline."""
        return self.content == "\n"

    def is_line_end(self) -> bool:
        """True if chunk ends with a newline."""
        return self.content.endswith("\n") if self.content else False

    def is_whitespace(self) -> bool:
        """True if chunk is only whitespace."""
        return self.content.isspace() if self.content else True

    def isspace(self) -> bool:
        """Check if content is whitespace (method form for compatibility)."""
        return self.content.isspace() and not self.content == "\n" if self.content else True
    
    def starts_with_tab(self) -> bool:
        """True if chunk starts with whitespace."""
        return self.content.startswith(('  ', '\t'))
    
    def split_tab(self) -> tuple["BlockChunk", "BlockChunk"]:
        """Split chunk at the first tab."""
        content = self.content.lstrip()
        position = len(self.content) - len(content)
        return self.split(position)
        
    def split(self, position: int) -> tuple["BlockChunk", "BlockChunk"]:
        """
        Split chunk at the given position (relative to chunk start).

        Returns two chunks: (left, right) where left contains content[:position]
        and right contains content[position:].

        Args:
            position: Position within content to split at

        Returns:
            Tuple of (left_chunk, right_chunk)
        """
        if position <= 0:
            # Return empty left, full right
            left_meta = ChunkMeta(
                start=self.meta.start,
                end=self.meta.start,
                logprob=self.meta.logprob,
                style=self.meta.style,
            )
            return BlockChunk("", meta=left_meta), self.copy()

        if position >= len(self.content):
            # Return full left, empty right
            right_meta = ChunkMeta(
                start=self.meta.end,
                end=self.meta.end,
                logprob=self.meta.logprob,
                style=self.meta.style,
            )
            return self.copy(), BlockChunk("", meta=right_meta)

        # Split in middle
        left_content = self.content[:position]
        right_content = self.content[position:]

        left_meta = ChunkMeta(
            start=self.meta.start,
            end=self.meta.start + position,
            logprob=self.meta.logprob,
            style=self.meta.style,
        )
        right_meta = ChunkMeta(
            start=self.meta.start + position,
            end=self.meta.end,
            logprob=self.meta.logprob,
            style=self.meta.style,
        )

        return BlockChunk(left_content, meta=left_meta), BlockChunk(right_content, meta=right_meta)

    def copy(self) -> BlockChunk:
        """Create a copy of this chunk."""
        return BlockChunk(self.content, meta=self.meta.copy())

    def model_dump(self) -> dict[str, Any]:
        """Serialize chunk to dictionary for frontend."""
        return {
            "id": self.meta.id,
            "content": self.content,
            "start": self.meta.start,
            "end": self.meta.end,
            "logprob": self.meta.logprob,
            "style": self.meta.style,
        }

    @classmethod
    def model_load(cls, data: dict[str, Any]) -> BlockChunk:
        """Deserialize chunk from dictionary."""
        meta = ChunkMeta(
            start=data.get("start", 0),
            end=data.get("end", 0),
            logprob=data.get("logprob"),
            style=data.get("style"),
            id=data.get("id", _generate_id()),
        )
        return cls(data.get("content", ""), meta=meta)

    @classmethod
    def from_meta(cls, meta: ChunkMeta, text: str) -> BlockChunk:
        """
        Create Chunk from ChunkMeta by extracting content from text.

        Args:
            meta: The chunk metadata with positions
            text: The full text to extract content from
        """
        content = text[meta.start:meta.end]
        return cls(content, meta=meta)

    def __repr__(self) -> str:
        content_preview = self.content[:20] if len(self.content) <= 20 else self.content[:17] + "..."
        parts = [f"content={content_preview!r}"]
        if self.logprob is not None:
            parts.append(f"logprob={self.logprob:.3f}")
        if self.style:
            parts.append(f"style={self.style!r}")
        return f"BlockChunk({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison based on content.

        Supports comparison with:
        - Another Chunk (compares content)
        - A string (compares to chunk's content)

        Usage:
            c1 == c2        # Compare two chunks
            c1 == "hello"   # Compare chunk to string
            "hello" == c1   # Also works
        """
        if isinstance(other, BlockChunk):
            return self.content == other.content
        elif isinstance(other, str):
            return self.content == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        """
        Inequality comparison based on content.

        Usage:
            c1 != c2        # Compare two chunks
            c1 != "hello"   # Compare chunk to string
        """
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        """
        Hash based on id (not content) to allow chunks in sets/dicts.

        Note: Chunks are mutable, so we hash by id rather than content.
        """
        return hash(self.meta.id)

    def __bool__(self) -> bool:
        """
        Boolean evaluation based on content.

        Empty chunks are falsy, non-empty chunks are truthy.

        Usage:
            if not chunk:
                print("empty")
            if chunk:
                print("has content")
        """
        return len(self.content) > 0




    
    
    
    
