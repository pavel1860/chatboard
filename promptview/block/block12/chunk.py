"""
ChunkMeta - Lightweight metadata for text chunks.

ChunkMeta stores metadata (logprob, style) for a region of text within a block.
Positions are relative to the owning block's start position.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
