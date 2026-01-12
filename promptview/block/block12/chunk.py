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


@dataclass
class Chunk:
    """
    A chunk of text with metadata.

    Combines content text with ChunkMeta for frontend consumption.

    Attributes:
        content: The actual text content
        meta: ChunkMeta with position and metadata info
    """
    content: str
    meta: ChunkMeta

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

    @property
    def is_empty(self) -> bool:
        """True if chunk has no content."""
        return len(self.content) == 0

    @property
    def is_newline(self) -> bool:
        """True if chunk is just a newline."""
        return self.content == "\n"

    @property
    def is_line_end(self) -> bool:
        """True if chunk ends with a newline."""
        return self.content.endswith("\n") if self.content else False

    @property
    def is_whitespace(self) -> bool:
        """True if chunk is only whitespace."""
        return self.content.isspace() if self.content else True

    def isspace(self) -> bool:
        """Check if content is whitespace (method form for compatibility)."""
        return self.content.isspace() if self.content else True

    def split(self, position: int) -> tuple["Chunk", "Chunk"]:
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
            return Chunk(content="", meta=left_meta), self.copy()

        if position >= len(self.content):
            # Return full left, empty right
            right_meta = ChunkMeta(
                start=self.meta.end,
                end=self.meta.end,
                logprob=self.meta.logprob,
                style=self.meta.style,
            )
            return self.copy(), Chunk(content="", meta=right_meta)

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

        return Chunk(content=left_content, meta=left_meta), Chunk(content=right_content, meta=right_meta)

    def copy(self) -> Chunk:
        """Create a copy of this chunk."""
        return Chunk(
            content=self.content,
            meta=self.meta.copy(),
        )

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
    def model_load(cls, data: dict[str, Any]) -> Chunk:
        """Deserialize chunk from dictionary."""
        meta = ChunkMeta(
            start=data.get("start", 0),
            end=data.get("end", 0),
            logprob=data.get("logprob"),
            style=data.get("style"),
            id=data.get("id", _generate_id()),
        )
        return cls(
            content=data.get("content", ""),
            meta=meta,
        )

    @classmethod
    def from_meta(cls, meta: ChunkMeta, text: str) -> Chunk:
        """
        Create Chunk from ChunkMeta by extracting content from text.

        Args:
            meta: The chunk metadata with positions
            text: The full text to extract content from
        """
        content = text[meta.start:meta.end]
        return cls(content=content, meta=meta)

    @classmethod
    def from_content(
        cls,
        content: str,
        logprob: float | None = None,
        style: str | None = None,
    ) -> Chunk:
        """
        Create Chunk from content string.

        Positions are set to 0:len(content) - useful for incoming parser chunks
        where final positions aren't known yet.

        Args:
            content: The text content
            logprob: Optional log probability
            style: Optional style label
        """
        meta = ChunkMeta(
            start=0,
            end=len(content),
            logprob=logprob,
            style=style,
        )
        return cls(content=content, meta=meta)

    def __repr__(self) -> str:
        content_preview = self.content[:20] if len(self.content) <= 20 else self.content[:17] + "..."
        parts = [f"content={content_preview!r}"]
        if self.logprob is not None:
            parts.append(f"logprob={self.logprob:.3f}")
        if self.style:
            parts.append(f"style={self.style!r}")
        return f"Chunk({', '.join(parts)})"
