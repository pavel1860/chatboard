from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from .block_text import BlockText


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid4().hex[:8]


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


def split_chunks(chunks: list[BlockChunk], sep: str) -> tuple[list[BlockChunk], list[BlockChunk], list[BlockChunk]]:
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


@dataclass
class BlockChunk:
    """
    Atomic text unit with optional metadata.

    Chunks are the smallest unit of text storage. They can carry
    metadata like logprobs from LLM responses.
    """
    content: str
    is_text: bool = False
    id: str = field(default_factory=_generate_id)
    logprob: float | None = None
    
    
    
    def __post_init__(self):
        self.is_text = not (self.is_line_end or self.isspace())
        
    @property
    def is_line_end(self) -> bool:
        """True if content ends with newline."""
        return "\n" in self.content
    
    def isspace(self) -> bool:
        return self.content.isspace()
    
    def isalpha(self) -> bool:
        return self.content.isalpha()
    
    def isdigit(self) -> bool:
        return self.content.isdigit()
    
    def isalnum(self) -> bool:
        return self.content.isalnum()
    
    
    def __contains__(self, item: str) -> bool:
        return item in self.content


    def __len__(self) -> int:
        """Length of content."""
        return len(self.content)

    def copy(self) -> BlockChunk:
        """Create a copy of this chunk."""
        return BlockChunk(
            content=self.content,
            logprob=self.logprob,
        )
        
    def split(self, offset: int) -> tuple[BlockChunk, BlockChunk]:
        """Split a chunk into two chunks at the given start and end positions."""
        left = self.content[:offset]
        right = self.content[offset:]        
        lchunk = BlockChunk(content=left, logprob=self.logprob)
        rchunk = BlockChunk(content=right, logprob=self.logprob)
        return lchunk, rchunk

    def __repr__(self) -> str:
        return f"Chunk({self.content!r})"

    def model_dump(self, *, exclude_none: bool = True) -> dict:
        """Serialize chunk to JSON-compatible dict."""
        result = {"content": self.content}
        if self.logprob is not None or not exclude_none:
            result["logprob"] = self.logprob
        return result

    @classmethod
    def model_validate(cls, data: dict) -> "BlockChunk":
        """Deserialize dict to BlockChunk."""
        return cls(
            content=data["content"],
            logprob=data.get("logprob"),
        )


@dataclass
class Span:
    """
    A block's head: prefix + content + postfix.

    Spans are the atomic unit of storage. They form a linked list
    within BlockText for ordering.

    Ownership: BlockText owns Spans (Span.owner = BlockText).
    Blocks reference Spans but don't own them.
    """
    prefix: list[BlockChunk] = field(default_factory=list)
    content: list[BlockChunk] = field(default_factory=list)
    postfix: list[BlockChunk] = field(default_factory=list)

    # Linked list pointers (managed by BlockText)
    prev: Span | None = field(default=None, repr=False)
    next: Span | None = field(default=None, repr=False)

    # Owner reference (BlockText that owns this span)
    owner: BlockText | None = field(default=None, repr=False)

    # Unique identifier
    id: str = field(default_factory=_generate_id)

    # --- Text Access ---

    @property
    def prefix_text(self) -> str:
        """Get prefix as string."""
        return "".join(c.content for c in self.prefix)

    @property
    def content_text(self) -> str:
        """Get content as string."""
        return "".join(c.content for c in self.content)

    @property
    def postfix_text(self) -> str:
        """Get postfix as string."""
        return "".join(c.content for c in self.postfix)

    @property
    def text(self) -> str:
        """Get full text: prefix + content + postfix."""
        return self.prefix_text + self.content_text + self.postfix_text

    # --- State ---

    @property
    def is_empty(self) -> bool:
        """True if all three lists are empty."""
        return not self.prefix and not self.content and not self.postfix

    def has_newline(self) -> bool:
        """True if postfix ends with newline."""
        if self.postfix:
            return any(c.is_line_end for c in self.postfix)
            # return self.postfix[-1].is_line_end
        # if self.content:
        #     return self.content[-1].is_line_end
        return False
    
    def add_newline(self) -> Span:
        """Add newline to postfix."""
        self.postfix.append(BlockChunk(content="\n"))
        return self

    # --- Chunk Iteration ---

    def chunks(self) -> Iterator[BlockChunk]:
        """Iterate over all chunks in order: prefix, content, postfix."""
        yield from self.prefix
        yield from self.content
        yield from self.postfix

    def __len__(self) -> int:
        """Total number of chunks."""
        return len(self.prefix) + len(self.content) + len(self.postfix)

    # --- Mutation: Content ---

    def append_content(self, chunks: list[BlockChunk]) -> Span:
        """Append chunks to content."""
        self.content.extend(chunks)
        return self

    def prepend_content(self, chunks: list[BlockChunk]) -> Span:
        """Prepend chunks to content."""
        self.content = chunks + self.content
        return self

    # --- Mutation: Prefix ---

    def append_prefix(self, chunks: list[BlockChunk]) -> Span:
        """Append chunks to prefix."""
        self.prefix.extend(chunks)
        return self

    def prepend_prefix(self, chunks: list[BlockChunk]) -> Span:
        """Prepend chunks to prefix."""
        self.prefix = chunks + self.prefix
        return self

    # --- Mutation: Postfix ---

    def append_postfix(self, chunks: list[BlockChunk]) -> Span:
        """Append chunks to postfix."""
        self.postfix.extend(chunks)
        return self

    def prepend_postfix(self, chunks: list[BlockChunk]) -> Span:
        """Prepend chunks to postfix."""
        self.postfix = chunks + self.postfix
        return self

    # --- Copy ---

    def copy(self) -> Span:
        """
        Create a deep copy of this span.

        The copy has new chunk instances and no owner/linked list pointers.
        """
        return Span(
            prefix=[c.copy() for c in self.prefix],
            content=[c.copy() for c in self.content],
            postfix=[c.copy() for c in self.postfix],
        )

    # --- Debug ---

    def __repr__(self) -> str:
        prefix_str = f"prefix={self.prefix_text!r}, " if self.prefix else ""
        postfix_str = f", postfix={self.postfix_text!r}" if self.postfix else ""
        return f"Span({prefix_str}content={self.content_text!r}{postfix_str})"

    # --- Serialization ---

    def model_dump(
        self,
        *,
        include_chunks: bool = False,
        exclude_none: bool = True,
    ) -> dict:
        """
        Serialize span to JSON-compatible dict.

        Args:
            include_chunks: Include chunk-level detail with logprobs
            exclude_none: Exclude None/empty values

        Returns:
            Dict with prefix, content, postfix (as text or chunks)
        """
        result: dict = {
            "prefix": self.prefix_text,
            "content": self.content_text,
            "postfix": self.postfix_text,
        }

        if include_chunks:
            result["prefix_chunks"] = [c.model_dump(exclude_none=exclude_none) for c in self.prefix]
            result["content_chunks"] = [c.model_dump(exclude_none=exclude_none) for c in self.content]
            result["postfix_chunks"] = [c.model_dump(exclude_none=exclude_none) for c in self.postfix]

        return result

    @classmethod
    def model_validate(cls, data: dict) -> "Span":
        """
        Deserialize dict to Span.

        If chunks are present, uses them (preserving logprobs).
        Otherwise creates chunks from text.

        Args:
            data: Dict from model_dump() or JSON

        Returns:
            Reconstructed Span
        """
        # Check if we have chunk-level data
        if data.get("prefix_chunks") or data.get("content_chunks") or data.get("postfix_chunks"):
            prefix = [BlockChunk.model_validate(c) for c in data.get("prefix_chunks", [])]
            content = [BlockChunk.model_validate(c) for c in data.get("content_chunks", [])]
            postfix = [BlockChunk.model_validate(c) for c in data.get("postfix_chunks", [])]
        else:
            # Create simple chunks from text
            prefix = [BlockChunk(content=data["prefix"])] if data.get("prefix") else []
            content = [BlockChunk(content=data["content"])] if data.get("content") else []
            postfix = [BlockChunk(content=data["postfix"])] if data.get("postfix") else []

        return cls(prefix=prefix, content=content, postfix=postfix)
