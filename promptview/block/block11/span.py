from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator
from uuid import uuid4

if TYPE_CHECKING:
    from .block_text import BlockText


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid4().hex[:8]


@dataclass
class Chunk:
    """
    Atomic text unit with optional metadata.

    Chunks are the smallest unit of text storage. They can carry
    metadata like logprobs from LLM responses.
    """
    content: str
    id: str = field(default_factory=_generate_id)
    logprob: float | None = None

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


    def __len__(self) -> int:
        """Length of content."""
        return len(self.content)

    def copy(self) -> Chunk:
        """Create a copy of this chunk."""
        return Chunk(
            content=self.content,
            logprob=self.logprob,
        )
        
    def split(self, offset: int) -> tuple[Chunk, Chunk]:
        """Split a chunk into two chunks at the given start and end positions."""
        left = self.content[:offset]
        right = self.content[offset:]        
        lchunk = Chunk(content=left, logprob=self.logprob)
        rchunk = Chunk(content=right, logprob=self.logprob)
        return lchunk, rchunk

    def __repr__(self) -> str:
        return f"Chunk({self.content!r})"


@dataclass
class Span:
    """
    A block's head: prefix + content + postfix.

    Spans are the atomic unit of storage. They form a linked list
    within BlockText for ordering.

    Ownership: BlockText owns Spans (Span.owner = BlockText).
    Blocks reference Spans but don't own them.
    """
    prefix: list[Chunk] = field(default_factory=list)
    content: list[Chunk] = field(default_factory=list)
    postfix: list[Chunk] = field(default_factory=list)

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

    def has_end_of_line(self) -> bool:
        """True if postfix ends with newline."""
        if self.postfix:
            return any(c.is_line_end for c in self.postfix)
            # return self.postfix[-1].is_line_end
        # if self.content:
        #     return self.content[-1].is_line_end
        return False

    # --- Chunk Iteration ---

    def chunks(self) -> Iterator[Chunk]:
        """Iterate over all chunks in order: prefix, content, postfix."""
        yield from self.prefix
        yield from self.content
        yield from self.postfix

    def __len__(self) -> int:
        """Total number of chunks."""
        return len(self.prefix) + len(self.content) + len(self.postfix)

    # --- Mutation: Content ---

    def append_content(self, chunks: list[Chunk]) -> Span:
        """Append chunks to content."""
        self.content.extend(chunks)
        return self

    def prepend_content(self, chunks: list[Chunk]) -> Span:
        """Prepend chunks to content."""
        self.content = chunks + self.content
        return self

    # --- Mutation: Prefix ---

    def append_prefix(self, chunks: list[Chunk]) -> Span:
        """Append chunks to prefix."""
        self.prefix.extend(chunks)
        return self

    def prepend_prefix(self, chunks: list[Chunk]) -> Span:
        """Prepend chunks to prefix."""
        self.prefix = chunks + self.prefix
        return self

    # --- Mutation: Postfix ---

    def append_postfix(self, chunks: list[Chunk]) -> Span:
        """Append chunks to postfix."""
        self.postfix.extend(chunks)
        return self

    def prepend_postfix(self, chunks: list[Chunk]) -> Span:
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
