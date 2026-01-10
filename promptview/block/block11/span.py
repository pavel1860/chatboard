from __future__ import annotations
from collections import UserList
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, TypedDict
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


def split_chunks(chunks: BlockChunkList, sep: str) -> tuple[BlockChunkList, BlockChunkList, BlockChunkList]:
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
        return BlockChunkList(chunks=chunks), BlockChunkList(chunks=[]), BlockChunkList(chunks=[])

    # Build full text
    full_text = "".join(c.content for c in chunks)

    # Find separator
    sep_idx = full_text.find(sep)
    if sep_idx == -1:
        return BlockChunkList(chunks=chunks), BlockChunkList(chunks=[]), BlockChunkList(chunks=[])

    sep_end_idx = sep_idx + len(sep)

    # Track position as we iterate
    pos = 0
    before: BlockChunkList = BlockChunkList(chunks=[])
    separator: BlockChunkList = BlockChunkList(chunks=[])
    after: BlockChunkList = BlockChunkList(chunks=[])

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

class SpanEvent(TypedDict):
    cmd: Literal["append_content", "prepend_content", "append_prefix", "prepend_prefix", "append_postfix", "prepend_postfix"]
    chunks: list[BlockChunk]
    
    
ContentStylesSet = set(["space", "alpha", "digit", None])


def sanitize_styles(styles: set[str] | str | None, add_default: bool = True) -> set[str]:
    if styles is not None:
        if isinstance(styles, str):
            styles = {styles}
    else:
        styles = set([])        
    if add_default:
        styles = styles | ContentStylesSet
    return styles

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
    style: str | None = None
    
    
    
    def __post_init__(self):
        self.is_text = not (self.is_line_end or self.isspace())
        if not self.style:
            if self.is_line_end:
                self.style = "newline"
            elif self.isspace():
                self.style = "space"
            elif self.isalpha():
                self.style = "alpha"
            elif self.isdigit():
                self.style = "digit"
        
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
    
    
    def lower(self) -> BlockChunk:
        return BlockChunk(content=self.content.lower(), logprob=self.logprob, style=self.style)
    
    def upper(self) -> BlockChunk:
        return BlockChunk(content=self.content.upper(), logprob=self.logprob, style=self.style)
    
    def title(self) -> BlockChunk:
        return BlockChunk(content=self.content.title(), logprob=self.logprob, style=self.style)
    
    def capitalize(self) -> BlockChunk:
        return BlockChunk(content=self.content.capitalize(), logprob=self.logprob, style=self.style)
    
    def swapcase(self) -> BlockChunk:
        return BlockChunk(content=self.content.swapcase(), logprob=self.logprob, style=self.style)
    
    def replace(self, old: str, new: str) -> BlockChunk:
        return BlockChunk(content=self.content.replace(old, new), logprob=self.logprob, style=self.style)
    
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
        
BlockSpanEvent = Literal["append_content", "prepend_content", "lower", "upper", "title", "capitalize", "swapcase", "snake_case"]

# @dataclass
# class BlockChunkList:
#     chunks: list[BlockChunk]
#     event: BlockSpanEvent | None = None
    
    
#     def __getitem__(self, index: int) -> BlockChunk:
#         return self.chunks[index]
    
#     def __len__(self) -> int:
#         return len(self.chunks)
    
#     def __iter__(self) -> Iterator[BlockChunk]:
#         return iter(self.chunks)
    
#     def __repr__(self) -> str:
#         return f"BlockChunkList({self.chunks})"
    
#     def __str__(self) -> str:
#         return f"BlockChunkList({self.chunks})"
    
#     def __hash__(self) -> int:
#         return hash(tuple(self.chunks))
    
#     def append(self, chunk: BlockChunk) -> BlockChunkList:
#         self.chunks.append(chunk)
#         return self
    
#     def prepend(self, chunk: BlockChunk) -> BlockChunkList:
#         self.chunks.insert(0, chunk)
#         return self
    
#     def extend(self, chunks: list[BlockChunk]) -> BlockChunkList:
#         self.chunks.extend(chunks)
#         return self
    
#     def extend_left(self, chunks: list[BlockChunk]) -> BlockChunkList:
#         self.chunks = chunks + self.chunks
#         return self
    
#     def extend_right(self, chunks: list[BlockChunk]) -> BlockChunkList:
#         self.chunks = self.chunks + chunks
#         return self
class BlockChunkList(UserList[BlockChunk]):
    
    def __init__(self, chunks: list[BlockChunk] | list[str] | BlockChunkList | None = None, event: BlockSpanEvent | None = None):
        if chunks is None:
            chunks = []
        elif isinstance(chunks, list):
            if not all(isinstance(c, str) or isinstance(c, BlockChunk) for c in chunks):
                raise ValueError("chunks must be a list of strings")
            chunks = [c if isinstance(c, BlockChunk) else BlockChunk(content=c) for c in chunks]
        elif isinstance(chunks, BlockChunkList):
            chunks = chunks.data
        super().__init__(chunks)
        self.event = event
        
        
    @property
    def text(self) -> str:
        return "".join(c.content for c in self)
        
    def contains(self, chunks: list[BlockChunk] | BlockChunkList | str) -> bool:
        if isinstance(chunks, BlockChunkList):
            return chunks_contain(self.data, chunks.data)
        elif isinstance(chunks, list):
            return chunks_contain(self.data, chunks)
        elif isinstance(chunks, str):
            return chunks_contain(self.data, chunks)
        

    def split(self, sep: str) -> tuple[BlockChunkList, BlockChunkList, BlockChunkList]:
        return split_chunks(self, sep)
    
    def split_prefix(self, sep: str, create_on_empty: bool = True) -> tuple[BlockChunkList, BlockChunkList]:
        before, separator, after = self.split(sep)
        if not separator:
            if create_on_empty:
                return BlockChunkList(chunks=[BlockChunk(content=sep)]), self
            return BlockChunkList(chunks=[]), self
        return before + separator, after
    
    def split_postfix(self, sep: str, create_on_empty: bool = True) -> tuple[BlockChunkList, BlockChunkList]:
        before, separator, after = self.split(sep)
        if not separator:
            if create_on_empty:
                return self, BlockChunkList(chunks=[BlockChunk(content=sep)])
            return self, BlockChunkList(chunks=[])
        return before, separator + after
    
    
    
    def filter(self, styles: set[str] | str | None = None) -> BlockChunkList:
        styles = sanitize_styles(styles, add_default=False)
        return BlockChunkList(chunks=[c for c in self if c.style in styles])
    
    
    def apply_style(self, style: str | None = None) -> BlockChunkList:
        if style:
            for c in self:
                c.style = style
        return self
    
    
    def lower(self) -> BlockChunkList:
        chunks = [c.lower() for c in self]
        return BlockChunkList(chunks=chunks, event="lower")
    
    def upper(self) -> BlockChunkList:
        chunks = [c.upper() for c in self]
        return BlockChunkList(chunks=chunks, event="upper")
    
    def title(self) -> BlockChunkList:
        chunks = [c.title() for c in self]
        return BlockChunkList(chunks=chunks, event="title")
    
    def capitalize(self) -> BlockChunkList:
        chunks = [c.capitalize() for c in self]
        return BlockChunkList(chunks=chunks, event="capitalize")
    
    def swapcase(self) -> BlockChunkList:
        chunks = [c.swapcase() for c in self]
        return BlockChunkList(chunks=chunks, event="swapcase")
    
    def snake_case(self) -> BlockChunkList:
        chunks = []
        for c in self:
            if c.style == "space":
                chunks.append(BlockChunk(content="_", logprob=c.logprob, style=None))
            else:
                chunks.append(c.lower().replace(" ", "_"))
        return BlockChunkList(chunks=chunks, event="snake_case")
                
@dataclass
class Span:
    """
    A block's head: prefix + content + postfix.

    Spans are the atomic unit of storage. They form a linked list
    within BlockText for ordering.

    Ownership: BlockText owns Spans (Span.owner = BlockText).
    Blocks reference Spans but don't own them.
    """    
    chunks: BlockChunkList = field(default_factory=BlockChunkList)    

    # Linked list pointers (managed by BlockText)
    prev: Span | None = field(default=None, repr=False)
    next: Span | None = field(default=None, repr=False)

    # Owner reference (BlockText that owns this span)
    owner: BlockText | None = field(default=None, repr=False)
    
    _last_event: SpanEvent | None = field(default=None, repr=False)

    # Unique identifier
    id: str = field(default_factory=_generate_id)

    # --- Text Access ---


    @property
    def text(self) -> str:
        """Get full text: prefix + content + postfix."""
        return self.chunks.text
    
    @property
    def content(self) -> BlockChunkList:
        chunks = []
        found_content = False
        for c in self.chunks:
            if c.style in ContentStylesSet:
                found_content = True
            if found_content:
                if c.style not in ContentStylesSet:
                    break
                chunks.append(c.copy())
        return BlockChunkList(chunks=chunks)
    
    @property
    def prefix(self) -> BlockChunkList:
        chunks = []
        for c in self.chunks:
            if c.style in ContentStylesSet:
                break
            chunks.append(c.copy())
        return BlockChunkList(chunks=chunks)
    
    
    # @property
    # def postfix(self) -> BlockChunkList:
    #     chunks = []
    #     for c in reversed(self.chunks):
    #         if c.style in ContentStylesSet:
    #             break
    #         chunks.append(c.copy())
    #     return BlockChunkList(chunks=chunks)
    @property
    def postfix(self) -> BlockChunkList:
        chunks = []
        found_content = False        
        found_postfix = False
        for c in self.chunks:
            if not found_postfix:
                if not found_content:
                    if c.style in ContentStylesSet:
                        found_content = True
                else:
                    if c.style not in ContentStylesSet:
                        found_postfix = True                    
            if found_postfix:
                chunks.append(c.copy())
                continue
        return BlockChunkList(chunks=chunks)

    # --- State ---

    @property
    def is_empty(self) -> bool:
        """True if all three lists are empty."""
        return not self.chunks

    def has_newline(self) -> bool:
        """True if postfix ends with newline."""        
        for c in reversed(self.chunks):
            if c.is_line_end:
                return True
            elif c.isspace():
                continue
            else:
                return False
        return False
    
    def add_newline(self) -> Span:
        """Add newline to postfix."""
        self.chunks.append(BlockChunk(content="\n", style="newline"))
        return self

    # --- Chunk Iteration ---



    def __len__(self) -> int:
        """Total number of chunks."""
        return len(self.chunks)

    # --- Mutation: Content ---

    def append(self, chunks: list[BlockChunk] | BlockChunkList, style: str | None = None) -> BlockChunkList:
        """Append chunks to content."""
        if style:
            for c in chunks:
                c.style = style
        self.chunks.extend(chunks)
        return BlockChunkList(chunks=chunks, event="append_content")

    def prepend(self, chunks: list[BlockChunk] | BlockChunkList, style: str | None = None) -> BlockChunkList:
        """Prepend chunks to content."""
        if style:
            for c in chunks:
                c.style = style
        self.chunks = chunks + self.chunks
        return BlockChunkList(chunks=chunks, event="prepend_content")
    
    
    def extract_content(self, styles: set[str] | str | None = None) -> Span:
        styles = sanitize_styles(styles)   
        chunks = self.chunks.filter(styles)
        return Span(chunks=chunks)
    
    
    def append_next(self, chunks: list[BlockChunk] | BlockChunkList | None = None, style: str | None = None) -> Span:
        if chunks is None:
            chunks = BlockChunkList(chunks=[])
        elif isinstance(chunks, list):
            chunks = BlockChunkList(chunks=chunks)
        chunks.event = "init"
        if style:
            for c in chunks:
                c.style = style
        new_span = Span(chunks=chunks)
        if self.owner is None:
            raise ValueError("Span is not owned by a BlockText")
        self.owner.insert_after(self, new_span)
        return new_span
            

    def copy(self) -> Span:
        """
        Create a deep copy of this span.

        The copy has new chunk instances and no owner/linked list pointers.
        """
        return Span(
            chunks=self.chunks.copy(),
        )

    # --- Debug ---

    def __repr__(self) -> str:
        prev_id = f"prev={self.prev.id}" if self.prev is not None else None
        next_id = f"next={self.next.id}" if self.next is not None else None        
        return f"Span({self.chunks.text!r}, id={self.id}, {prev_id}, {next_id})"

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
            "content": self.chunks.text,
        }

        if include_chunks:
            result["content_chunks"] = [c.model_dump(exclude_none=exclude_none) for c in self.chunks]

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
            content = [BlockChunk.model_validate(c) for c in data.get("content_chunks", [])]
        else:
            # Create simple chunks from text
            content = [BlockChunk(content=data["content"])] if data.get("content") else []

        return cls(content=BlockChunkList(chunks=content))
