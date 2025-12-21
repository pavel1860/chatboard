from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pydantic_core import core_schema
from pydantic import BaseModel, GetCoreSchemaHandler
from typing import Any, Generator, Iterator, Literal, Self, Type, TYPE_CHECKING, TypeVar, overload
from uuid import uuid4
from abc import ABC

from .chunk import BlockChunk, BlockText
from .span import Span, SpanAnchor, VirtualBlockText
from .path import Path
from ...utils.type_utils import UNSET, UnsetType
if TYPE_CHECKING:
    pass


# Type alias for chunk mapping during copy
ChunkMap = dict[str, BlockChunk]  # old_chunk_id -> new_chunk


def _generate_id() -> str:
    """Generate a short unique ID for blocks."""
    return uuid4().hex[:8]

def parse_style(style: str | list[str] | None) -> list[str]:
    if isinstance(style, str):
        return list(style.split(" "))
    elif type(style) is list:
        return style
    else:
        return []

ContentType = str | list[str] | list[BlockChunk] | BlockChunk
PathType = str | list[int] | int | Path



class BlockBase(ABC):
    """
    Common functionality for both BlockSchema and Block.

    Holds:
    - Style for rendering format (xml, markdown, plain)
    - Attributes (key-value pairs)
    """

    
    __slots__ = [
        "styles", 
        "tags", 
        "role",
        "parent", 
        "children",
        "block_text", 
        "span",         
        "postfix_span",
        "prefix_span",
    ]
    
    def __init__(
        self,
        content: ContentType | None = None,
        children: list["BlockBase"] | None = None,
        role: str | None = None,
        style: str | list[str] | None = None,
        tags: list[str] | None = None,
        parent: "BlockBase | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
        _skip_content: bool = False,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ):
        self.styles = parse_style(style) if style is not None else styles or []
        self.role = role
        self.tags = tags or []
        self.parent = parent
        self.block_text = block_text or BlockText()
        self.children = children or []
        self.postfix_span: Span | None = None
        self.prefix_span: Span | None = None

        if _skip_content:
            # For copy operations - span will be set by caller
            self.span = None
        elif content is None:
            # Wrapper block - no content, no span
            self.span = None
        else:
            # Block with content (including empty string "")
            chunks_list = self.promote_content(content)
            chunks = self.block_text.extend(chunks_list)
            self.span = Span.from_chunks(chunks, start_offset=start_offset, end_offset=end_offset)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __len__(self) -> int:
        """Return number of children."""
        return len(self.children)

    def __bool__(self) -> bool:
        """Block is truthy if it has children."""
        return len(self.children) > 0

    @property
    def is_wrapper(self) -> bool:
        """Check if this is a wrapper block (no content)."""
        return self.span is None or self.span.is_empty

    @property
    def content(self) -> "BlockBase":
        """
        Return this block's content as a new independent Block (no children/style/tags).

        Creates a new BlockText with copies of the content chunks, preserving
        logprob metadata. The returned block is fully independent and can be
        used with operators like & without affecting the original.

        Returns:
            A new Block with copied content chunks

        Raises:
            ValueError: If called on a wrapper block (no content)
        """
        if self.span is None:
            raise ValueError("Cannot get content of wrapper block (no content). Check is_wrapper first.")

        # Fork just the content span's chunks
        new_block_text = self.block_text.fork(
            start=self.span.start.chunk,
            end=self.span.end.chunk
        )

        # Create block with the new BlockText
        block = Block(
            block_text=new_block_text,
            _skip_content=True,
        )
        block.span = Span.from_chunks(list(new_block_text))
        return block
    
    # def get_content(self, raise_on_wrapper: bool = True) -> "BlockBase":
    #     if self.span is None:
    #         if raise_on_wrapper:
    #             raise ValueError("Cannot get content of wrapper block (no content). Check is_wrapper first.")
    #         return Block(_skip_content=True)

    #     new_block_text = self.block_text.fork(
    #         start=self.span.start.chunk,
    #         end=self.span.end.chunk
    #     )

    #     # Create block with the new BlockText
    #     block = Block(
    #         block_text=new_block_text,
    #         _skip_content=True,
    #     )
    #     block.span = Span.from_chunks(list(new_block_text))
    #     return block
    
    @property
    def content_str(self) -> str:
        if self.is_wrapper:
            return ""
        content = self.content
        text = content.block_text.text()
        return text
    
    @property
    def body(self) -> "BlockList":
        return BlockList(self.children, role=self.role, tags=self.tags)
    @property
    def last_descendant(self) -> "BlockBase":
        """
        Get the deepest last descendant of this block.

        Recursively follows children[-1] until reaching a leaf block.
        Returns self if no children.

        Returns:
            The deepest last descendant, or self if no children
        """
        if self.children:
            return self.children[-1].last_descendant
        return self

    @property
    def content_end_chunk(self) -> BlockChunk | None:
        """Get the last chunk of content span, or None if wrapper."""
        return self.span.end.chunk if self.span else None

    @property
    def content_start_chunk(self) -> BlockChunk | None:
        """Get the first chunk of content span, or None if wrapper."""
        return self.span.start.chunk if self.span else None
    
    @property
    def end_postfix_chunk(self) -> BlockChunk | None:
        if self.postfix_span is None:
            return self.content_end_chunk
        return self.postfix_span.end.chunk
    
    @property
    def start_prefix_chunk(self) -> BlockChunk | None:
        if self.prefix_span is None:
            return self.content_start_chunk
        return self.prefix_span.start.chunk

    # -------------------------------------------------------------------------
    # Boundary properties (for entire subtree)
    # -------------------------------------------------------------------------

    @property
    def start_chunk(self) -> BlockChunk | None:
        """
        Get the first chunk of this block's entire subtree.

        Returns the earliest chunk considering:
        - prefix span (if present)
        - content span (if present)
        - first child's start (if wrapper with children)

        Returns None only if wrapper block with no children.
        """
        if self.start_prefix_chunk is not None:
            return self.start_prefix_chunk
        elif self.content_start_chunk is not None:
            return self.content_start_chunk
        elif self.children:
            # Wrapper block - get start from first child
            return self.children[0].start_chunk
        else:
            # Empty wrapper block
            return None

    @property
    def end_chunk(self) -> BlockChunk | None:
        """
        Get the last chunk of this block's entire subtree.

        Returns the latest chunk considering:
        - postfix span of deepest last child
        - or this block's postfix/content span if no children

        Returns None only if wrapper block with no children.
        """
        if self.children:
            # Recursively get boundary_end of the last child
            return self.children[-1].end_chunk
        elif self.end_postfix_chunk is not None:
            # No children - return this block's end
            return self.end_postfix_chunk
        elif self.content_end_chunk is not None:
            return self.content_end_chunk
        else:
            # Empty wrapper block
            return None

    def get_boundaries(self) -> tuple[BlockChunk | None, BlockChunk | None]:
        """
        Get the start and end chunks of this block's entire subtree.

        Returns:
            Tuple of (start_chunk, end_chunk) covering the entire block tree
            including prefix, content, children, and postfix.

        Example:
            start, end = block.get_boundaries()
            # Fork just this block's chunks
            new_text = block._block_text.fork(start=start, end=end)
        """
        return (self.start_chunk, self.end_chunk)

    def get_chunks(self) -> list[BlockChunk]:
        """
        Get all chunks within this block's boundaries.

        Returns:
            List of chunks from start_chunk to end_chunk (inclusive),
            covering prefix, content, children, and postfix.
        """
        start, end = self.get_boundaries()
        if start is None:
            return []

        chunks = []
        current = start
        while current is not None:
            chunks.append(current)
            if current is end:
                break
            current = current.next

        return chunks
    
    
    
    def set_block_text(self, chunks: list[BlockChunk], block_text: BlockText):
        self.block_text = block_text
        self.span = Span.from_chunks(chunks)
        return self


    # -------------------------------------------------------------------------
    # Path properties
    # -------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """
        Compute current path dynamically.

        Returns a Path object with both index-based and tag-based paths.
        The path always includes the current block itself.
        """
        indices = []
        tags = []

        # Walk up from this block to root, collecting indices and tags
        current = self
        while current is not None:
            if current.parent is not None:
                # Get index of current in parent's children
                idx = current.parent.children.index(current)
                indices.append(idx)
            else:
                # Root block - use index 0 as convention
                indices.append(0)

            # Collect first tag if present
            if current.tags:
                tags.append(current.tags[0])

            current = current.parent

        # Reverse since we collected from leaf to root
        indices.reverse()
        tags.reverse()

        return Path(indices, tags)

    @property
    def tag_path(self) -> list[str]:
        """Get tag-based path as a list of strings."""
        return list(self.path.tags)

    @property
    def depth(self) -> int:
        """Get depth in tree (0 for root)."""
        return self.path.depth

    @property
    def index(self) -> int | None:
        """Get index in parent's children list."""
        if self.parent is None:
            return None
        return self.parent.children.index(self)
    
    def _parse_path(self, path: str | list[int] | int | Path) -> Path:
        if isinstance(path, str):
            return Path.from_string(path)
        elif isinstance(path, int):
            return Path([path])
        elif isinstance(path, list):
            return Path(path)
        elif isinstance(path, Path):
            return path
        else:
            raise ValueError(f"Invalid path type: {type(path)}")

    def path_get(self, path: str | list[int] | Path) -> "BlockBase | None":
        """
        Get block at the given path relative to this block.

        Args:
            path: Path as string "0.2.1", list [0, 2, 1], or Path object

        Returns:
            Block at path, or None if not found
        """
        if isinstance(path, str):
            path = Path.from_string(path)
        elif isinstance(path, list):
            path = Path(path)

        target = self
        for idx in path.indices:
            if idx >= len(target.children):
                return None
            target = target.children[idx]

        return target

    def path_exists(self, path: str | list[int] | Path) -> bool:
        """Check if a path exists relative to this block."""
        return self.path_get(path) is not None

    # -------------------------------------------------------------------------
    # Tag-based search methods
    # -------------------------------------------------------------------------

    def traverse(self) -> Iterator["BlockBase"]:
        """
        Iterate over this block and all descendants (pre-order depth-first).

        Yields:
            This block, then recursively all descendants
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    def get_all(self, tags: str | list[str]) -> list["BlockBase"]:
        """
        Get all blocks matching a tag path.

        Supports dot-notation for nested tag searches:
        - "response" - find all blocks with tag "response"
        - "response.thinking" - find "thinking" blocks that are descendants of "response"

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            List of matching blocks
        """
        if isinstance(tags, str):
            tags = tags.split(".")

        if not tags:
            return []

        # Find all blocks matching first tag
        candidates = [b for b in self.traverse() if tags[0] in b.tags]

        # Filter through remaining tags
        for tag in tags[1:]:
            next_candidates = []
            for blk in candidates:
                for child in blk.traverse():
                    if child is not blk and tag in child.tags:
                        next_candidates.append(child)
            candidates = next_candidates

        return candidates

    def get_one(self, tags: str | list[str]) -> "BlockBase":
        """
        Get the first block matching a tag path.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block

        Raises:
            ValueError: If no matching block found
        """
        result = self.get_all(tags)
        if not result:
            raise ValueError(f'Tag path "{tags}" does not exist')
        return result[0]

    def get_one_or_none(self, tags: str | list[str]) -> "BlockBase | None":
        """
        Get the first block matching a tag path, or None if not found.

        Args:
            tags: Single tag, dot-separated path, or list of tags

        Returns:
            First matching block, or None
        """
        result = self.get_all(tags)
        return result[0] if result else None
    
    def get_list(self, tag: str) -> "BlockList | BlockListSchema":
        block = self.get_one(tag)
        if block is None:
            raise ValueError(f"Block {tag} not found")
        if not isinstance(block, BlockList) and not isinstance(block, BlockListSchema):
            raise ValueError(f"Block {block} is not a BlockList or BlockListSchema")
        return block
        

    def get(self, tag: str) -> "BlockBase | None":
        """
        Get the first direct or nested child with the given tag.

        Simple recursive search - does not support dot-notation paths.

        Args:
            tag: Tag to search for

        Returns:
            First matching block, or None
        """
        if tag in self.tags:
            return self
        for child in self.children:
            if tag in child.tags:
                return child
            if (block := child.get(tag)) is not None:
                return block
        return None

    def get_last(self, tag: str) -> "BlockBase | None":
        """
        Get the last block with the given tag.

        Args:
            tag: Tag to search for

        Returns:
            Last matching block, or None
        """
        result = None
        for blk in self.traverse():
            if tag in blk.tags:
                result = blk
        return result
    
    
    def max_depth(self) -> int:
        """
        Get the maximum depth of this block.
        """
        return max(blk.depth for blk in self.traverse())
    
    def min_depth(self) -> int:
        """
        Get the minimum depth of this block.
        """
        return min(blk.depth for blk in self.traverse())
    
    
    # -------------------------------------------------------------------------
    # Content processing methods
    # -------------------------------------------------------------------------


    def promote_content(self, content: "BlockBase |ContentType") -> list[BlockChunk]:
        """
        Convert content to a list of Chunks.

        Note: None is not accepted - wrapper blocks should not call this method.
        """
        if isinstance(content, str):
            return [BlockChunk(content)]
        elif isinstance(content, list):
            return content
        elif isinstance(content, Block):
            if len(content) > 0:
                raise ValueError("Cant append a block with body content")
            return  [chunk.copy() for chunk in content.span.chunks()] if content.span else []
        elif isinstance(content, BlockChunk):
            return [content]
        
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
    
    def promote_block_content(self, content: BlockBase |ContentType | None, style: str | None = None, tags: list[str] | None = None, role: str | None = None) -> "BlockBase":
        if isinstance(content, BlockBase):
            return content.copy()
        if content is None:
            return Block(parent=self, block_text=self.block_text, style=style, tags=tags, role=role)
        else:
            content = self.promote_content(content)
            return Block(content=content, block_text=self.block_text, style=style, tags=tags, role=role)    
        
    def _append_separator(self, content: list[BlockChunk], sep: str | None, append: bool = True):
        if sep:
            if not content[-1].is_line_end:
                if append:
                    content = [BlockChunk(content=sep)] + content
                else:
                    content = content + [BlockChunk(content=sep)]
        return content
    
    # def _split_new_lines(self, chunks: list[BlockChunk]) -> list[list[BlockChunk]]:
    #     parts = [[]]
    #     for i, chunk in enumerate(chunks):            
    #         if chunk.is_line_end and i < len(chunks) - 1:
    #             parts.append(chunk)
    #             parts.append([])
    #         else:
    #             parts[-1].append(chunk)
    #     return parts
    
    # def append(self, content: "BlockBase | ContentType", sep: str | None = " "):
    #     # if isinstance(content, BlockBase):
    #     #     self.append_child(content)
    #     #     return
    #     content = self.promote_content(content)
    #     content = self._append_separator(content, sep, append=True)
    #     chunks = self.block_text.extend(content, after=self.content_end_chunk)
    #     if self.span is None:
    #         # Wrapper block becoming a content block
    #         self.span = Span.from_chunks(chunks)
    #     else:
    #         self.span.end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
    
    def _has_end_of_line(self) -> bool:
        if self.postfix_span is not None:            
            for c in self.postfix_span.chunks():
                if c.is_line_end:
                    return True
        return False
    
    
    def _split_new_lines(self, chunks: list[BlockChunk]) -> Generator[tuple[list[BlockChunk], str], Any, Any]:
        sentence = []
        tag = "content"
        for i, chunk in enumerate(chunks):            
            if chunk.is_line_end:
                if sentence:
                    yield sentence, tag
                yield [chunk], "new_line"
                tag = "sentence"
                sentence = []
            else:
                sentence.append(chunk)
        if sentence:
            yield sentence, tag
            
            
        
    
    def append(self, content: BlockBase | ContentType, sep: str | None = " ", as_child: bool = False, start_offset: int | None = None, end_offset: int | None = None):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=True)
        target_block = self
        if len(self) > 0:
            target_block = self.children[-1]
        if target_block._has_end_of_line() or as_child:
            self.append_child(content, add_new_line=False, start_offset=start_offset, end_offset=end_offset)
            return
        for contentpart, tag in self._split_new_lines(content):            
            if tag == "content":
                target_block.inline_append(contentpart, start_offset=start_offset, end_offset=end_offset)
            elif tag == "new_line":
                target_block.postfix_append(contentpart, start_offset=start_offset, end_offset=end_offset)
            elif tag == "sentence":
                self.append_child(contentpart, add_new_line=False)

    def prepend(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=False)
        chunks = self.block_text.left_extend(content, before=self.content_start_chunk)
        if self.span is None:
            # Wrapper block becoming a content block
            self.span = Span.from_chunks(chunks)
        else:
            self.span.start = SpanAnchor(chunk=chunks[0], offset=0)
            
            
    def inline_append(self, chunks: list[BlockChunk], start_offset: int | None = None, end_offset: int | None = None):
        start_offset = start_offset if start_offset is not None else 0
        end_offset = end_offset if end_offset is not None else len(chunks[-1].content)
        chunks = self.block_text.extend(chunks, after=self.content_end_chunk)
        if self.span is None:
            # Wrapper block becoming a content block
            self.span = Span.from_chunks(chunks, start_offset=start_offset, end_offset=end_offset)
        else:
            # self.span.end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
            self.span.end = SpanAnchor(chunk=chunks[-1], offset=end_offset)

        
    def postfix_append(self, content: ContentType, sep: str | None = "", start_offset: int | None = None, end_offset: int | None = None):
        content = self.promote_content(content)
        start_offset = start_offset if start_offset is not None else 0
        end_offset = end_offset if end_offset is not None else len(content[-1].content)
        content = self._append_separator(content, sep, append=True)
        chunks = self.block_text.extend(content, after=self.end_postfix_chunk)
        self.postfix_span = Span.from_chunks(chunks, start_offset=start_offset, end_offset=end_offset)
        
    def prefix_prepend(self, content: ContentType, sep: str | None = "", start_offset: int | None = None, end_offset: int | None = None):
        content = self.promote_content(content)
        start_offset = start_offset if start_offset is not None else 0
        end_offset = end_offset if end_offset is not None else len(content[-1].content)
        content = self._append_separator(content, sep, append=False)
        chunks = self.block_text.left_extend(content, before=self.start_prefix_chunk)
        self.prefix_span = Span.from_chunks(chunks, start_offset=start_offset, end_offset=end_offset)
        
    def append_child(self, child_content: BlockBase | ContentType, copy: bool = True, add_new_line: bool = True, start_offset: int | None = None, end_offset: int | None = None):
        """
        Append a child block to this block's children.

        Inserts chunks at the correct position in the BlockText (after this
        block's last child, or after this block's content if no children).

        Args:
            child_content: Block or content to append as child
            copy: If True (default), copy the block before appending.
                  If False, move the block directly (caller loses ownership).

        Returns:
            The appended block
        """
        # Create or copy the block
        if isinstance(child_content, BlockBase):
            block = child_content.copy() if copy else child_content
        else:
            block = Block(content=child_content, start_offset=start_offset, end_offset=end_offset)

        # Add newline separator (skip for wrapper blocks with no content)
        if add_new_line:
            # if len(self) == 0:
            #     if not self._has_end_of_line():
            #         self.add_new_line()
            if self.children:
                self.last_descendant.add_new_line()
            elif not self.is_wrapper:
                self.add_new_line()

        # Only move chunks if block has a different BlockText
        # (If block already shares our BlockText, chunks are already in place)
        if block.block_text is not self.block_text:
            # Determine insertion point - after last child or after this block's content
            if self.children:
                insert_after_chunk = self.children[-1].end_chunk
            else:
                insert_after_chunk = self.end_postfix_chunk or self.content_end_chunk

            # Move chunks from block's BlockText to this block's BlockText
            self.block_text.extend_block_text(
                block.block_text,
                after=insert_after_chunk,
                copy=False
            )

            # Remap the block's span and update BlockText reference for entire subtree
            self._remap_block_text(block, self.block_text)

        # Add to children list
        self.children.append(block)
        block.parent = self
        return block
        
    def insert(self, path: PathType, content: Block | ContentType):
        """
        Insert a block at the given path.

        The path specifies where to insert. The last index in the path is the
        position within the parent's children list. The preceding indices
        navigate to the parent block.

        If content is a Block with its own BlockText, its chunks are moved
        into this block's BlockText and spans are remapped.

        Args:
            path: Path to insert at. Examples:
                - [0] or "0": insert at index 0 of this block's children
                - [1, 2] or "1.2": navigate to child[1], insert at index 2
            content: Block or content to insert

        Returns:
            The inserted block

        Raises:
            ValueError: If path is empty or parent doesn't exist
        """
        path = self._parse_path(path)
        if len(path) == 0:
            raise ValueError("Path cannot be empty for insert")

        # Navigate to parent block
        if len(path) == 1:
            parent = self
        else:
            parent_path = Path(list(path.indices[:-1]))
            parent = self.path_get(parent_path)
            if parent is None:
                raise ValueError(f"Parent path {parent_path} does not exist")

        index = path.indices[-1]

        # Create or copy the block
        if isinstance(content, BlockBase):
            block = content.copy()
        else:
            # Create a new block with its own temporary BlockText
            block = Block(content=content)

        # Determine insertion point in the linked list
        if parent.children and index < len(parent.children):
            # Insert before the child at index
            insert_before_chunk = parent.children[index].start_chunk
            inserted_chunks = self.block_text.left_extend_block_text(
                block.block_text,
                before=insert_before_chunk,
                copy=False
            )
        elif parent.children:
            # Append after last child
            insert_after_chunk = parent.children[-1].end_chunk
            inserted_chunks = self.block_text.extend_block_text(
                block.block_text,
                after=insert_after_chunk,
                copy=False
            )
        else:
            # No children yet, insert after parent's content
            insert_after_chunk = parent.end_chunk
            inserted_chunks = self.block_text.extend_block_text(
                block.block_text,
                after=insert_after_chunk,
                copy=False
            )

        # Remap the block's span to the new chunks in our BlockText
        if inserted_chunks:
            block.block_text = self.block_text
            block.span = Span.from_chunks(inserted_chunks)

        # Insert into children list
        if parent.children:
            parent.children.insert(index, block)
        else:
            parent.children = [block]

        block.parent = parent
        block.add_new_line()
        return block
        

    def replace(self, path: PathType, other: BlockBase):
        """
        Replace the block at the given path with the other block.

        Replaces both the block in the tree and its chunks in the BlockText.

        Args:
            path: Path or string path
            other: Block to replace with

        Returns:
            The replaced (removed) block
        """
        path = self._parse_path(path)
        target = self.path_get(path)
        if target is None:
            raise ValueError(f"Invalid path: {path}, target not found")
        parent = target.parent
        if parent is None:
            # Replacing root - can't modify BlockText
            raise ValueError("Cannot replace root block")

        # Copy the replacement block if it has its own BlockText
        if isinstance(other, BlockBase):
            replacement = other.copy() if other.block_text is not self.block_text else other
        else:
            replacement = Block(content=other)

        # Get target's chunk boundaries
        target_start = target.start_chunk
        target_end = target.end_chunk

        # Replace chunks in BlockText
        removed_chunks, inserted_chunks = self.block_text.replace_block_text(
            target_start,
            target_end,
            replacement.block_text,
            copy=False
        )

        # Remap replacement's span to the newly inserted chunks
        if inserted_chunks:
            replacement.block_text = self.block_text
            replacement.span = Span.from_chunks(inserted_chunks)

        # Update tree structure
        idx = parent.children.index(target)
        parent.children.remove(target)
        parent.children.insert(idx, replacement)
        replacement.parent = parent
        replacement.add_new_line()

        # Clear target's ownership
        target.parent = None

        return target

    def remove(self, index: int):
        """
        Remove child at the given index.

        Removes the child from the tree and its chunks from the BlockText.

        Args:
            index: Index of child to remove

        Returns:
            The removed block
        """
        if not self.children or index >= len(self.children):
            raise IndexError(f"Child index {index} out of range")

        child = self.children[index]

        # Remove chunks from BlockText
        self.block_text.replace(child.start_chunk, child.end_chunk, None)

        # Remove from tree
        self.children.pop(index)
        child.parent = None

        return child
        
    


    def _remap_block_text(self, block: "BlockBase", new_block_text: BlockText):
        """
        Recursively update _block_text reference for a block and all its descendants.

        Used after moving chunks between BlockTexts to ensure all blocks
        in a subtree point to the correct BlockText.
        """
        block.block_text = new_block_text
        for child in block.children:
            self._remap_block_text(child, new_block_text)

    def _append_block_child(self, block: "Block"):
        """
        Append an already-prepared block to children list.

        This is a low-level method used by _copy_tree. It only updates
        the tree structure, not the BlockText. Use append_child() for
        normal operations.
        """
        self.children.append(block)
        block.parent = self
        return block
    
    
    # -------------------------------------------------------------------------
    # String operations
    # -------------------------------------------------------------------------

    def strip(self) -> "BlockBase":
        """
        Remove whitespace from the outer edges of prefix and postfix spans.

        This modifies the block in-place by:
        1. Removing leading whitespace from prefix_span (whitespace before the block)
        2. Removing trailing whitespace from postfix_span (whitespace after the block)

        For chunks that are partially whitespace, the chunk content is modified
        to strip the whitespace portion. Whitespace-only chunks are removed from
        the BlockText entirely.

        Does NOT remove whitespace from within the main content span itself,
        only from the prefix/postfix decorations.

        Returns:
            self for method chaining
        """
        # Strip prefix: remove leading whitespace (outer edge, before content)
        if self.prefix_span is not None:
            self.prefix_span = self._strip_span_start(self.prefix_span)

        # Strip postfix: remove trailing whitespace (outer edge, after content)
        if self.postfix_span is not None:
            self.postfix_span = self._strip_span_end(self.postfix_span)

        return self

    def _strip_span_end(self, span: Span) -> Span | None:
        """
        Remove trailing whitespace from a span.

        Removes whitespace-only chunks from the BlockText and adjusts
        chunk content for partial whitespace. Returns the modified span
        or None if the span becomes empty.
        """
        if span.is_empty:
            return None

        # Collect chunks in the span
        chunks_to_check: list[BlockChunk] = list(span.chunks())
        chunks_to_remove: list[BlockChunk] = []
        new_end_chunk: BlockChunk | None = None
        new_end_offset: int = 0

        # Walk backwards from end, find trailing whitespace
        for chunk in reversed(chunks_to_check):
            if chunk == span.end.chunk:
                # Check the portion within the span
                end_offset = span.end.offset
                start_offset = span.start.offset if chunk == span.start.chunk else 0
                text_in_span = chunk.content[start_offset:end_offset]
                stripped = text_in_span.rstrip()

                if len(stripped) == 0:
                    # Entire portion is whitespace - mark for removal or skip
                    if chunk == span.start.chunk:
                        # Entire span is whitespace
                        chunks_to_remove = chunks_to_check
                        break
                    else:
                        chunks_to_remove.append(chunk)
                        continue
                else:
                    # Partial whitespace - adjust the chunk content
                    new_end_chunk = chunk
                    new_end_offset = start_offset + len(stripped)
                    # Modify chunk content to remove trailing whitespace
                    chunk.content = chunk.content[:new_end_offset]
                    break
            else:
                # Middle or start chunk
                start_offset = span.start.offset if chunk == span.start.chunk else 0
                text_in_span = chunk.content[start_offset:]
                stripped = text_in_span.rstrip()

                if len(stripped) == 0:
                    # Entire portion is whitespace
                    if chunk == span.start.chunk:
                        # From start to here is all whitespace
                        chunks_to_remove = chunks_to_check
                        break
                    else:
                        chunks_to_remove.append(chunk)
                        continue
                else:
                    # Found non-whitespace
                    new_end_chunk = chunk
                    new_end_offset = start_offset + len(stripped)
                    chunk.content = chunk.content[:new_end_offset]
                    break

        # Remove whitespace-only chunks from BlockText
        for chunk in chunks_to_remove:
            if chunk in self.block_text:
                self.block_text.remove(chunk)

        # Return updated span or None
        if new_end_chunk is None:
            return None

        return Span(
            start=span.start,
            end=SpanAnchor(chunk=new_end_chunk, offset=new_end_offset)
        )

    def _strip_span_start(self, span: Span) -> Span | None:
        """
        Remove leading whitespace from a span.

        Removes whitespace-only chunks from the BlockText and adjusts
        chunk content for partial whitespace. Returns the modified span
        or None if the span becomes empty.
        """
        if span.is_empty:
            return None

        # Collect chunks in the span
        chunks_to_check: list[BlockChunk] = list(span.chunks())
        chunks_to_remove: list[BlockChunk] = []
        new_start_chunk: BlockChunk | None = None
        new_start_offset: int = 0

        # Walk forward from start, find leading whitespace
        for chunk in chunks_to_check:
            if chunk == span.start.chunk:
                # Check the portion within the span
                start_offset = span.start.offset
                end_offset = span.end.offset if chunk == span.end.chunk else len(chunk.content)
                text_in_span = chunk.content[start_offset:end_offset]
                stripped = text_in_span.lstrip()

                if len(stripped) == 0:
                    # Entire portion is whitespace - mark for removal or skip
                    if chunk == span.end.chunk:
                        # Entire span is whitespace
                        chunks_to_remove = chunks_to_check
                        break
                    else:
                        chunks_to_remove.append(chunk)
                        continue
                else:
                    # Partial whitespace - adjust
                    whitespace_len = len(text_in_span) - len(stripped)
                    new_start_chunk = chunk
                    new_start_offset = start_offset + whitespace_len
                    # Modify chunk content to remove leading whitespace
                    chunk.content = chunk.content[new_start_offset:]
                    new_start_offset = 0  # After modification, start is at 0
                    break
            else:
                # Middle or end chunk
                end_offset = span.end.offset if chunk == span.end.chunk else len(chunk.content)
                text_in_span = chunk.content[:end_offset]
                stripped = text_in_span.lstrip()

                if len(stripped) == 0:
                    # Entire portion is whitespace
                    if chunk == span.end.chunk:
                        chunks_to_remove = chunks_to_check
                        break
                    else:
                        chunks_to_remove.append(chunk)
                        continue
                else:
                    # Found non-whitespace
                    whitespace_len = len(text_in_span) - len(stripped)
                    new_start_chunk = chunk
                    new_start_offset = whitespace_len
                    chunk.content = chunk.content[new_start_offset:]
                    new_start_offset = 0
                    break

        # Remove whitespace-only chunks from BlockText
        for chunk in chunks_to_remove:
            if chunk in self.block_text:
                self.block_text.remove(chunk)

        # Return updated span or None
        if new_start_chunk is None:
            return None

        return Span(
            start=SpanAnchor(chunk=new_start_chunk, offset=new_start_offset),
            end=span.end
        )
        
    
    # -------------------------------------------------------------------------
    # Rendering operations
    # -------------------------------------------------------------------------

    
    def transform(self) -> "BlockBase":
        from .block_transformers import transform
        # transform() handles copying internally - no need for upfront copy
        return transform(self)

    def render(self) -> str:
        block = self.transform()
        return block.block_text.text()
    
    
    # -------------------------------------------------------------------------
    # Serialization operations
    # -------------------------------------------------------------------------
    
    
    def model_dump(self, exclude: set[str] | None = None, include: set[str] | None = None, overrides: dict[str, Any] | None = None):        
        dump = overrides or {}
        for field in self.__slots__:
            if (exclude and field in exclude) or (include and field not in include) or (overrides and field in overrides):
                continue
            value = getattr(self, field)
            dump[field] = value
        return dump
    
    
    
    def model_copy(self, copy_content: bool = True, fork_block_text: bool = False, copy_children: bool = True) -> "BlockBase":
        raise NotImplementedError("model_copy is not implemented")

    def model_metadata_copy(self, overrides: dict[str, Any] | None = None) -> Self:
        dump = {
            "role": self.role,
            "tags": self.tags,
            "styles": self.styles,
        }
        if overrides:
            dump.update(overrides)
        return self.__class__(**dump)

    
    
    # def copy_metadata(self) -> "BlockBase":
    #     """
    #     Copy this block's metadata and content, but NOT children.

    #     Creates a new BlockText with forked content chunks. The returned
    #     block has no children and no parent - ready for building a new tree.

    #     For wrapper blocks (no content), creates an empty wrapper.
    #     """
    #     if self.span is None:
    #         # Wrapper block - no content to fork
    #         return Block(
    #             styles=list(self.styles),
    #             tags=list(self.tags),
    #             role=self.role,
    #         )

    #     # Fork just this block's content (not children)
    #     new_block_text = self.block_text.fork(
    #         start=self.span.start.chunk,
    #         end=self.span.end.chunk
    #     )

    #     new_block = Block(
    #         styles=list(self.styles),
    #         tags=list(self.tags),
    #         role=self.role,
    #         block_text=new_block_text,
    #         _skip_content=True,
    #     )
    #     new_block.span = Span.from_chunks(list(new_block_text))
    #     return new_block
    def _copy_content_block_text(self) -> BlockText:
        if self.span is None:
            return BlockText()
        return self.block_text.fork(
            start=self.span.start.chunk,
            end=self.span.end.chunk
        )
        
        
    def _copy_block_text(self) -> BlockText:
        return self.block_text.fork(
            start=self.start_chunk,
            end=self.end_chunk
        )
    
    
    def copy_metadata(self, with_prefix: bool = False, with_postfix: bool = False) -> "BlockBase":
        """
        Copy this block's metadata and content, but NOT children.

        Creates a new BlockText with forked content chunks. The returned
        block has no children and no parent - ready for building a new tree.

        For wrapper blocks (no content), creates an empty wrapper.
        """
        if self.span is None:
            # Wrapper block - no content to fork
            return self.model_metadata_copy()

        # Fork just this block's content (not children)
        new_block_text = self.block_text.fork(
            start=self.span.start.chunk,
            end=self.span.end.chunk
        )

        # new_block = Block(
        #     styles=list(self.styles),
        #     tags=list(self.tags),
        #     role=self.role,
        #     block_text=new_block_text,
        #     _skip_content=True,
        # )
        new_block = self.model_metadata_copy(overrides={"block_text": new_block_text, "_skip_content": True})
        new_block.span = Span.from_chunks(list(new_block_text))
        return new_block
    


    def copy(self) -> "BlockBase":
        """
        Create a deep copy of this block and its subtree.

        Copies the underlying BlockText and rebuilds spans to point to new chunks.
        For wrapper blocks with no content, copies only the tree structure.
        """
        start, end = self.get_boundaries()

        if start is None:
            # Empty wrapper block - no chunks to copy
            new_block_text = BlockText()
            chunk_map = {}
        else:
            # Copy the BlockText (creates new chunks with same content)
            new_block_text = self.block_text.fork(start, end)

            # Build chunk mapping: old_id -> new_chunk
            chunk_map = {}
            current = start
            new_chunk_iter = iter(new_block_text)
            while current is not None:
                chunk_map[current.id] = next(new_chunk_iter)
                if current is end:
                    break
                current = current.next

        # Copy block tree with remapped spans
        return self._copy_tree(chunk_map, new_block_text)

    def _copy_tree(self, chunk_map: dict, new_block_text: BlockText) -> "BlockBase":
        """Copy this block using chunk mapping."""
        # new_block = Block(
        #     styles=list(self.styles),
        #     tags=list(self.tags),
        #     role=self.role,
        #     block_text=new_block_text,
        #     _skip_content=True,
        # )
        new_block = self.model_metadata_copy(overrides={"block_text": new_block_text, "_skip_content": True})

        # Remap spans (only if not a wrapper block)
        if self.span is not None:
            new_block.span = Span(
                start=SpanAnchor(chunk_map[self.span.start.chunk.id], self.span.start.offset),
                end=SpanAnchor(chunk_map[self.span.end.chunk.id], self.span.end.offset),
            )

        if self.prefix_span:
            new_block.prefix_span = Span(
                start=SpanAnchor(chunk_map[self.prefix_span.start.chunk.id], self.prefix_span.start.offset),
                end=SpanAnchor(chunk_map[self.prefix_span.end.chunk.id], self.prefix_span.end.offset),
            )

        if self.postfix_span:
            new_block.postfix_span = Span(
                start=SpanAnchor(chunk_map[self.postfix_span.start.chunk.id], self.postfix_span.start.offset),
                end=SpanAnchor(chunk_map[self.postfix_span.end.chunk.id], self.postfix_span.end.offset),
            )

        # Copy children
        for child in self.children:
            new_block._append_block_child(child._copy_tree(chunk_map, new_block_text))

        return new_block
    

    def traverse(self) -> Iterator["BlockBase"]:
        yield self
        for child in self.children:
            yield from child.traverse()

            
            
    def apply_style(self, style: str, only_views: bool = False):
        block_copy = self.copy()
        styles = parse_style(style)
        for block in block_copy.traverse():
            if only_views and not isinstance(block, BlockSchema):
                continue
            block.styles.extend(styles)
        return self
    
    
    
    def extract_schema(self) -> "BlockSchema":
        """
        Extract a new BlockSchema tree containing only BlockSchema nodes.

        Traverses this block's subtree and creates a new BlockSchema with
        only BlockSchema children, preserving the schema hierarchy while
        filtering out regular Block nodes.

        Returns:
            A new BlockSchema tree with only schema nodes
        """
        # Create a copy of this schema (without children)
        # new_schema = self.__class__(
        #     name=self.name,
        #     type=self.type,
        #     role=self.role,
        #     styles=list(self.styles),
        #     tags=list(self.tags),
        # )
        if isinstance(self, BlockSchema):
            new_schema = self.model_copy()
        else:
            new_schema = BlockSchema(
                self.content_str,
                role=self.role,
                tags=self.tags,
                styles=self.styles,
                is_virtual=True,
            )

        # Recursively extract schemas from children
        for child in self.children:
            if isinstance(child, BlockSchema) or isinstance(child, BlockListSchema):
                # Recursively extract and add as child
                child_schema = child.extract_schema()
                new_schema.children.append(child_schema)
                child_schema.parent = new_schema
            else:
                # For regular blocks, check their children for nested schemas
                self._extract_nested_schemas(child, new_schema)
                
                
        if new_schema.is_wrapper and len(new_schema.children) == 1:
            new_schema = new_schema.children[0]
            new_schema.parent = None

        return new_schema


    def _extract_nested_schemas(self, block: "BlockBase", parent_schema: "BlockSchema"):
        """
        Recursively search a Block's children for BlockSchema nodes.

        Any found BlockSchema nodes are extracted and added to parent_schema.
        """
        for child in block.children:
            if isinstance(child, BlockSchema):
                child_schema = child.extract_schema()
                parent_schema.children.append(child_schema)
                child_schema.parent = parent_schema
            else:
                # Keep searching deeper
                self._extract_nested_schemas(child, parent_schema)

    

            
            
    # -------------------------------------------------------------------------
    # Text operations
    # -------------------------------------------------------------------------
    
    def add_new_line(self):
        self.postfix_append(BlockChunk(content="\n"))
        
        
        
    # -------------------------------------------------------------------------
    # Prompt Context operations
    # -------------------------------------------------------------------------
    
    
    def __call__(
        self,
        content: ContentType | BlockBase | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "Block":
        block = self.promote_block_content(content, style=style, tags=tags, role=role)
        self.append_child(block, copy=False)
        return block
    
    @classmethod
    def schema_view(cls, name: str | None = None, type: Type | None = None, tags: list[str] | None = None, style: str | None = None) -> "BlockSchema":
        schema_block = BlockSchema(
            name,
            type=type,
            tags=tags,
            styles=["xml"] if style is None and name is not None else parse_style(style),
        )
        return schema_block
    
    def view(
        self,
        name: str | None = None,
        type: Type | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "BlockSchema":
        schema_block = BlockSchema(
            name,
            type=type,
            tags=tags,
            styles=["xml"] if style is None and name is not None else parse_style(style),
        )
        self.append_child(schema_block, copy=False)
        return schema_block


    def view_list(
        self,
        item_name: str,
        key: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "BlockListSchema":
        schema_block = BlockListSchema(
            item_name=item_name,
            key=key,
            name=name,
            tags=tags,
            styles=["xml-list"] if style is None else parse_style(style),
        )
        self.append_child(schema_block, copy=False)
        return schema_block
    
    # -------------------------------------------------------------------------
    # Debug methods
    # -------------------------------------------------------------------------

    def debug_tree(self, indent: int = 0) -> str:
        """
        Generate a debug representation of this block and its subtree.

        Shows for each block:
        - Path, tags, styles
        - Content chunks (with repr for visibility of newlines/spaces)
        - Children recursively indented

        Returns:
            Debug string representation
        """
        lines = []
        prefix = "  " * indent

        # Block header
        path = self.path
        path_str = str(path) if path else "(root)"
        wrapper_str = "(wrapper)" if self.is_wrapper else ""
        tags_str = f"tags={self.tags}" if self.tags else ""
        styles_str = f"styles={self.styles}" if self.styles else ""
        header_parts = [f"[{path_str}]", wrapper_str, tags_str, styles_str]
        header = " ".join(p for p in header_parts if p)
        lines.append(f"{prefix}{header}")

        # Content span chunks
        if self.span:
            content_chunks = []
            chunk = self.span.start.chunk
            while chunk is not None:
                content_chunks.append(repr(chunk.content))
                if chunk is self.span.end.chunk:
                    break
                chunk = chunk.next
            lines.append(f"{prefix}  content: [{', '.join(content_chunks)}]")

        # Prefix span if present
        if self.prefix_span:
            prefix_chunks = []
            chunk = self.prefix_span.start.chunk
            while chunk is not None:
                prefix_chunks.append(repr(chunk.content))
                if chunk is self.prefix_span.end.chunk:
                    break
                chunk = chunk.next
            lines.append(f"{prefix}  prefix: [{', '.join(prefix_chunks)}]")

        # Postfix span if present
        if self.postfix_span:
            postfix_chunks = []
            chunk = self.postfix_span.start.chunk
            while chunk is not None:
                postfix_chunks.append(repr(chunk.content))
                if chunk is self.postfix_span.end.chunk:
                    break
                chunk = chunk.next
            lines.append(f"{prefix}  postfix: [{', '.join(postfix_chunks)}]")

        # Children
        for child in self.children:
            lines.append(child.debug_tree(indent + 1))

        return "\n".join(lines)

    def print_debug(self):
        """Print debug tree representation."""
        print(self.debug_tree())

    def debug_block_text(self) -> str:
        """
        Generate a debug representation of the entire BlockText.

        Shows all chunks in order with their IDs and content.
        """
        lines = ["BlockText chunks:"]
        for i, chunk in enumerate(self.block_text):
            lines.append(f"  [{i}] {chunk.id[:8]}: {repr(chunk.content)}")
        return "\n".join(lines)

    def print_block_text(self):
        """Print BlockText debug representation."""
        print(self.debug_block_text())
        
        
    # -------------------------------------------------------------------------
    # Pydantic model support
    # -------------------------------------------------------------------------
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize
            )
        )
        
    @staticmethod
    def _validate(v: Any) -> Any:
        if isinstance(v, Block):
            return v
        elif isinstance(v, dict):
            if "_type" in v and v["_type"] == "Block":
                return Block.model_validate(v)
        else:
            raise ValueError(f"Invalid block: {v}")

    @staticmethod
    def _serialize(v: Any) -> Any:
        if isinstance(v, Block):
            return v.model_dump()
        else:
            raise ValueError(f"Invalid block: {v}")


    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def print(self):
        print(self.render())
        
        
    def __itruediv__(self, other: BlockBase | ContentType):
        self.append_child(other)
        return self
    
    def __add__(self, other: BlockBase | ContentType):
        self_copy = self.copy()
        self_copy.append(other)
        return self_copy
    
    def __radd__(self, other: ContentType):
        self.prepend(other)
        return self
    
    def __iadd__(self, other: ContentType):
        self.append(other)
        return self
    
    def __and__(self, other: ContentType):
        # self.append(other, sep="")
        self_copy = self.copy()
        self_copy.inline_append(self.promote_content(other))
        return self_copy
    
    def __rand__(self, other: ContentType):
        self.prepend(other, sep="")
        return self
    
    def __iand__(self, other: ContentType):
        self.append(other, sep="")
        return self
    
    # def __isub__(self, other: ContentType):
    #     self.
    
    
    
        
    
    
class Block(BlockBase):
    """
    Block is a tree node with structure and style.
    """
    
    
    def __init__(
        self,
        content: ContentType | None = None,
        children: list["BlockBase"] | None = None,
        role: str | None = None,
        style: str | list[str] | None = None,
        tags: list[str] | None = None,
        parent: "Block | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
        _skip_content: bool = False,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ):
        super().__init__(content, children=children, role=role, style=style, tags=tags, parent=parent, block_text=block_text, styles=styles, _skip_content=_skip_content, start_offset=start_offset, end_offset=end_offset)

 
        
    # def __call__(
    #     self, 
    #     content: ContentType | BlockBase | None = None, 
    #     role: str | None = None,
    #     tags: list[str] | None = None,
    #     style: str | None = None,
    # ) -> "Block":         
    #     block = self.promote_block_content(content, style=style, tags=tags, role=role)
    #     self.append_block_child(block)
    #     return block

    
    def model_copy(self, overrides: dict[str, Any] | None = None, copy_content: bool = False, fork_block_text: bool = False, copy_children: bool = True) -> "Block":
        block = Block(
            role=self.role,
            tags=self.tags,
            styles=self.styles,            
        )
        return block
    
    
    def __repr__(self) -> str:
        postfix = ", postfix=" + repr(self.postfix_span.text()) + " " if self.postfix_span else ""
        prefix = ", prefix=" + repr(self.prefix_span.text()) + " " if self.prefix_span else ""
        return f"""Block({prefix}content="{str(self.content_str)}", {postfix}children={len(self.children)}, role={self.role}, tags={self.tags}, styles={self.styles})"""




class BlockSchema(BlockBase):
    """
    BlockSchema is a block that is used to describe the schema of a block.
    """
    
    __slots__ = [
        "name",
        "type",
        "_transformer",
    ]
    
    def __init__(
        self,
        name: str | None = None,
        type: Type | None = None,
        children: list["BlockBase"] | None = None,   
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        parent: "BlockBase | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
        is_virtual: bool = False,
        _skip_content: bool = False,
        _transformer = None,
    ):
        tags = tags or []        
        if name is None:
            if len(tags) == 0:
                raise ValueError("you must provide a name or tags for the schema. empty names produce virtual schemas that are not rendered.")
            is_virtual = True
        else:
            if name not in tags:
                tags.insert(0, name)
        styles = styles or parse_style(style)
            
        super().__init__(
            content=name if not is_virtual else None, 
            children=children, 
            role=role, 
            style=style, 
            tags=tags, 
            parent=parent, 
            block_text=block_text, 
            styles=styles, 
            _skip_content=_skip_content
        )
        self.name = name if not is_virtual else tags[0]
        self.type = type
        self._transformer = _transformer
        
    
    def instantiate(
        self, 
        content: ContentType | dict | None = None,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET
    ) -> "Block":
        if isinstance(content, dict):
            return self.inst_from_dict(content)
        styles = (parse_style(style) or self.styles) if style is not UNSET and style is not None else None
        tags = (tags or self.tags) if tags is not UNSET and tags is not None else None
        role = role or self.role if role is not UNSET and role is not None else None
        # if isinstance(self.parent, BlockListSchema)
        return Block(
            content=content or (self.name if not role else None),
            # content=content,
            tags=tags,
            styles=styles,
            role=role or self.role,
        )
        
        
    def inst_from_dict(self, data: dict) -> "Block":
        from .block_builder import BlockBuilderContext
        from .pydantic_helpers import traverse_dict
        builder = BlockBuilderContext(self)
        builder.init_root()

        for k, v, path, label_path, action, field_type in traverse_dict(data):
            if action == "open":        
                if field_type == "list-item" or field_type == "model-list-item":
                    builder.instantiate_list_item(k, force_schema=True)
                else:
                    builder.instantiate(k, k, force_schema=True)
                # builder.append(v, force_schema=True)
                builder.curr_block.append_child(v)
            elif action == "close":
                builder.commit(k, force_schema=True)
            elif action == "open-close":
                builder.instantiate(k, k, force_schema=True)
                # builder.append(v, force_schema=True)
                builder.curr_block.append_child(v)
                builder.commit(k, force_schema=True)
                
        return builder.result

    
    @property
    def is_list_item(self) -> bool:
        return isinstance(self.parent, BlockListSchema)
    
    def get_item_name(self) -> str:
        if self.is_list_item:
            return self.parent.item_name
        return self.name

    # def extract_schema(self) -> "BlockSchema":
    #     """
    #     Extract a new BlockSchema tree containing only BlockSchema nodes.

    #     Traverses this block's subtree and creates a new BlockSchema with
    #     only BlockSchema children, preserving the schema hierarchy while
    #     filtering out regular Block nodes.

    #     Returns:
    #         A new BlockSchema tree with only schema nodes
    #     """
    #     # Create a copy of this schema (without children)
    #     # new_schema = self.__class__(
    #     #     name=self.name,
    #     #     type=self.type,
    #     #     role=self.role,
    #     #     styles=list(self.styles),
    #     #     tags=list(self.tags),
    #     # )
    #     new_schema = self.model_copy()

    #     # Recursively extract schemas from children
    #     for child in self.children:
    #         if isinstance(child, BlockSchema) or isinstance(child, BlockListSchema):
    #             # Recursively extract and add as child
    #             child_schema = child.extract_schema()
    #             new_schema.children.append(child_schema)
    #             child_schema.parent = new_schema
    #         else:
    #             # For regular blocks, check their children for nested schemas
    #             self._extract_nested_schemas(child, new_schema)

    #     return new_schema

    # def _extract_nested_schemas(self, block: "BlockBase", parent_schema: "BlockSchema"):
    #     """
    #     Recursively search a Block's children for BlockSchema nodes.

    #     Any found BlockSchema nodes are extracted and added to parent_schema.
    #     """
    #     for child in block.children:
    #         if isinstance(child, BlockSchema):
    #             child_schema = child.extract_schema()
    #             parent_schema.children.append(child_schema)
    #             child_schema.parent = parent_schema
    #         else:
    #             # Keep searching deeper
    #             self._extract_nested_schemas(child, parent_schema)
    
    def model_metadata_copy(self, overrides: dict[str, Any] | None = None) -> Self:
        dump = {
            "name": self.name,
            "type": self.type,
            "role": self.role,
            "tags": self.tags,
            "styles": self.styles,
        }
        if overrides:
            dump.update(overrides)
        return self.__class__(**dump)

    
    def model_copy(self, copy_content: bool = False, fork_block_text: bool = False, copy_children: bool = True) -> "BlockSchema":
        block = BlockSchema(
            name=self.name,
            type=self.type,
            role=self.role,
            tags=self.tags,
            styles=self.styles,
        )
        if copy_children:
            for child in self.children:
                block.append_child(child.model_copy(copy_content, fork_block_text, copy_children))
        return block
                
                
    def __repr__(self) -> str:
        return f"""BlockSchema(name="{self.name}", type={self.type}, children={len(self.children)}, role={self.role}, tags={self.tags}, styles={self.styles})"""
    
    
    
    
    
    
    
    
PropNameType = Literal["content", "prefix", "postfix", "role", "tags", "styles", "attrs"]



MutatorType = TypeVar("MutatorType")

class ListMutator[MutatorType]:
    
    def __init__(self, lst: list[Block]):
        self.lst = lst
        
    def __itruediv__(self, value: MutatorType):
        self.append(value)
        return self
    
    def __iand__(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")
        
        
    def append(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    def prepend(self, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    def replace(self, index: int, value: MutatorType):
        raise NotImplementedError("Subclass must implement this method")

    # def __getitem__(self, index: int) -> Block:
    #     return self.lst[index]

    # def __setitem__(self, index: int, value: Block):
    #     self.lst[index] = value
    
# class ListItemMutator(ListMutator):


class ContentPrefixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.content.prefix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prefix = value + blk.content.prefix

    def replace(self, index: int, value: str):
        self.lst[index].content.prefix = value
        return self


class ContentPostfixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.content.postfix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.postfix = value + blk.content.postfix
        return self

    def replace(self, index: int, value: str):
        self.lst[index].content.postfix = value
        return self
    
    
class BlockListPrefixMutator(ListMutator[str]):
    
    def append(self, value: str):
        for blk in self.lst:
            blk.prefix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.prefix = value + blk.prefix

    def replace(self, index: int, value: str):
        self.lst[index].prefix = value
        return self


class BlockListPostfixMutator(ListMutator[str]):
    def append(self, value: str):
        for blk in self.lst:
            blk.postfix += value
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.postfix = value + blk.postfix

    def replace(self, index: int, value: str):
        self.lst[index].postfix = value
        return self


class SentMutator(ListMutator):
    
    def append(self, value: str):
        for blk in self.lst:
            blk.content.append(value)
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prepend(value)
        return self
    

class BlockListContentMutator(ListMutator[str]):
    
    
    @property
    def prefix(self) -> ListMutator:
        return ContentPrefixMutator(self.lst)
    
    @prefix.setter
    def prefix(self, value: str):
        pass
    
    @property
    def postfix(self) -> ListMutator:
        return ContentPostfixMutator(self.lst)
    
    @postfix.setter
    def postfix(self, value: str):
        pass
    
    def append(self, value: str):
        for blk in self.lst:
            blk.content.append(value)
        return self

    def prepend(self, value: str):
        for blk in self.lst:
            blk.content.prepend(value)
        return self

    def replace(self, index: int, value: str):
        self.lst[index] = value
        return self



class ListEditor:
    
    def __init__(self, block_list: "BlockList"):
        self.block_list = block_list
    
    @property
    def content(self) -> BlockListContentMutator:
        return BlockListContentMutator(self.block_list)
    
    
    @content.setter
    def content(self, value: str):
        pass
    
    @property
    def prefix(self) -> BlockListPrefixMutator:
        return BlockListPrefixMutator(self.block_list)
    
    @prefix.setter
    def prefix(self, value: str):
        pass
    
    @property
    def postfix(self) -> BlockListPostfixMutator:
        return BlockListPostfixMutator(self.block_list)
    
    @postfix.setter
    def postfix(self, value: str):
        pass


class BlockList(Block):
    
    
    def __init__(
        self, 
        children: list[str] | list["Block"] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
        styles: list[str] | None = None,
        parent: "Block | None" = None,
        is_wrapper: bool = False,
        schema: "BlockSchema | None" = None,
        block_text: BlockText | None = None,
        _skip_content: bool = False,
    ):
        super().__init__(children=children, role=role, tags=tags, style=style, styles=styles, parent=parent, block_text=block_text, _skip_content=_skip_content)
    
    def edit(self) -> ListEditor:
        return ListEditor(self)
    
    def replace_all(self, prop: PropNameType, value: Any):
        for blk in self.children:
            setattr(blk, prop, value)
        return self
            
            
    # def prefix_append(self, value: str):
    #     for blk in self.children:
    #         blk.prefix += value
    #     return self
            
    # def postfix_append(self, value: str):
    #     for blk in self:
    #         blk.postfix += value
    #     return self        
    
    def values(self) -> Generator[Any, None, None]:
        for blk in self.children:
            yield blk.value

            
            
    def model_copy(self, copy_content: bool = False, fork_block_text: bool = False, copy_children: bool = True) -> "BlockList":
        block = BlockList(            
            role=self.role,
            tags=self.tags,
            styles=self.styles,
        )
        if copy_children:
            for child in self.children:
                block.append_child(child.model_copy(copy_content, fork_block_text, copy_children))
        return block
    
    
    def __getitem__(self, index: int) -> "BlockBase":
        return self.children[index]
    
    def __setitem__(self, index: int, value: "Block"):
        self.insert(index, value)
    
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def __repr__(self) -> str:        
        return f"BlockList({self.children})"
    
    
    
    
    
    
    
class BlockListSchema(BlockSchema):
    """
    BlockListSchema is a schema for a list of blocks.
    """
    __slots__ = [
        "item_name",
        "list_schemas",
        "list_models",
        "key",
    ]
    
    def __init__(
        self, 
        item_name: str,
        name: str | None = None,
        key: str | None = None,
        type: Type | None = None, 
        children: list["BlockSchema"] | None = None, 
        role: str | None = None, 
        tags: list[str] | None = None, 
        style: str | None = None, 
        parent: "BlockSchema | None" = None, 
        styles: list[str] | None = None, 
        block_text: BlockText | None = None, 
        _skip_content: bool = False
    ):
        tags = tags or []
        if not styles:
            styles = ["xml-list"]
        name = name or f"{item_name}_list"
        if name not in tags:
            tags.insert(0, name)
        super().__init__(None, type=type, children=children, role=role, tags=tags, style=style, parent=parent, styles=styles, block_text=block_text, _skip_content=_skip_content)
        self.name = name
        self.item_name = item_name
        self.tags.insert(0, name)
        self.list_schemas = []
        self.list_models = {}
        self.key = key
    
    
    def instantiate(
        self,
        content: ContentType | None = None,
        children: list["BlockSchema"] | None = None,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET
    ) -> "BlockList":
        return BlockList(
            # content or self.name,
            children=children,
            tags=self.tags if tags is not UNSET and tags is not None else None,
            styles=self.styles if style is not UNSET and style is not None else None,
            role=self.role,
        )
        
        
    def instantiate_item(
        self,
        value: Any,
        content: ContentType | None = None,
        style: str | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET
    ) -> "Block":
        blk = super().instantiate(value, style=style, role=role, tags=tags)
        return blk
    
    
    def register(self, target: BlockSchema | Type[BaseModel]):
        # if isinstance(target, BaseModel):
            # block = pydantic_object_description(target)
        from .pydantic_helpers import pydantic_to_block
        if isinstance(target, type) and issubclass(target, BaseModel):
            if self.key is None:
                raise ValueError("key_field is required")
            block = pydantic_to_block(self.item_name, target, self.key)
            self.list_models[target.__name__] = target
        elif isinstance(target, BlockSchema):
            block = target
        else:
            raise ValueError(f"Invalid target type: {type(target)}")
        self.list_schemas.append(block)
        self.append_child(block)
        return block
    
    def model_metadata_copy(self, overrides: dict[str, Any] | None = None) -> Self:
        dump = {
            "item_name": self.item_name,
            "name": self.name,
            "key": self.key,
            "type": self.type,
            "role": self.role,
            "tags": self.tags,
            "styles": self.styles,
        }
        if overrides:
            dump.update(overrides)
        return self.__class__(**dump)

    
    def model_copy(self, copy_content: bool = False, fork_block_text: bool = False, copy_children: bool = True) -> "BlockListSchema":
        block = BlockListSchema(
            item_name=self.item_name,
            name=self.name,
            key=self.key,
            type=self.type,
            role=self.role,
            tags=self.tags,
            styles=self.styles,
        )
        return block
        if copy_children:
            for child in self.children:
                block.append_child(child.model_copy(copy_content, fork_block_text, copy_children))
        return block
    
    
    def __getitem__(self, index: int) -> "BlockBase":
        return self.children[index]
    
    def __setitem__(self, index: int, value: "Block"):
        self.insert(index, value)

    def __repr__(self) -> str:
        return f"BlockListSchema(name={self.name}, key={self.key}, children={len(self.children)}, role={self.role}, tags={self.tags}, styles={self.styles})"