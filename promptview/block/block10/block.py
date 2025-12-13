from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Type, TYPE_CHECKING, overload
from uuid import uuid4
from abc import ABC

from .chunk import Chunk, BlockText
from .span import Span, SpanAnchor, VirtualBlockText
from .path import Path

if TYPE_CHECKING:
    pass


# Type alias for chunk mapping during copy
ChunkMap = dict[str, Chunk]  # old_chunk_id -> new_chunk


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

ContentType = str | list[str] | list[Chunk] | Chunk
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
        "parent", 
        "_block_text", 
        "_span", 
        "children",
        "_postfix_span",
        "_prefix_span",
        "role",
    ]
    
    def __init__(
        self,
        content: ContentType | None = None,
        children: list["BlockBase"] | None = None,
        role: str | None = None,
        style: str | None = None,
        tags: list[str] | None = None,
        parent: "BlockBase | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
        _skip_content: bool = False,
    ):
        self.styles = parse_style(style) if style is not None else styles or []
        self.role = role
        self.tags = tags or []
        self.parent = parent
        self._block_text = block_text or BlockText()
        self.children = children or []
        self._postfix_span: Span | None = None
        self._prefix_span: Span | None = None

        if _skip_content:
            # For copy operations - span will be set by caller
            self._span = None
        else:
            content = self.promote_content(content)
            chunks = self._block_text.extend(content)
            self._span = Span.from_chunks(chunks)
    
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
    def content(self) -> "BlockBase":
        """
        Return this block's content as a new independent Block (no children/style/tags).

        Creates a new BlockText with copies of the content chunks, preserving
        logprob metadata. The returned block is fully independent and can be
        used with operators like & without affecting the original.

        Returns:
            A new Block with copied content chunks
        """
        # Fork just the content span's chunks
        new_block_text = self._block_text.fork(
            start=self._span.start.chunk,
            end=self._span.end.chunk
        )

        # Create block with the new BlockText
        block = Block(
            block_text=new_block_text,
            _skip_content=True,
        )
        block._span = Span.from_chunks(list(new_block_text))
        return block

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
    def content_end_chunk(self) -> Chunk:
        return self._span.end.chunk

    @property
    def content_start_chunk(self) -> Chunk:
        return self._span.start.chunk
    
    @property
    def end_postfix_chunk(self) -> Chunk | None:
        if self._postfix_span is None:
            return self.content_end_chunk
        return self._postfix_span.end.chunk
    
    @property
    def start_prefix_chunk(self) -> Chunk | None:
        if self._prefix_span is None:
            return self.content_start_chunk
        return self._prefix_span.start.chunk

    # -------------------------------------------------------------------------
    # Boundary properties (for entire subtree)
    # -------------------------------------------------------------------------

    @property
    def start_chunk(self) -> Chunk | None:
        """
        Get the first chunk of this block's entire subtree.

        Returns the earliest chunk considering:
        - prefix span (if present)
        - content span
        """
        if self.start_prefix_chunk is not None:
            return self.start_prefix_chunk
        else:
            return self.content_start_chunk

    @property
    def end_chunk(self) -> Chunk:
        """
        Get the last chunk of this block's entire subtree.

        Returns the latest chunk considering:
        - postfix span of deepest last child
        - or this block's postfix/content span if no children
        """
        if self.children:
            # Recursively get boundary_end of the last child
            return self.children[-1].end_chunk
        elif self.end_postfix_chunk is not None:
            # No children - return this block's end
            return self.end_postfix_chunk
        else:
            return self.content_end_chunk

    def get_boundaries(self) -> tuple[Chunk | None, Chunk | None]:
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

    def get_chunks(self) -> list[Chunk]:
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
    
    
    def set_block_text(self, chunks: list[Chunk], block_text: BlockText):
        self._block_text = block_text
        self._span = Span.from_chunks(chunks)
        return self


    # -------------------------------------------------------------------------
    # Path properties
    # -------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """
        Compute current path dynamically.

        Returns a Path object with both index-based and tag-based paths.
        """
        indices = []
        tags = []

        # Walk up from this block to root, collecting indices and tags
        current = self
        while current.parent is not None:
            # Get index of current in parent's children
            idx = current.parent.children.index(current)
            indices.append(idx)

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


    def promote_content(self, content: "ContentType | None") -> list[Chunk]:
        if content is None:
            return [Chunk(content="")]
        if isinstance(content, str):
            return [Chunk(content)]
        elif isinstance(content, list):
            return content
        elif isinstance(content, Chunk):
            return [content]
        else:
            raise ValueError(f"Invalid content type: {type(content)}")
        return content
    
    def promote_block_content(self, content: BlockBase |ContentType | None, style: str | None = None, tags: list[str] | None = None, role: str | None = None) -> "BlockBase":
        if isinstance(content, BlockBase):
            return content.copy()
        if content is None:
            return Block(parent=self, block_text=self._block_text, style=style, tags=tags, role=role)
        else:
            content = self.promote_content(content)
            return Block(content=content, block_text=self._block_text, style=style, tags=tags, role=role)    
        
    def _append_separator(self, content: list[Chunk], sep: str | None, append: bool = True):
        if sep:
            if not content[-1].is_line_end:
                if append:
                    content = [Chunk(content=sep)] + content
                else:
                    content = content + [Chunk(content=sep)]
        return content
    
    def append(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=True)
        chunks = self._block_text.extend(content, after=self.content_end_chunk)
        self._span.end = SpanAnchor(chunk=chunks[-1], offset=len(chunks[-1].content))
        
    def prepend(self, content: ContentType, sep: str | None = " "):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=False)
        chunks = self._block_text.left_extend(content, before=self.content_start_chunk)
        self._span.start = SpanAnchor(chunk=chunks[0], offset=0)
        
    def postfix_append(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=True)
        chunks = self._block_text.extend(content, after=self.end_postfix_chunk)
        self._postfix_span = Span.from_chunks(chunks)
        
    def prefix_prepend(self, content: ContentType, sep: str | None = ""):
        content = self.promote_content(content)
        content = self._append_separator(content, sep, append=False)
        chunks = self._block_text.left_extend(content, before=self.start_prefix_chunk)
        self._prefix_span = Span.from_chunks(chunks)
        
        
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
            inserted_chunks = self._block_text.left_extend_block_text(
                block._block_text,
                before=insert_before_chunk,
                copy=False
            )
        elif parent.children:
            # Append after last child
            insert_after_chunk = parent.children[-1].end_chunk
            inserted_chunks = self._block_text.extend_block_text(
                block._block_text,
                after=insert_after_chunk,
                copy=False
            )
        else:
            # No children yet, insert after parent's content
            insert_after_chunk = parent.end_chunk
            inserted_chunks = self._block_text.extend_block_text(
                block._block_text,
                after=insert_after_chunk,
                copy=False
            )

        # Remap the block's span to the new chunks in our BlockText
        if inserted_chunks:
            block._block_text = self._block_text
            block._span = Span.from_chunks(inserted_chunks)

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
            replacement = other.copy() if other._block_text is not self._block_text else other
        else:
            replacement = Block(content=other)

        # Get target's chunk boundaries
        target_start = target.start_chunk
        target_end = target.end_chunk

        # Replace chunks in BlockText
        removed_chunks, inserted_chunks = self._block_text.replace_block_text(
            target_start,
            target_end,
            replacement._block_text,
            copy=False
        )

        # Remap replacement's span to the newly inserted chunks
        if inserted_chunks:
            replacement._block_text = self._block_text
            replacement._span = Span.from_chunks(inserted_chunks)

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
        self._block_text.replace(child.start_chunk, child.end_chunk, None)

        # Remove from tree
        self.children.pop(index)
        child.parent = None

        return child
        
    
    def append_child(self, child_content: Block | ContentType):
        """
        Append a child block to this block's children.

        Inserts chunks at the correct position in the BlockText (after this
        block's last child, or after this block's content if no children).

        Args:
            child_content: Block or content to append as child

        Returns:
            The appended block
        """
        # Create the block with its own temporary BlockText
        if isinstance(child_content, BlockBase):
            block = child_content.copy()
        else:
            block = Block(content=child_content)

        # Add newline separator
        if self.children:
            self.last_descendant.add_new_line()
        else:
            self.add_new_line()

        # Determine insertion point - after last child or after this block's content
        if self.children:
            insert_after_chunk = self.children[-1].end_chunk
        else:
            insert_after_chunk = self.end_postfix_chunk or self.content_end_chunk

        # Move chunks from block's BlockText to this block's BlockText
        inserted_chunks = self._block_text.extend_block_text(
            block._block_text,
            after=insert_after_chunk,
            copy=False
        )

        # Remap the block's span and update BlockText reference for entire subtree
        if inserted_chunks:
            self._remap_block_text(block, self._block_text)

        # Add to children list
        self.children.append(block)
        block.parent = self
        return block

    def _remap_block_text(self, block: "BlockBase", new_block_text: BlockText):
        """
        Recursively update _block_text reference for a block and all its descendants.

        Used after moving chunks between BlockTexts to ensure all blocks
        in a subtree point to the correct BlockText.
        """
        block._block_text = new_block_text
        for child in block.children:
            self._remap_block_text(child, new_block_text)

    def append_block_child(self, block: "Block"):
        """
        Append an already-prepared block to children list.

        This is a low-level method used by _copy_tree. It only updates
        the tree structure, not the BlockText. Use append_child() for
        normal operations.
        """
        self.children.append(block)
        block.parent = self
        return block

    
    def transform(self) -> "BlockBase":
        from .block_transformers import transform
        block = self.copy()
        block = transform(block)
        return block

    def render(self) -> str:
        block = self.transform()
        return block._block_text.text()

    
    def copy_metadata(self) -> "BlockBase":
        return Block(
            styles=list(self.styles),
            tags=list(self.tags),
            block_text=self._block_text,
            role=self.role,
        )


    def copy(self) -> "BlockBase":
        """
        Create a deep copy of this block and its subtree.

        Copies the underlying BlockText and rebuilds spans to point to new chunks.
        """
        # Copy the BlockText (creates new chunks with same content)
        start, end = self.get_boundaries()
        new_block_text = self._block_text.fork(start, end)

        # Build chunk mapping: old_id -> new_chunk
        # Only iterate over chunks within our boundaries
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
        new_block = Block(
            styles=list(self.styles),
            tags=list(self.tags),
            role=self.role,
            block_text=new_block_text,
            _skip_content=True,
        )

        # Remap spans
        new_block._span = Span(
            start=SpanAnchor(chunk_map[self._span.start.chunk.id], self._span.start.offset),
            end=SpanAnchor(chunk_map[self._span.end.chunk.id], self._span.end.offset),
        )

        if self._prefix_span:
            new_block._prefix_span = Span(
                start=SpanAnchor(chunk_map[self._prefix_span.start.chunk.id], self._prefix_span.start.offset),
                end=SpanAnchor(chunk_map[self._prefix_span.end.chunk.id], self._prefix_span.end.offset),
            )

        if self._postfix_span:
            new_block._postfix_span = Span(
                start=SpanAnchor(chunk_map[self._postfix_span.start.chunk.id], self._postfix_span.start.offset),
                end=SpanAnchor(chunk_map[self._postfix_span.end.chunk.id], self._postfix_span.end.offset),
            )

        # Copy children
        for child in self.children:
            new_block.append_block_child(child._copy_tree(chunk_map, new_block_text))

        return new_block
    

    def traverse(self) -> Iterator["BlockBase"]:
        yield self
        for child in self.children:
            yield from child.traverse()
            
            
    # -------------------------------------------------------------------------
    # Text operations
    # -------------------------------------------------------------------------
    
    def add_new_line(self):
        self.postfix_append(Chunk(content="\n"))
        
        
        
    
    
    
    
    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------
    
    def print(self):
        print(self.render())
        
        
    def __itruediv__(self, other: Block | ContentType):
        self.append_child(other)
        return self
    
    def __add__(self, other: ContentType):
        self.append(other)
        return self
    
    def __radd__(self, other: ContentType):
        self.prepend(other)
        return self
    
    def __iadd__(self, other: ContentType):
        self.append(other)
        return self
    
    def __and__(self, other: ContentType):
        self.append(other, sep="")
        return self
    
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
        style: str | None = None,
        tags: list[str] | None = None,
        parent: "Block | None" = None,
        styles: list[str] | None = None,
        block_text: BlockText | None = None,
        _skip_content: bool = False,
    ):
        super().__init__(content, children=children, role=role, style=style, tags=tags, parent=parent, block_text=block_text, styles=styles, _skip_content=_skip_content)

 
        
    def __call__(
        self, 
        content: ContentType | BlockBase | None = None, 
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | None = None,
    ) -> "Block":         
        block = self.promote_block_content(content, style=style, tags=tags, role=role)
        self.append_block_child(block)
        return block
