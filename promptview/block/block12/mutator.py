"""
Mutator - Strategy for style-aware block operations.

Mutators intercept block operations (append, prepend, append_child) and
modify behavior based on the block's style. The base Mutator provides
direct pass-through. Subclasses like XmlMutator add prefix/postfix handling.

MutatorMeta is a metaclass that registers mutators by their style names.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from .block import Block
    from .chunk import ChunkMeta


# Global registry of style -> mutator class
_style_registry: dict[str, type[Mutator]] = {}


class MutatorMeta(type):
    """
    Metaclass that registers Mutator subclasses by their styles.

    When a class with `styles = ["xml", ...]` is defined, this metaclass
    registers it in the global style registry for lookup.
    """

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        new_cls = super().__new__(mcs, name, bases, attrs)

        # Register by styles if defined
        if styles := attrs.get("styles"):
            for style in styles:
                _style_registry[style] = new_cls

        return new_cls

    @classmethod
    def get_mutator(mcs, style: str) -> type[Mutator]:
        """
        Get the Mutator class for a given style.

        Returns the base Mutator if no specific mutator is registered.
        """
        return _style_registry.get(style, Mutator)

    @classmethod
    def list_styles(mcs) -> list[str]:
        """List all registered styles."""
        return list(_style_registry.keys())


class Mutator(metaclass=MutatorMeta):
    """
    Base mutator providing direct pass-through to block operations.

    Mutators wrap a block and intercept operations like append, prepend,
    and append_child. The base Mutator simply delegates to the block's
    raw operations without modification.

    Subclasses override methods to provide style-specific behavior:
    - XmlMutator adds prefix/content/postfix region handling
    - Other mutators can implement different transformation logic

    Attributes:
        block: The block this mutator operates on
        styles: Class attribute listing styles this mutator handles
    """

    styles: tuple[str, ...] = ()

    def __init__(self, block: Block):
        """
        Initialize mutator with a block.

        Args:
            block: The block to wrap
        """
        self._block = block

    @property
    def block(self) -> Block:
        """Get the wrapped block."""
        return self._block

    # =========================================================================
    # Content Operations
    # =========================================================================

    def append(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Append content to the block.

        Base implementation appends to end. Subclasses may insert before
        postfix regions or perform other transformations.
        """
        return self.block._raw_append(content, logprob=logprob, style=style)

    def prepend(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Prepend content to the block.

        Base implementation prepends to start. Subclasses may insert after
        prefix regions or perform other transformations.
        """
        return self.block._raw_prepend(content, logprob=logprob, style=style)

    def insert(
        self,
        rel_position: int,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Insert content at a relative position.

        Used by subclasses for style-aware insertion.
        """
        return self.block._raw_insert(rel_position, content, logprob=logprob, style=style)

    # =========================================================================
    # Child Operations
    # =========================================================================

    def append_child(
        self,
        child: Block | None = None,
    ) -> Block:
        """
        Append a child block.

        Base implementation appends at end. Subclasses may insert before
        postfix or perform other transformations.
        """
        return self.block._raw_append_child(child=child)

    def prepend_child(
        self,
        child: Block | None = None,
    ) -> Block:
        """
        Prepend a child block.

        Base implementation prepends at start. Subclasses may insert after
        prefix or perform other transformations.
        """
        return self.block._raw_prepend_child(child=child)

    def insert_child(
        self,
        index: int,
        child: Block | None = None,
    ) -> Block:
        """
        Insert a child block at index.
        """
        return self.block._raw_insert_child(index, child=child)

    # =========================================================================
    # Region Queries
    # =========================================================================

    def get_content_region(self) -> tuple[int, int]:
        """
        Get the content region (start, end) relative to block.

        Base implementation returns full block range.
        Subclasses may exclude prefix/postfix regions.
        """
        return (0, self.block.length)

    def get_prefix_region(self) -> tuple[int, int] | None:
        """
        Get the prefix region if any.

        Base implementation returns None (no prefix).
        """
        return None

    def get_postfix_region(self) -> tuple[int, int] | None:
        """
        Get the postfix region if any.

        Base implementation returns None (no postfix).
        """
        return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def init(
        self,
        content: str | None = None,
        attrs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize block structure.

        Called when block is first created with this style.
        Subclasses may add prefix/postfix chunks.
        """
        if content:
            self.block._raw_append(content)

    def commit(self, add_newline: bool = True) -> None:
        """
        Finalize block structure.

        Called when block is being closed/committed.
        Subclasses may add closing tags, etc.
        """
        pass

    # =========================================================================
    # Debug
    # =========================================================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(block={self.block.id})"


class XmlMutator(Mutator):
    """
    Mutator for XML-style blocks with opening/closing tags.

    Manages three regions:
    - prefix: Opening tag (e.g., "<tag>")
    - content: Body content (user appends go here)
    - postfix: Closing tag (e.g., "</tag>")

    Content operations insert into the content region, preserving
    the prefix and postfix structure.

    Chunk styles:
    - "xml-open": Opening tag characters
    - "xml-close": Closing tag characters
    - "content": Default content region
    """

    styles = ("xml",)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def init(
        self,
        content: str | None = None,
        attrs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize XML structure with opening tag.

        Args:
            content: Tag name (e.g., "example" creates <example>)
            attrs: Optional XML attributes
        """
        if not content:
            return

        tag_name = content

        # Build attribute string
        attr_str = ""
        if attrs:
            parts = [f'{k}="{v}"' for k, v in attrs.items()]
            attr_str = " " + " ".join(parts)

        # Create opening tag as prefix
        opening_tag = f"<{tag_name}{attr_str}>"
        self.block._raw_append(opening_tag, style="xml-open")

        # Store tag name for commit
        self.block.attrs["_xml_tag"] = tag_name

    def commit(self, add_newline: bool = True) -> None:
        """
        Add closing tag.
        """
        tag_name = self.block.attrs.get("_xml_tag")
        if not tag_name:
            return

        # Optionally add newline before closing tag
        if add_newline and self.block.length > 0:
            # Check if block already ends with newline
            if not self.block.text.endswith("\n"):
                self.block._raw_append("\n", style="xml-content")

        # Add closing tag
        closing_tag = f"</{tag_name}>"
        self.block._raw_append(closing_tag, style="xml-close")

    # =========================================================================
    # Region Queries
    # =========================================================================

    def get_prefix_region(self) -> tuple[int, int] | None:
        """Get the opening tag region."""
        chunks = self.block.get_chunks_by_style("xml-open")
        if not chunks:
            return None
        # Combine all prefix chunks
        start = min(c.start for c in chunks)
        end = max(c.end for c in chunks)
        return (start, end)

    def get_postfix_region(self) -> tuple[int, int] | None:
        """Get the closing tag region."""
        chunks = self.block.get_chunks_by_style("xml-close")
        if not chunks:
            return None
        start = min(c.start for c in chunks)
        end = max(c.end for c in chunks)
        return (start, end)

    def get_content_region(self) -> tuple[int, int]:
        """Get the content region (between opening and closing tags)."""
        prefix = self.get_prefix_region()
        postfix = self.get_postfix_region()

        start = prefix[1] if prefix else 0
        end = postfix[0] if postfix else self.block.length

        return (start, end)

    # =========================================================================
    # Content Operations (style-aware)
    # =========================================================================

    def append(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Append content before the closing tag (postfix).
        """
        postfix = self.get_postfix_region()

        if postfix:
            # Insert before postfix
            insert_pos = postfix[0]
            return self.block._raw_insert(
                insert_pos,
                content,
                logprob=logprob,
                style=style or "xml-content"
            )
        else:
            # No postfix yet, append normally
            return self.block._raw_append(
                content,
                logprob=logprob,
                style=style or "xml-content"
            )

    def prepend(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Prepend content after the opening tag (prefix).
        """
        prefix = self.get_prefix_region()

        if prefix:
            # Insert after prefix
            insert_pos = prefix[1]
            return self.block._raw_insert(
                insert_pos,
                content,
                logprob=logprob,
                style=style or "xml-content"
            )
        else:
            # No prefix, prepend normally
            return self.block._raw_prepend(
                content,
                logprob=logprob,
                style=style or "xml-content"
            )

    # =========================================================================
    # Child Operations
    # =========================================================================

    def append_child(
        self,
        child: Block | None = None,
    ) -> Block:
        """
        Append child, adding newline after opening tag if first child.
        """
        from .block import Block as BlockClass

        # If this is first child, add newline after opening tag
        if not self.block.children:
            prefix = self.get_prefix_region()
            if prefix:
                # Check if there's already a newline after prefix
                content_start = prefix[1]
                block_text = self.block.text
                if content_start < len(block_text) and block_text[content_start] != "\n":
                    self.block._raw_insert(content_start, "\n", style="xml-content")

        # Create child if needed
        if child is None:
            child = BlockClass()

        # Append child
        return self.block._raw_append_child(child)


class MarkdownMutator(Mutator):
    """
    Mutator for Markdown-style blocks with heading prefix.

    Adds heading prefix based on block depth.
    """

    styles = ("md", "markdown")

    def init(
        self,
        content: str | None = None,
        attrs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize with heading prefix based on depth.
        """
        # Add heading prefix
        depth = self.block.depth + 1
        prefix = "#" * depth + " "
        self.block._raw_append(prefix, style="md-prefix")

        # Add content if provided
        if content:
            self.block._raw_append(content, style="md-content")

    def get_prefix_region(self) -> tuple[int, int] | None:
        """Get the heading prefix region."""
        chunks = self.block.get_chunks_by_style("md-prefix")
        if not chunks:
            return None
        start = min(c.start for c in chunks)
        end = max(c.end for c in chunks)
        return (start, end)

    def prepend(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Prepend after heading prefix.
        """
        prefix = self.get_prefix_region()

        if prefix:
            insert_pos = prefix[1]
            return self.block._raw_insert(
                insert_pos,
                content,
                logprob=logprob,
                style=style or "md-content"
            )
        else:
            return self.block._raw_prepend(
                content,
                logprob=logprob,
                style=style or "md-content"
            )


class ListMutator(Mutator):
    """
    Mutator for list items with bullet prefix.
    """

    styles = ("list", "li")

    def init(
        self,
        content: str | None = None,
        attrs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize with bullet prefix.
        """
        bullet = attrs.get("bullet", "* ") if attrs else "* "
        self.block._raw_append(bullet, style="list-prefix")

        if content:
            self.block._raw_append(content, style="list-content")

    def get_prefix_region(self) -> tuple[int, int] | None:
        """Get the bullet prefix region."""
        chunks = self.block.get_chunks_by_style("list-prefix")
        if not chunks:
            return None
        start = min(c.start for c in chunks)
        end = max(c.end for c in chunks)
        return (start, end)

    def prepend(
        self,
        content: str,
        logprob: float | None = None,
        style: str | None = None
    ) -> ChunkMeta:
        """
        Prepend after bullet prefix.
        """
        prefix = self.get_prefix_region()

        if prefix:
            insert_pos = prefix[1]
            return self.block._raw_insert(
                insert_pos,
                content,
                logprob=logprob,
                style=style or "list-content"
            )
        else:
            return self.block._raw_prepend(
                content,
                logprob=logprob,
                style=style or "list-content"
            )
