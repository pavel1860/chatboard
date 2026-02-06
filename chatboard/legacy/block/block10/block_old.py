"""
Block and BlockSchema - Structure layer for block10.

BlockBase provides common functionality for both schemas and instances.
BlockSchema is the template for parsing - defines structure constraints.
Block is the instance with actual content.

Design Principles:
- ALL text is stored in BlockText as chunks (preserves LLM logprobs)
- Block's first parameter is the text content (tag name/header)
- Block.children contain child blocks
- Style determines rendering: xml → <text>children</text>, markdown → # text\nchildren
- tags are for identification/fetching (separate from text content)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Type, TYPE_CHECKING, overload
from uuid import uuid4
from abc import ABC

from .chunk import BlockChunk, BlockText
from .span import Span, SpanAnchor, VirtualBlockText

if TYPE_CHECKING:
    pass


def _generate_id() -> str:
    """Generate a short unique ID for blocks."""
    return uuid4().hex[:8]


@dataclass
class AttrSchema:
    """Schema for a block attribute."""

    name: str
    type: Type = str
    description: str | None = None
    required: bool = False
    default: Any = None


@dataclass
class BlockBase(ABC):
    """
    Common functionality for both BlockSchema and Block.

    Holds:
    - Style for rendering format (xml, markdown, plain)
    - Attributes (key-value pairs)
    """

    style: str = "plain"
    attrs: dict[str, Any] = field(default_factory=dict)

    def _render_attrs(self) -> str:
        """Render attributes for XML style."""
        if not self.attrs:
            return ""
        parts = [f'{k}="{v}"' for k, v in self.attrs.items()]
        return " " + " ".join(parts)


@dataclass
class BlockSchema(BlockBase):
    """
    Template for block structure - defines constraints for parsing.

    BlockSchema defines:
    - What style is used for parsing/rendering
    - What attributes are allowed
    - What children schemas are allowed

    Example:
        response_schema = BlockSchema(
            name="response",
            style="xml",
            children=[
                BlockSchema(name="thinking", style="xml"),
                BlockSchema(name="answer", style="xml"),
            ]
        )

        # Create instance from schema
        block = response_schema.instantiate(source)
    """

    name: str = ""
    type: Type | None = None
    attr_schemas: dict[str, AttrSchema] = field(default_factory=dict)
    children: list[BlockSchema] = field(default_factory=list)
    is_wrapper: bool = False
    split_on_newlines: bool = True

    def instantiate(self, source: BlockText) -> Block:
        """
        Create a block instance from this schema.

        Args:
            source: BlockText to store chunks in

        Returns:
            New Block instance matching this schema
        """
        block = Block(
            text=self.name,
            tags=[self.name] if self.name else [],
            style=self.style,
            attrs=dict(self.attrs),
            schema=self,
            _source=source,
        )
        return block

    def get_child_schema(self, name: str) -> BlockSchema | None:
        """
        Get child schema by name.

        Args:
            name: Name of child schema

        Returns:
            Child schema or None if not found
        """
        for child in self.children:
            if child.name == name:
                return child
        return None

    def validate(self, block: Block) -> list[str]:
        """
        Validate a block against this schema.

        Args:
            block: Block to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required attributes
        for attr_name, attr_schema in self.attr_schemas.items():
            if attr_schema.required and attr_name not in block.attrs:
                errors.append(f"Missing required attribute: '{attr_name}'")

        # Check children against child schemas
        for child in block.children:
            # Find matching schema by checking if child has matching tag
            child_schema = None
            for tag in child.tags:
                child_schema = self.get_child_schema(tag)
                if child_schema:
                    break

            if child_schema is None and self.children:
                errors.append(f"Unexpected child with tags: {child.tags}")
            elif child_schema:
                errors.extend(child_schema.validate(child))

        return errors

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "style": self.style,
            "type": self.type.__name__ if self.type else None,
            "attrs": self.attrs,
            "attr_schemas": {
                k: {"name": v.name, "type": v.type.__name__, "required": v.required}
                for k, v in self.attr_schemas.items()
            },
            "children": [c.model_dump() for c in self.children],
            "is_wrapper": self.is_wrapper,
            "split_on_newlines": self.split_on_newlines,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> BlockSchema:
        """Deserialize from dictionary."""
        children = [cls.model_validate(c) for c in data.get("children", [])]
        return cls(
            name=data.get("name", ""),
            style=data.get("style", "plain"),
            attrs=data.get("attrs", {}),
            children=children,
            is_wrapper=data.get("is_wrapper", False),
            split_on_newlines=data.get("split_on_newlines", True),
        )


class Block(BlockBase):
    """
    Tree node with content stored in BlockText.

    Block is the main user-facing class. Key concepts:
    - First parameter (text) is the tag name/header content
    - children contain child blocks (where nested content lives)
    - tags are for identification/fetching
    - style determines rendering format

    Rendering:
    - xml: <{text}{attrs}>{children}</{text}>
    - markdown: {"#" * depth} {text}\n{children}\n
    - plain: {text}{children}

    Example (user-defined):
        block = Block("greeting", tags=["greeting", "intro"], style="xml")
        block.add_child(Block("Hello world"))

        print(block.render())  # "<greeting>Hello world</greeting>"

    Example (style switching):
        block.style = "markdown"
        print(block.render())  # "# greeting\nHello world\n"
    """

    def __init__(
        self,
        text: str | VirtualBlockText | None = None,
        *,
        tags: list[str] | None = None,
        style: str = "plain",
        attrs: dict[str, Any] | None = None,
        children: list[Block] | None = None,
        parent: Block | None = None,
        schema: BlockSchema | None = None,
        prefix_span: Span | None = None,
        postfix_span: Span | None = None,
        _source: BlockText | None = None,
        id: str | None = None,
    ):
        """
        Initialize a Block.

        Args:
            text: The text content (tag name/header). Can be str or VirtualBlockText.
            tags: Tags for identification/fetching
            style: Rendering style (xml, markdown, plain)
            attrs: Attributes for rendering
            children: Child blocks
            parent: Parent block
            schema: Schema this block was created from
            prefix_span: For parsed blocks, span to opening tag
            postfix_span: For parsed blocks, span to closing tag
            _source: Shared BlockText storage
            id: Unique identifier
        """
        self.style = style
        self.attrs = attrs or {}
        self.tags = tags or []
        self.children = children or []
        self.parent = parent
        self.schema = schema
        self.prefix_span = prefix_span
        self.postfix_span = postfix_span
        self._source = _source
        self.id = id or _generate_id()

        # Initialize text storage
        self._text: VirtualBlockText | None = None

        # Handle text parameter
        if text is not None:
            if isinstance(text, str):
                self.set_text(text)
            elif isinstance(text, VirtualBlockText):
                self._text = text
                if self._source is None:
                    self._source = text.source

        # Set parent references on children
        for child in self.children:
            child.parent = self

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def text(self) -> VirtualBlockText | None:
        """Get the text content as VirtualBlockText."""
        return self._text

    @text.setter
    def text(self, value: VirtualBlockText | None) -> None:
        """Set the text content."""
        self._text = value

    @property
    def tag(self) -> str | None:
        """Get primary tag (first in tags list). For quick access."""
        return self.tags[0] if self.tags else None

    @property
    def is_parsed(self) -> bool:
        """Check if this block was parsed (has prefix/postfix spans)."""
        return self.prefix_span is not None or self.postfix_span is not None

    @property
    def is_empty(self) -> bool:
        """Check if block has no text and no children."""
        text_empty = self._text is None or self._text.is_empty
        return text_empty and len(self.children) == 0

    @property
    def text_content(self) -> str:
        """Get text content as string (the tag name/header)."""
        if self._text is None:
            return ""
        return self._text.render()

    @property
    def depth(self) -> int:
        """Get depth in tree (root is 0)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    @property
    def root(self) -> Block:
        """Get root block of tree."""
        current = self
        while current.parent is not None:
            current = current.parent
        return current

    @property
    def path(self) -> list[int]:
        """Get path from root as list of child indices."""
        if self.parent is None:
            return []
        index = self.parent.children.index(self)
        return self.parent.path + [index]

    # -------------------------------------------------------------------------
    # Tag Operations
    # -------------------------------------------------------------------------

    def has_tag(self, tag: str) -> bool:
        """Check if block has a specific tag."""
        return tag in self.tags

    def add_tag(self, tag: str) -> None:
        """Add a tag to this block."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this block."""
        if tag in self.tags:
            self.tags.remove(tag)

    # -------------------------------------------------------------------------
    # Tree Operations
    # -------------------------------------------------------------------------

    def add_child(self, child: Block) -> Block:
        """
        Add a child block.

        Args:
            child: Block to add as child

        Returns:
            The added child
        """
        child.parent = self
        if child._source is None:
            child._source = self._source
        self.children.append(child)
        return child

    def insert_child(self, index: int, child: Block) -> Block:
        """
        Insert a child at specific index.

        Args:
            index: Position to insert at
            child: Block to insert

        Returns:
            The inserted child
        """
        child.parent = self
        if child._source is None:
            child._source = self._source
        self.children.insert(index, child)
        return child

    def remove_child(self, child: Block) -> Block:
        """
        Remove a child block.

        Args:
            child: Block to remove

        Returns:
            The removed child
        """
        child.parent = None
        self.children.remove(child)
        return child

    def get_child_by_tag(self, tag: str) -> Block | None:
        """
        Get first child that has the given tag.

        Args:
            tag: Tag to search for

        Returns:
            Child block or None
        """
        for child in self.children:
            if child.has_tag(tag):
                return child
        return None

    def get_all_by_tag(self, tag: str) -> list[Block]:
        """
        Get all descendants that have the given tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching blocks
        """
        result = []
        for block in self.traverse():
            if block.has_tag(tag):
                result.append(block)
        return result

    def traverse(self) -> Iterator[Block]:
        """
        Iterate over this block and all descendants (pre-order).

        Yields:
            This block, then each descendant
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    def traverse_path(self) -> Iterator[Block]:
        """
        Iterate from root to this block.

        Yields:
            Each ancestor from root, ending with self
        """
        if self.parent is not None:
            yield from self.parent.traverse_path()
        yield self

    # -------------------------------------------------------------------------
    # Text Operations (using BlockText)
    # -------------------------------------------------------------------------

    def _ensure_source(self) -> BlockText:
        """Ensure we have a source BlockText, creating one if needed."""
        if self._source is None:
            self._source = BlockText()
        return self._source

    def _ensure_text(self) -> VirtualBlockText:
        """Ensure we have a VirtualBlockText, creating one if needed."""
        if self._text is None:
            self._text = VirtualBlockText(self._ensure_source())
        return self._text

    def set_text(self, content: str) -> BlockChunk:
        """
        Set the text content (tag name/header) from a string.

        Creates a chunk in the source BlockText and sets up the VirtualBlockText.

        Args:
            content: The text to set

        Returns:
            The created Chunk
        """
        source = self._ensure_source()

        # Create chunk with the content
        chunk = BlockChunk(content=content)
        source.append(chunk)

        # Create VirtualBlockText with span covering the chunk
        span = Span(
            start=SpanAnchor(chunk, 0),
            end=SpanAnchor(chunk, len(content)),
        )
        self._text = VirtualBlockText(source, [span])

        return chunk

    def set_text_from_chunk(self, chunk: BlockChunk) -> None:
        """
        Set the text content from an existing chunk.

        Used when parsing LLM output where chunks already exist.

        Args:
            chunk: Existing chunk to use
        """
        if chunk._owner is None:
            raise ValueError("Chunk must belong to a BlockText")

        self._source = chunk._owner
        span = Span(
            start=SpanAnchor(chunk, 0),
            end=SpanAnchor(chunk, len(chunk.content)),
        )
        self._text = VirtualBlockText(self._source, [span])

    def set_text_from_span(self, span: Span, source: BlockText) -> None:
        """
        Set the text content from a span.

        Used when parsing LLM output where spans point into existing chunks.

        Args:
            span: Span covering the text
            source: The BlockText containing the chunks
        """
        self._source = source
        self._text = VirtualBlockText(source, [span])

    def append_text(self, content: str) -> BlockChunk:
        """
        Append text to end of this block's text content.

        Args:
            content: Text to append

        Returns:
            The created chunk
        """
        source = self._ensure_source()
        text = self._ensure_text()

        new_chunk = BlockChunk(content=content)

        if text.spans:
            # Insert after last chunk of last span
            last_span = text.spans[-1]
            source.insert_after(last_span.end.chunk, new_chunk)
        else:
            # No existing content
            source.append(new_chunk)

        # Add span for new chunk
        new_span = Span(
            start=SpanAnchor(new_chunk, 0),
            end=SpanAnchor(new_chunk, len(content)),
        )
        text.append_span(new_span)

        return new_chunk

    def prepend_text(self, content: str) -> BlockChunk:
        """
        Prepend text to beginning of this block's text content.

        Args:
            content: Text to prepend

        Returns:
            The created chunk
        """
        source = self._ensure_source()
        text = self._ensure_text()

        new_chunk = BlockChunk(content=content)

        if text.spans:
            # Insert before first chunk of first span
            first_span = text.spans[0]
            source.insert_before(first_span.start.chunk, new_chunk)
        else:
            # No existing content
            source.append(new_chunk)

        # Add span for new chunk
        new_span = Span(
            start=SpanAnchor(new_chunk, 0),
            end=SpanAnchor(new_chunk, len(content)),
        )
        text.prepend_span(new_span)

        return new_chunk

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render_prefix(self) -> str:
        """
        Render prefix based on style.

        Uses self.text_content as the tag name.

        Returns:
            Prefix string (e.g., "<tagname>" for XML)
        """
        tag_name = self.text_content
        if not tag_name:
            return ""

        if self.style == "xml":
            attrs_str = self._render_attrs()
            return f"<{tag_name}{attrs_str}>"
        elif self.style in ("markdown", "md"):
            return "#" * (self.depth + 1) + " " + tag_name + "\n"
        elif self.style == "plain":
            return ""
        return ""

    def render_postfix(self) -> str:
        """
        Render postfix based on style.

        Uses self.text_content as the tag name.

        Returns:
            Postfix string (e.g., "</tagname>" for XML)
        """
        tag_name = self.text_content
        if not tag_name:
            return ""

        if self.style == "xml":
            return f"</{tag_name}>"
        elif self.style in ("markdown", "md"):
            return "\n"
        elif self.style == "plain":
            return ""
        return ""

    def render(self) -> str:
        """
        Render block to string.

        For parsed blocks: uses prefix_span/postfix_span
        For user blocks: computes prefix/postfix from style using text as tag name

        Leaf blocks (no children) with plain style: text IS the content
        Non-leaf blocks or styled blocks: text is the tag name

        Returns:
            Rendered string
        """
        result = []

        # Leaf block with plain style: text IS the content, not a tag
        if not self.children and self.style == "plain" and self._text:
            return self.text_content

        # Prefix
        if self.prefix_span:
            result.append(self.prefix_span.text())
        elif self.text_content:
            result.append(self.render_prefix())

        # Children (this is where content lives)
        for child in self.children:
            result.append(child.render())

        # Postfix
        if self.postfix_span:
            result.append(self.postfix_span.text())
        elif self.text_content:
            result.append(self.render_postfix())

        return "".join(result)

    def render_content_only(self) -> str:
        """
        Render only content, without prefix/postfix.

        Returns:
            Content string
        """
        result = []

        # If this block has no children, its text IS the content
        if not self.children and self._text:
            result.append(self.text_content)

        # Render children
        for child in self.children:
            result.append(child.render_content_only())

        return "".join(result)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "tags": self.tags,
            "text": self.text_content,
            "style": self.style,
            "attrs": self.attrs,
            "children": [c.model_dump() for c in self.children],
            "is_parsed": self.is_parsed,
        }

    @classmethod
    def model_validate(cls, data: dict[str, Any], source: BlockText | None = None) -> Block:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary representation
            source: Optional BlockText to use (creates new one if None)

        Returns:
            Block instance
        """
        if source is None:
            source = BlockText()

        block = cls(
            text=data.get("text"),
            id=data.get("id"),
            tags=data.get("tags", []),
            style=data.get("style", "plain"),
            attrs=data.get("attrs", {}),
            _source=source,
        )

        # Recursively create children
        for child_data in data.get("children", []):
            child = cls.model_validate(child_data, source)
            block.add_child(child)

        return block

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        tags_str = ",".join(self.tags) if self.tags else "no-tags"
        text_preview = self.text_content[:20] + "..." if len(self.text_content) > 20 else self.text_content
        return f'Block[{tags_str}](text="{text_preview}", children={len(self.children)})'

    def repr_tree(self, indent: int = 0) -> str:
        """
        Get tree representation.

        Args:
            indent: Current indentation level

        Returns:
            Tree string
        """
        prefix = "  " * indent
        tags_str = ",".join(self.tags) if self.tags else "no-tags"
        text = self.text_content[:30] + "..." if len(self.text_content) > 30 else self.text_content
        text = text.replace("\n", "\\n")

        lines = [f'{prefix}Block[{tags_str}] style={self.style} text="{text}"']
        for child in self.children:
            lines.append(child.repr_tree(indent + 1))
        return "\n".join(lines)

    def print_tree(self) -> None:
        """Print tree representation."""
        print(self.repr_tree())
