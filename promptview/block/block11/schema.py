from __future__ import annotations
from typing import TYPE_CHECKING, Type, Any

from .block import Block, Mutator, ContentType, parse_style
from promptview.utils.type_utils import UnsetType, UNSET

if TYPE_CHECKING:
    from .block_text import BlockText


class BlockSchema(Block):
    """
    A block template that can be rendered (to show field structure)
    and instantiated (to create content blocks).

    Schema content = field name (for rendering)
    Block content = actual value (after instantiate)

    Example:
        # Define schema
        response_schema = BlockSchema("response", style="xml")
        response_schema /= BlockSchema("thinking")
        response_schema /= BlockSchema("answer")

        # Render schema (shows structure)
        # <response>
        #   <thinking></thinking>
        #   <answer></answer>
        # </response>

        # Instantiate with content
        block = response_schema.instantiate("Hello")
        # <response>Hello</response>
    """

    __slots__ = ["name", "type", "is_required", "attrs"]

    def __init__(
        self,
        name: str,
        *,
        type: Type | None = None,
        style: str | None = "xml",
        tags: list[str] | None = None,
        role: str | None = None,
        is_required: bool = True,
        attrs: dict[str, Any] | None = None,
        # Internal
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
        _span = None,
        _children: list[Block] | None = None,
    ):
        """
        Create a block schema.

        Args:
            name: The schema/field name (becomes content for rendering)
            type: Optional type annotation for validation
            style: Style for rendering (default: "xml")
            tags: Additional tags (name is auto-added as first tag)
            role: Role identifier
            is_required: Whether this field is required
            attrs: Additional attributes (e.g., for XML attributes)
        """
        # Prepare tags - name should be first tag
        tags = list(tags) if tags else []
        if name not in tags:
            tags.insert(0, name)

        # Initialize Block with name as content
        super().__init__(
            content=name,
            role=role,
            tags=tags,
            style=style,
            mutator=mutator,
            block_text=block_text,
            _span=_span,
            _children=_children,
        )

        # Schema-specific attributes
        self.name = name
        self.type = type
        self.is_required = is_required
        self.attrs = attrs or {}

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def instantiate(
        self, 
        content: ContentType | None = None,
        style: str | list[str] | None | UnsetType = UNSET, 
        role: str | None | UnsetType = UNSET, 
        tags: list[str] | None | UnsetType = UNSET
    ) -> Block:
        """
        Create a Block instance from this schema.

        Delegates to mutator.instantiate() for style-specific behavior.

        Args:
            content: Content for the new block (default: schema name)
            **kwargs: Additional arguments passed to mutator

        Returns:
            A new Block instance
        """
        from .mutator_meta import MutatorMeta
        config = MutatorMeta.resolve(self.style)
        if config.mutator is None:
            raise ValueError(f"No mutator found for style {UnsetType.get_value(style, self.style)}")
        mutator = config.mutator()
        block = mutator.call_instantiate(content, role=UnsetType.get_value(role, self.role), tags=UnsetType.get_value(tags, self.tags), style=UnsetType.get_value(style, self.style))
        return block
    
    

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self, deep: bool = True) -> BlockSchema:
        """
        Copy this schema.

        Args:
            deep: If True, recursively copy children
        """
        from .block_text import BlockText

        if deep:
            new_block_text = BlockText()
            new_span = self.span.copy() if self.span else None
            if new_span:
                new_block_text.append(new_span)

            new_schema = BlockSchema(
                name=self.name,
                type=self.type,
                style=self._style.copy() if self._style else [],
                tags=self.tags.copy() if self.tags else [],
                role=self.role,
                is_required=self.is_required,
                attrs=self.attrs.copy() if self.attrs else {},
                block_text=new_block_text,
                _span=new_span,
                _children=[],
            )

            # Deep copy children
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_schema.append_child(child_copy)

            return new_schema
        else:
            return BlockSchema(
                name=self.name,
                type=self.type,
                style=self._style,
                tags=self.tags,
                role=self.role,
                is_required=self.is_required,
                attrs=self.attrs,
                block_text=self.block_text,
                _span=self.span,
                _children=self.children.copy(),
            )

    def copy_head(self) -> BlockSchema:
        """Copy only the schema's span (head) without children."""
        from .block_text import BlockText

        new_block_text = BlockText()
        new_span = self.span.copy() if self.span else None
        if new_span:
            new_block_text.append(new_span)

        return BlockSchema(
            name=self.name,
            type=self.type,
            style=self._style.copy() if self._style else [],
            tags=self.tags.copy() if self.tags else [],
            role=self.role,
            is_required=self.is_required,
            attrs=self.attrs.copy() if self.attrs else {},
            block_text=new_block_text,
            _span=new_span,
            _children=[],
        )

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self.type:
            parts.append(f"type={self.type.__name__}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        if self.role:
            parts.append(f"role={self.role!r}")
        if not self.is_required:
            parts.append("optional")
        return f"BlockSchema({', '.join(parts)})"
