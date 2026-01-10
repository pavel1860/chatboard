"""
BlockSchema - Template for creating blocks with structure.

BlockSchema defines a block template that can be rendered (to show structure)
and instantiated (to create content blocks).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Type, Any

from .block import Block, ContentType, _parse_style

if TYPE_CHECKING:
    pass


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

    __slots__ = ["name", "_type", "is_required"]

    def __init__(
        self,
        name: str | None = None,
        *,
        type: Type | None = None,
        style: str | list[str] | None = None,
        tags: list[str] | None = None,
        role: str | None = None,
        is_required: bool = True,
        attrs: dict[str, Any] | None = None,
    ):
        """
        Create a block schema.

        Args:
            name: The schema/field name (becomes content for rendering)
            type: Optional type annotation for validation
            style: Style for rendering (e.g., "xml")
            tags: Additional tags (name is auto-added as first tag)
            role: Role identifier
            is_required: Whether this field is required
            attrs: Additional attributes (e.g., for XML attributes)
        """
        # Prepare tags - name should be first tag
        tags = list(tags) if tags else []
        if name and name not in tags:
            tags.insert(0, name)

        # Initialize Block with name as content
        super().__init__(
            content=name,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
        )

        # Schema-specific attributes
        self.name = name
        self._type = type
        self.is_required = is_required

    # -------------------------------------------------------------------------
    # Schema Properties
    # -------------------------------------------------------------------------

    @property
    def is_wrapper(self) -> bool:
        """True if this is a wrapper schema (no name)."""
        return self.name is None

    @property
    def type(self) -> Type | None:
        """Get the type annotation for this schema."""
        return self._type

    # -------------------------------------------------------------------------
    # Instantiation
    # -------------------------------------------------------------------------

    def instantiate(
        self,
        content: ContentType | dict | None = None,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Block:
        """
        Create a Block instance from this schema.

        Args:
            content: Content for the block (str, dict, or None)
            style: Override schema style
            role: Override schema role
            tags: Override schema tags
            attrs: Override schema attrs

        Returns:
            A new Block instance
        """
        if isinstance(content, dict):
            return self._inst_from_dict(
                content,
                style=style,
                role=role,
                tags=tags,
                attrs=attrs,
            )
        else:
            return self._inst_content(
                content,
                style=style,
                role=role,
                tags=tags,
                attrs=attrs,
            )

    def _inst_content(
        self,
        content: ContentType | None = None,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Block:
        """
        Create a Block with content.

        Args:
            content: Content for the block
            style: Override schema style
            role: Override schema role
            tags: Override schema tags
            attrs: Override schema attrs

        Returns:
            A new Block instance
        """
        # Use schema values as defaults
        block = Block(
            content=self.name,
            role=role if role is not None else self.role,
            tags=tags if tags is not None else list(self.tags),
            style=style if style is not None else list(self.style),
            attrs=attrs if attrs is not None else dict(self.attrs),
        )

        # Append content if provided
        if content is not None:
            if isinstance(content, Block):
                block._raw_append(content.text)
            elif isinstance(content, str):
                block._raw_append(content)
            else:
                block._raw_append(str(content))

        return block

    def _inst_from_dict(
        self,
        data: dict,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Block:
        """
        Create a Block instance from a dictionary.

        Matches dict keys to child schema names and recursively instantiates
        child schemas with corresponding values.

        Args:
            data: Dictionary with keys matching child schema names
            style: Override schema style
            role: Override schema role
            tags: Override schema tags
            attrs: Override schema attrs

        Returns:
            A new Block instance with instantiated children
        """
        # Create the parent block with schema name as content
        block = self._inst_content(
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
        )

        # Build lookup of child schemas by name
        child_schemas: dict[str, BlockSchema] = {}
        for child in self.children:
            if isinstance(child, BlockSchema):
                child_schemas[child.name] = child

        # Track which fields were provided
        provided_fields = set()

        # Instantiate children in schema order (predictable structure)
        for child_schema in self.children:
            if not isinstance(child_schema, BlockSchema):
                continue

            if child_schema.name in data:
                provided_fields.add(child_schema.name)
                value = data[child_schema.name]

                # Recursively instantiate child
                if isinstance(value, dict) and child_schema.children:
                    # Nested dict with nested schema
                    child_block = child_schema._inst_from_dict(value, style=style)
                else:
                    # Scalar value
                    child_block = child_schema._inst_content(value, style=style)

                block.append_child(child_block)
            elif child_schema.is_required:
                raise ValueError(f"Required field '{child_schema.name}' not provided")

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
        new_schema = BlockSchema(
            name=self.name,
            type=self._type,
            style=list(self.style),
            tags=list(self.tags),
            role=self.role,
            is_required=self.is_required,
            attrs=dict(self.attrs),
        )

        if deep:
            # Deep copy children
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_schema._raw_append_child(child_copy)

        return new_schema

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def model_dump(self, include_text: bool = True) -> dict[str, Any]:
        """
        Serialize BlockSchema to a JSON-compatible dict.

        Includes schema-specific fields: name, type, is_required.
        """
        # Get base block fields
        result = super().model_dump(include_text=include_text)

        # Add schema-specific fields
        result["name"] = self.name
        if self._type is not None:
            result["type"] = self._type.__name__
        if not self.is_required:
            result["is_required"] = self.is_required

        return result

    @classmethod
    def model_load(cls, data: dict[str, Any]) -> BlockSchema:
        """
        Deserialize a dict to a BlockSchema.

        Args:
            data: Dict from model_dump() or JSON

        Returns:
            Reconstructed BlockSchema
        """
        from .chunk import ChunkMeta, _generate_id

        # Extract schema-specific fields
        name = data.get("name")
        is_required = data.get("is_required", True)

        # Create BlockSchema
        schema = cls(
            name=name,
            type=None,  # Type reconstruction would need a type registry
            style=data.get("style", []),
            tags=data.get("tags", []),
            role=data.get("role"),
            is_required=is_required,
            attrs=data.get("attrs", {}),
        )

        # Restore position info
        schema.id = data.get("id", schema.id)
        schema.start = data.get("start", 0)
        schema.end = data.get("end", 0)
        schema._text = data.get("_text", "")

        # Restore chunks
        schema.chunks = [
            ChunkMeta(
                id=c.get("id", _generate_id()),
                start=c["start"],
                end=c["end"],
                logprob=c.get("logprob"),
                style=c.get("style"),
            )
            for c in data.get("chunks", [])
        ]

        # Recursively load children
        for child_data in data.get("children", []):
            # Check if child is a schema or regular block
            if child_data.get("name") is not None:
                child = BlockSchema.model_load(child_data)
            else:
                child = Block.model_load(child_data)
            child.parent = schema
            schema.children.append(child)

        return schema

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}"]
        if self._type:
            parts.append(f"type={self._type.__name__}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        if self.role:
            parts.append(f"role={self.role!r}")
        if not self.is_required:
            parts.append("optional")
        return f"BlockSchema({', '.join(parts)})"
