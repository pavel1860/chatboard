from __future__ import annotations
from typing import TYPE_CHECKING, Type, Any

from .block import Block, Mutator, ContentType, parse_style
from promptview.utils.type_utils import UnsetType, UNSET
from pydantic import BaseModel
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
        name: str | None = None,
        *,
        type: Type | None = None,
        style: str | None = None,
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
        if name and name not in tags:
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
    @property
    def is_wrapper(self) -> bool:
        return self.name is None

    def instantiate(
        self,
        content: ContentType | dict | BaseModel | None = None,
        name: ContentType | None = None,
        style: str | list[str] | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
        extract_schema: bool = True,
    ) -> Block:
        """
        Create a Block instance from this schema.

        Delegates to mutator.instantiate() for style-specific behavior.

        Args:
            content: Content for the new block (default: schema name)
            style: Override style for the block
            role: Override role for the block
            tags: Override tags for the block

        Returns:
            A new Block instance
        """
        from .mutator_meta import MutatorMeta
        if extract_schema:
            schema = self.extract_schema()
            if schema is None:
                raise ValueError("No schema found")
            return schema.instantiate(content, name=name, style=style, role=role, tags=tags, extract_schema=False)
        
        role = UnsetType.get_value(role, self.role)
        tags = UnsetType.get_value(tags, self.tags)
        
        style = UnsetType.get_value(style, self.style)

        if isinstance(content, dict):
            return self.inst_from_dict(content, style=style, role=role, tags=tags)
        elif isinstance(content, BaseModel):
            return self.inst_from_dict(content.model_dump(), style=style, role=role, tags=tags)
        else:
            config = MutatorMeta.resolve(self.style)
            if config.mutator is None:
                raise ValueError(f"No mutator found for style {style}")
            mutator = config.mutator()
            block = mutator.call_instantiate(name or self.name, role=role, tags=tags, style=style)
            if content is not None:
                block.add_newline()
                block.append(content)
            return block

    def inst_from_dict(
        self,
        data: dict,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
    ) -> Block:
        """
        Create a Block instance from a dictionary.

        Matches dict keys to child schema names and recursively instantiates
        child schemas with corresponding values.

        Args:
            data: Dictionary with keys matching child schema names
            style: Style for the block
            role: Role for the block
            tags: Tags for the block

        Returns:
            A new Block instance with instantiated children
        """
        from .mutator_meta import MutatorMeta

        # Resolve style and get mutator
        style = style if style is not None else self.style
        role = role if role is not None else self.role
        tags = tags if tags is not None else self.tags
        # config = MutatorMeta.resolve(style)
        # config = MutatorMeta.resolve([])
        # if config.mutator is None:
        #     raise ValueError(f"No mutator found for style {style}")
        # mutator = config.mutator()

        # Create the parent block with schema name as content (for XML tag rendering)        
        # block = mutator.call_instantiate(self.name, role=role, tags=tags)
        block = self.instantiate(role=role, tags=tags, style=style)

        # Build lookup of child schemas by name
        child_schemas: dict[str, BlockSchema] = {}
        for child in self.body:
            if isinstance(child, BlockSchema):
                child_schemas[child.name] = child

        # Track which fields were provided
        provided_fields = set()

        # Instantiate children in schema order (predictable structure)
        for child_schema in self.body:
            if not isinstance(child_schema, BlockSchema):
                continue

            if child_schema.name in data:
                provided_fields.add(child_schema.name)
                value = data[child_schema.name]

                # Recursively instantiate child
                if isinstance(value, dict) and child_schema.children:
                    # Nested dict with nested schema
                    child_block = child_schema.inst_from_dict(value, style=style)
                elif isinstance(value, list) and isinstance(child_schema, BlockListSchema):
                    # List value for list schema
                    child_block = child_schema.inst_from_list(value, style=style)
                else:
                    # Scalar value
                    child_block = child_schema.instantiate(value, style=style)
                    # child_block.append(value)

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


class BlockList(Block):
    """
    A list container block for holding multiple items.

    BlockList is the instantiated form of BlockListSchema.
    It provides list-like access to its children.
    """

    __slots__ = ["item_schema"]

    def __init__(
        self,
        *,
        item_schema: "BlockSchema | None" = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
        _children: list[Block] | None = None,
    ):
        """
        Create a block list.

        Args:
            item_schema: Optional schema for list items
            role: Role identifier
            tags: Tags for categorization
            style: Style for rendering
        """
        super().__init__(
            content=None,  # List is a wrapper - no content
            role=role,
            tags=tags,
            style=style,
            mutator=mutator,
            block_text=block_text,
            _children=_children,
        )
        self.item_schema = item_schema

    # -------------------------------------------------------------------------
    # List Operations
    # -------------------------------------------------------------------------

    def __iter__(self):
        """Iterate over list items."""
        return iter(self.children)

    def __getitem__(self, index: int) -> Block:
        """Get item by index."""
        return self.children[index]

    def __len__(self) -> int:
        """Number of items."""
        return len(self.children)

    def append_item(self, content: ContentType | Block) -> Block:
        """
        Append an item to the list.

        If content is a Block, appends directly.
        If content is ContentType, creates a new block with the item_schema.
        """
        if isinstance(content, Block):
            item = content
        elif self.item_schema is not None:
            item = self.item_schema.instantiate(content)
        else:
            item = Block(content=content)

        self.append_child(item)
        return item

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self, deep: bool = True) -> "BlockList":
        """Copy this list."""
        from .block_text import BlockText

        if deep:
            new_block_text = BlockText()
            new_span = self.span.copy() if self.span else None
            if new_span:
                new_block_text.append(new_span)

            new_list = BlockList(
                item_schema=self.item_schema.copy() if self.item_schema else None,
                role=self.role,
                tags=self.tags.copy() if self.tags else [],
                style=self._style.copy() if self._style else [],
                block_text=new_block_text,
                _children=[],
            )

            for child in self.children:
                child_copy = child.copy(deep=True)
                new_list.append_child(child_copy)

            return new_list
        else:
            return BlockList(
                item_schema=self.item_schema,
                role=self.role,
                tags=self.tags,
                style=self._style,
                block_text=self.block_text,
                _children=self.children.copy(),
            )

    def __repr__(self) -> str:
        parts = [f"items={len(self.children)}"]
        if self.item_schema:
            parts.append(f"item_schema={self.item_schema.name!r}")
        if self.role:
            parts.append(f"role={self.role!r}")
        return f"BlockList({', '.join(parts)})"





class BlockListSchema(BlockSchema):
    """
    Schema for a list of blocks.

    BlockListSchema defines the structure for lists, including:
    - The name/tag for the list container
    - The item_name for individual items
    - Optional key field for item lookup

    Example:
        # Define list schema
        tools_schema = BlockListSchema("tool", name="tools")
        tools_schema /= BlockSchema("description")

        # Render shows structure:
        # <tools>
        #   <tool>
        #     <description></description>
        #   </tool>
        # </tools>

        # Instantiate creates empty list
        tools = tools_schema.instantiate()
        tools.append_item("hammer")
    """

    __slots__ = ["item_name", "key", "list_schemas", "list_models"]

    def __init__(
        self,
        item_name: str,
        *,
        name: str | None = None,
        key: str | None = None,
        type: Type | None = None,
        style: str | list[str] | None = None,
        tags: list[str] | None = None,
        role: str | None = None,
        is_required: bool = True,
        attrs: dict[str, Any] | None = None,
        # Internal
        mutator: Mutator | None = None,
        block_text: "BlockText | None" = None,
        _span=None,
        _children: list[Block] | None = None,
    ):
        """
        Create a list schema.

        Args:
            item_name: Name/tag for individual items
            name: Name/tag for the list container (defaults to item_name)
            key: Optional key field for item lookup
            type: Optional type for validation
            style: Style for rendering (default: "xml-list")
            tags: Additional tags
            role: Role identifier
            is_required: Whether this list is required
            attrs: Additional attributes
        """
        # Use item_name as container name if not specified
        list_name = name or item_name

        # Initialize BlockSchema with list_name as content
        super().__init__(
            name=list_name,
            type=type,
            style=style,
            tags=tags,
            role=role,
            is_required=is_required,
            attrs=attrs,
            mutator=mutator,
            block_text=block_text,
            _span=_span,
            _children=_children,
        )

        # List-specific attributes
        self.item_name = item_name
        self.key = key
        self.list_schemas: list[BlockSchema] = []
        self.list_models: dict[str, Type[BaseModel]] = {}
    # -------------------------------------------------------------------------
    # Schema Registration
    # -------------------------------------------------------------------------

    def register(self, target: BlockSchema | Type[BaseModel]):
        # if isinstance(target, BaseModel):
            # block = pydantic_object_description(target)
        from .object_helpers import pydantic_to_schema
        if isinstance(target, type) and issubclass(target, BaseModel):
            if self.key is None:
                raise ValueError("key_field is required")
            block = pydantic_to_schema(self.item_name, target, self.key)
            self.list_models[target.__name__] = target
        elif isinstance(target, BlockSchema):
            block = target
        else:
            raise ValueError(f"Invalid target type: {type(target)}")
        self.list_schemas.append(block)
        self.append_child(block)
        return block

    def get_item_schema(self, key_value: str | None = None) -> BlockSchema | None:
        """
        Get an item schema, optionally by key value.

        If key_value is provided and key is set, looks up by key.
        Otherwise returns the first registered schema.
        """
        if key_value is not None and self.key is not None:
            for schema in self.list_schemas:
                if schema.attrs.get(self.key) == key_value:
                    return schema
        return self.list_schemas[0] if self.list_schemas else None

    # -------------------------------------------------------------------------
    # Instantiation
    # -------------------------------------------------------------------------

    def instantiate(
        self,
        content: ContentType | None = None,
        style: str | list[str] | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
    ) -> BlockList:
        """
        Create a BlockList instance from this schema.

        Returns an empty BlockList with item_schema set for appending items.
        """
        # Get the first item schema if registered
        item_schema = self.list_schemas[0] if self.list_schemas else None

        return BlockList(
            item_schema=item_schema,
            role=UnsetType.get_value(role, self.role),
            tags=UnsetType.get_value(tags, self.tags),
            style=UnsetType.get_value(style, self._style),
        )

    def instantiate_item(
        self,
        content: ContentType | None = None,
        key_value: str | None = None,
        style: str | list[str] | None | UnsetType = UNSET,
        role: str | None | UnsetType = UNSET,
        tags: list[str] | None | UnsetType = UNSET,
    ) -> Block:
        """
        Instantiate a single item from the item schema.

        Args:
            content: Content for the item
            key_value: Optional key value for schema lookup
            style, role, tags: Override schema defaults
        """
        item_schema = self.get_item_schema(key_value)

        if item_schema is not None:
            return item_schema.instantiate(content, style=style, role=role, tags=tags)
        else:
            # No item schema - create a simple block with item_name as tag
            return Block(
                content=content,
                role=UnsetType.get_value(role, self.role),
                tags=[self.item_name] + (UnsetType.get_value(tags, []) or []),
                style=UnsetType.get_value(style, self._style),
            )

    def inst_from_list(
        self,
        data: list,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
    ) -> BlockList:
        """
        Create a BlockList instance from a list of values.

        Each item in the list is instantiated using the item schema.
        Items can be:
        - Scalar values (str, int, etc.) - instantiated directly
        - Dicts - instantiated using inst_from_dict on item schema
        - BaseModel instances - converted to dict and instantiated

        Args:
            data: List of values to instantiate as items
            style: Style for the list
            role: Role for the list
            tags: Tags for the list

        Returns:
            A BlockList with instantiated items
        """
        # Create the list container
        block_list = self.instantiate(
            style=style if style is not None else UNSET,
            role=role if role is not None else UNSET,
            tags=tags if tags is not None else UNSET,
        )

        # Get item schema for instantiation
        item_schema = self.get_item_schema()

        for item in data:
            if isinstance(item, dict):
                if item_schema is not None and item_schema.children:
                    # Dict with nested schema
                    item_block = item_schema.inst_from_dict(item)
                elif item_schema is not None:
                    # Dict but no nested schema - use instantiate
                    item_block = item_schema.instantiate(item)
                else:
                    # No item schema - create block with item_name tag
                    item_block = Block(
                        content=None,
                        tags=[self.item_name],
                        style=self._style,
                    )
                    # Add dict values as children
                    for k, v in item.items():
                        child = Block(content=v, tags=[k])
                        item_block.append_child(child)
            elif isinstance(item, BaseModel):
                if item_schema is not None:
                    item_block = item_schema.inst_from_dict(item.model_dump())
                else:
                    item_block = Block(
                        content=None,
                        tags=[self.item_name],
                        style=self._style,
                    )
            else:
                # Scalar value
                if item_schema is not None:
                    item_block = item_schema.instantiate(item)
                else:
                    item_block = Block(
                        content=item,
                        tags=[self.item_name],
                        style=self._style,
                    )

            block_list.append_item(item_block)

        return block_list

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy_head(self) -> "BlockListSchema":
        """Copy only the list schema's span (head) without children."""
        from .block_text import BlockText

        new_block_text = BlockText()
        new_span = self.span.copy() if self.span else None
        if new_span:
            new_block_text.append(new_span)

        return BlockListSchema(
            item_name=self.item_name,
            name=self.name,
            key=self.key,
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

    def copy(self, deep: bool = True) -> "BlockListSchema":
        """Copy this list schema."""
        from .block_text import BlockText

        if deep:
            new_block_text = BlockText()
            new_span = self.span.copy() if self.span else None
            if new_span:
                new_block_text.append(new_span)

            new_schema = BlockListSchema(
                item_name=self.item_name,
                name=self.name,
                key=self.key,
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

            # Deep copy children and rebuild list_schemas
            for child in self.children:
                child_copy = child.copy(deep=True)
                new_schema.append_child(child_copy)
                if isinstance(child_copy, BlockSchema):
                    new_schema.list_schemas.append(child_copy)

            return new_schema
        else:
            schema = BlockListSchema(
                item_name=self.item_name,
                name=self.name,
                key=self.key,
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
            schema.list_schemas = self.list_schemas.copy()
            return schema

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"item_name={self.item_name!r}"]
        if self.name != self.item_name:
            parts.append(f"name={self.name!r}")
        if self.key:
            parts.append(f"key={self.key!r}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        if not self.is_required:
            parts.append("optional")
        return f"BlockListSchema({', '.join(parts)})"
