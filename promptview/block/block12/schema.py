"""
BlockSchema - Template for creating blocks with structure.

BlockSchema defines a block template that can be rendered (to show structure)
and instantiated (to create content blocks).

BlockList - A list container block for holding multiple items.
BlockListSchema - Schema for defining list structures with item schemas.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Type, Any

from pydantic import BaseModel

from .block import Block, ContentType, _parse_style, _generate_id
from .chunk import ChunkMeta, BlockChunk

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

    __slots__ = ["name", "_type", "is_required", "is_root", "is_virtual"]

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
        is_root: bool = False,
        is_virtual: bool = False,
        id: str | None = None,
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
            id: Optional unique identifier (auto-generated if not provided)
        """
        # Prepare tags - name should be first tag
        tags = list(tags) if tags else []
        if name and name not in tags:
            tags.insert(0, name)
            
        style = style or "xml" if not is_root else "block"

        # Initialize Block with name as content
        super().__init__(
            content=name if not is_root and not is_virtual else None,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
            id=id,
        )

        # Schema-specific attributes
        self.name = name
        self._type = type
        self.is_required = is_required
        self.is_root = is_root
        self.is_virtual = is_virtual
        
    # -------------------------------------------------------------------------
    # Schema Properties
    # -------------------------------------------------------------------------

    @property
    def is_wrapper(self) -> bool:
        """True if this is a wrapper schema (no name)."""
        return self.name is None or self.is_root or self.is_virtual

    @property
    def type(self) -> Type | None:
        """Get the type annotation for this schema."""
        return self._type
    
    
    @property
    def type_str(self) -> str | None:
        from ...utils.type_utils import type_to_str
        return type_to_str(self.type) if self.type is not None else None

    # -------------------------------------------------------------------------
    # Instantiation
    # -------------------------------------------------------------------------
    
    def init_partial(self, content: ContentType, use_mutator: bool = True, is_streaming: bool = False):
        from .mutator import MutatorMeta
        from .mutator import BlockMutator
        content = content if use_mutator else self.name
        config = MutatorMeta.resolve(self.style if use_mutator else None, default=BlockMutator)
        if isinstance(content, Block):
            tran_block = config.build_block(content, is_streaming=is_streaming)
        else:
            tran_block = config.create_block(content, tags=self.tags, role=self._role, style=self.style, attrs=self.attrs, is_streaming=is_streaming, type=self._type)
        
        return tran_block


    def instantiate(
        self,
        content: ContentType | dict | BaseModel | None = None,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Block:
        """
        Create a Block instance from this schema.

        Args:
            content: Content for the block (str, dict, BaseModel, or None)
            style: Override schema style
            role: Override schema role
            tags: Override schema tags
            attrs: Override schema attrs

        Returns:
            A new Block instance
        """
        schema = self.extract_schema()
        if schema is None:
            raise ValueError("Schema is not supported for instantiation")
        if isinstance(content, dict):
            return schema._inst_from_dict(
                content,
                style=style,
                role=role,
                tags=tags,
                attrs=attrs,
            )
        elif isinstance(content, BaseModel):
            return schema._inst_from_dict(
                content.model_dump(),
                style=style,
                role=role,
                tags=tags,
                attrs=attrs,
            )
        else:
            return schema._inst_content(
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
        # Name becomes the block's text (used for XML tag)
        block = Block(
            content=self.name,
            role=role if role is not None else self._role,
            tags=tags if tags is not None else list(self.tags),
            style=style if style is not None else list(self.style),
            attrs=attrs if attrs is not None else dict(self.attrs),
            type=self._type,
        )

        # Add content as a child block (not appended to same block's text)
        # This way: name -> tag, content -> child inside tag
        if content is not None:
            if isinstance(content, Block):
                block._raw_append_child(content.copy())
            else:
                # Pass content with its actual type preserved
                block._raw_append_child(Block(content=content))

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
                    child_block = child_schema._inst_from_dict(value, style=style)
                elif isinstance(value, list) and isinstance(child_schema, BlockListSchema):
                    # List value for list schema
                    child_block = child_schema._inst_from_list(value, style=style)
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

    def copy(self, deep: bool = True, copy_id: bool = False) -> BlockSchema:
        """
        Copy this schema.

        Args:
            deep: If True, recursively copy children
            copy_id: If True, copy the id from this schema. If False, generate a new id.
        """
        new_schema = BlockSchema(
            name=self.name,
            type=self._type,
            style=list(self.style),
            tags=list(self.tags),
            role=self._role,
            is_required=self.is_required,
            attrs=dict(self.attrs),
            is_root=self.is_root,
            id=self.id if copy_id else None,
        )

        if deep:
            # Deep copy children
            for child in self.children:
                child_copy = child.copy(deep=True, copy_id=copy_id)
                new_schema._raw_append_child(child_copy)

        return new_schema

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def model_dump(self) -> dict[str, Any]:
        """
        Serialize BlockSchema to a JSON-compatible dict.

        Includes schema-specific fields: name, type, is_required.
        """
        from ...utils.type_utils import type_to_str
        # Get base block fields
        result = super().model_dump()

        # Add schema-specific fields
        result["name"] = self.name
        if self._type is not None:
            # result["type"] = self._type.__name__
            result["type"] = type_to_str(self._type)
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
        from ...utils.type_utils import str_to_type
        # Extract schema-specific fields
        name = data.get("name")
        is_required = data.get("is_required", True)

        # Create BlockSchema
        schema = cls(
            name=name,
            type=str_to_type(data.get("type"), False) if data.get("type") else None,  # Type reconstruction would need a type registry
            style=data.get("style", []),
            tags=data.get("tags", []),
            role=data.get("role"),
            is_required=is_required,
            attrs=data.get("attrs", {}),
            is_root=data.get("is_root", False),
        )
        
        schema = cls._load_mutator(schema, data)

        # Restore ID and text
        schema.id = data.get("id", schema.id)
        schema._text = data.get("text", "")

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

        # Recursively load children - Block.model_load handles class dispatch
        for child_data in data.get("children", []):
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


class BlockList(Block):
    """
    A list container block for holding multiple items.

    BlockList is the instantiated form of BlockListSchema.
    It provides list-like access to its children.
    """

    __slots__ = ["item_schema"]

    def __init__(
        self,
        children: list[Block] | None = None,
        *,
        item_schema: BlockSchema | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        style: str | list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        id: str | None = None,
    ):
        """
        Create a block list.

        Args:
            item_schema: Optional schema for list items
            role: Role identifier
            tags: Tags for categorization
            style: Style for rendering
            attrs: Additional attributes
            id: Optional unique identifier (auto-generated if not provided)
        """
        super().__init__(
            content=None,  # List is a wrapper - no content
            children=children,
            role=role,
            tags=tags,
            style=style,
            attrs=attrs,
            id=id,
        )
        self.item_schema = item_schema

    # -------------------------------------------------------------------------
    # List Operations
    # -------------------------------------------------------------------------

    def __iter__(self):
        """Iterate over list items."""
        return iter(self.body)

    def __len__(self) -> int:
        """Number of items."""
        return len(self.body)

    def append_item(self, content: ContentType | Block | BaseModel) -> Block:
        """
        Append an item to the list.

        If content is a Block, appends directly.
        If content is ContentType, creates a new block with the item_schema.
        If content is a BaseModel, converts to dict and instantiates.
        """
        if isinstance(content, Block):
            item = content
        elif isinstance(content, BaseModel):
            if self.item_schema is not None:
                item = self.item_schema.instantiate(content)
            else:
                # No schema - create block from model dump
                item = Block(content=str(content.model_dump()))
        elif self.item_schema is not None:
            item = self.item_schema.instantiate(content)
        else:
            item = Block(content=content)

        self.append_child(item)
        return item
    
    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------
    def apply(self, func: Callable[[Block], Block]) -> "BlockList":
        return BlockList(children=[func(b) for b in self.body], item_schema=self.item_schema, role=self.role, tags=self.tags, style=self.style, attrs=self.attrs)

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self, deep: bool = True, copy_id: bool = False) -> BlockList:
        """Copy this list.

        Args:
            deep: If True, recursively copy children
            copy_id: If True, copy the id from this list. If False, generate a new id.
        """
        new_list = BlockList(
            item_schema=self.item_schema.copy(copy_id=copy_id) if self.item_schema else None,
            role=self._role,
            tags=list(self.tags),
            style=list(self.style),
            attrs=dict(self.attrs),
            id=self.id if copy_id else None,
        )

        if deep:
            for child in self.children:
                child_copy = child.copy(deep=True, copy_id=copy_id)
                new_list._raw_append_child(child_copy)

        return new_list

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    @classmethod
    def model_load(cls, data: dict[str, Any]) -> BlockList:
        """Deserialize dict to BlockList."""
        block_list = cls(
            role=data.get("role"),
            tags=data.get("tags", []),
            style=data.get("style", []),
            attrs=data.get("attrs", {}),
        )

        block_list = cls._load_mutator(block_list, data)

        # Restore ID and text
        block_list.id = data.get("id", block_list.id)
        block_list._text = data.get("text", "")

        # Restore chunks
        block_list.chunks = [
            ChunkMeta(
                id=c.get("id", _generate_id()),
                start=c["start"],
                end=c["end"],
                logprob=c.get("logprob"),
                style=c.get("style"),
            )
            for c in data.get("chunks", [])
        ]

        # Recursively load children - Block.model_load handles class dispatch
        for child_data in data.get("children", []):
            child = Block.model_load(child_data)
            child.parent = block_list
            block_list.children.append(child)

        return block_list

    def __repr__(self) -> str:
        import textwrap
        # parts = [f"items={len(self.body)}"]
        children = [str(b) for b in self.body]
        children_str = '\n'.join(children)
        if len(children) > 1:
            children_str = f"\n{children_str}\n"
            children_str = textwrap.indent(children_str, "  ")
        parts = [f"[{children_str}]"]
        if self.item_schema:
            parts.append(f"item_schema={self.item_schema.name!r}")
        if self.role:
            parts.append(f"role={self.role!r}")
        return f"BlockList({', '.join(parts)})"

    
    def reverse(self) -> "BlockList":
        return BlockList(children=list(reversed(self.body)), item_schema=self.item_schema, role=self.role, tags=self.tags, style=self.style, attrs=self.attrs)
    
    def render_list(self, delimiter: str = "") -> str:        
        delimiter = f"\n{delimiter}\n" if delimiter else "\n\n"
        return delimiter.join([b.render() for b in self.body])
    
    
    def print_list_debug(self, metadata: bool = True, reverse: bool = True):
        target = reversed(self.body) if reverse else self.body
        for b in target:
            if metadata:
                # print(f"=============== [ {b.role} message ] | style: {b.style} | mutator: {b.mutator.__class__.__name__}\n")
                print(f"=============== [ {b.role} message ] ===============")
            b.print_debug()            
            print("\n")
    
    def print_list(self, metadata: bool = True, reverse: bool = False, delimiter: str = "\n\n"):
        # str_list = self.render_list(delimiter=delimiter)
        parts = []
        target = reversed(self.body) if reverse else self.body
        for b in target:
            part = ""
            if metadata:
                part += f"=============== [ {b.role} message ] | style: {b.style} | mutator: {b.mutator.__class__.__name__}\n"
            part += b.render()
            parts.append(part)        
        print(delimiter.join(parts))
    
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
        id: str | None = None,
    ):
        """
        Create a list schema.

        Args:
            item_name: Name/tag for individual items
            name: Name/tag for the list container (defaults to item_name + "_list")
            key: Optional key field for item lookup (for polymorphic lists)
            type: Optional type for validation
            style: Style for rendering
            tags: Additional tags
            role: Role identifier
            is_required: Whether this list is required
            attrs: Additional attributes
            id: Optional unique identifier (auto-generated if not provided)
        """
        # Use item_name + "_list" as container name if not specified
        list_name = name or f"{item_name}_list"

        # Initialize BlockSchema with list_name as content
        super().__init__(
            name=list_name,
            type=type,
            style=style,
            tags=[item_name] if name is None else [] + (tags or []),
            role=role,
            is_required=is_required,
            attrs=attrs,
            is_virtual=name is None,
            id=id,
        )

        # List-specific attributes
        self.item_name = item_name
        self.key = key
        self.list_schemas: list[BlockSchema] = []
        self.list_models: dict[str, Type[BaseModel]] = {}

    # -------------------------------------------------------------------------
    # Schema Registration
    # -------------------------------------------------------------------------

    def register(self, target: BlockSchema | Type[BaseModel]) -> BlockSchema:
        """
        Register an item schema or Pydantic model.

        For Pydantic models, converts to BlockSchema using pydantic_to_schema.
        The key field is used to differentiate between different model types.

        Args:
            target: BlockSchema or Pydantic BaseModel class

        Returns:
            The registered BlockSchema
        """
        from .object_helpers import pydantic_to_schema
        from ...utils.string_utils import camel_to_snake

        if isinstance(target, type) and issubclass(target, BaseModel):
            if self.key is None:
                raise ValueError("key field is required for Pydantic model registration")
            block = pydantic_to_schema(self.item_name, target, self.key)
            self.list_models[target.__name__] = target
        elif isinstance(target, BlockSchema):
            block = target
        else:
            raise ValueError(f"Invalid target type: {type(target)}")

        self.list_schemas.append(block)
        self.append_child(block)
        return block

    def get_item_schema(
        self,
        key_value: str | None = None,
        model_cls: Type[BaseModel] | None = None
    ) -> BlockSchema | None:
        """
        Get an item schema, optionally by key value.

        Args:
            key_value: Key value to look up
            model_cls: Pydantic model class to look up by name

        Returns:
            Matching BlockSchema or None
        """
        from ...utils.string_utils import camel_to_snake

        if model_cls is None and key_value is None:
            # Return first schema if no lookup specified
            return self.list_schemas[0] if self.list_schemas else None

        if model_cls is not None:
            key_value = camel_to_snake(model_cls.__name__)

        if key_value is not None and self.key is not None:
            for schema in self.list_schemas:
                if schema.attrs.get(self.key) == key_value:
                    return schema
        raise ValueError(f"Could not find item schema for model: {model_cls.__name__}")

    # -------------------------------------------------------------------------
    # Instantiation
    # -------------------------------------------------------------------------

    def instantiate(
        self,
        content: list | None = None,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> BlockList:
        """
        Create a BlockList instance from this schema.

        Args:
            content: Optional list of items to populate
            style: Override schema style
            role: Override schema role
            tags: Override schema tags
            attrs: Override schema attrs

        Returns:
            A BlockList with item_schema set for appending items
        """
        # Get the first item schema if registered
        item_schema = self.list_schemas[0] if self.list_schemas else None

        block_list = BlockList(
            item_schema=item_schema,
            role=role if role is not None else self._role,
            tags=tags if tags is not None else list(self.tags),
            style=style if style is not None else list(self.style),
            attrs=attrs if attrs is not None else dict(self.attrs),
        )

        # Populate with content if provided
        if content is not None:
            for item in content:
                block_list.append_item(item)

        return block_list

    def instantiate_item(
        self,
        content: ContentType | BaseModel | None = None,
        *,
        key_value: str | None = None,
        model_cls: Type[BaseModel] | None = None,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
    ) -> Block:
        """
        Instantiate a single item from the item schema.

        Args:
            content: Content for the item
            key_value: Optional key value for schema lookup
            model_cls: Optional model class for schema lookup
            style, role, tags: Override schema defaults
        """
        item_schema = self.get_item_schema(key_value=key_value, model_cls=model_cls)

        if item_schema is not None:
            return item_schema.instantiate(content, style=style, role=role, tags=tags)
        else:
            # No item schema - create a simple block with item_name as tag
            item_tags = [self.item_name] + (tags or [])
            return Block(
                content=content if not isinstance(content, BaseModel) else None,
                role=role if role is not None else self._role,
                tags=item_tags,
                style=style if style is not None else list(self.style),
            )

    def _inst_from_list(
        self,
        data: list,
        *,
        style: str | list[str] | None = None,
        role: str | None = None,
        tags: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
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
            attrs: Additional attributes

        Returns:
            A BlockList with instantiated items
        """
        from ...utils.string_utils import camel_to_snake

        # Create the list container
        block_list = self.instantiate(
            role=role,
            tags=tags,
            attrs=attrs,
        )

        for item in data:
            if isinstance(item, dict):
                # Dict - use first item schema or create simple block
                item_schema = self.get_item_schema()
                if item_schema is not None and item_schema.children:
                    # Dict with nested schema
                    item_block = item_schema._inst_from_dict(item, style=style)
                elif item_schema is not None:
                    # Dict but no nested schema - use instantiate
                    item_block = item_schema.instantiate(item, style=style)
                else:
                    # No item schema - create block with item_name tag
                    item_block = Block(tags=[self.item_name], style=style)
                    # Add dict values as children
                    for k, v in item.items():
                        child = Block(content=v, tags=[k])
                        item_block.append_child(child)

            elif isinstance(item, BaseModel):
                # Pydantic model - look up schema by model class
                item_schema = self.get_item_schema(model_cls=item.__class__)
                if item_schema is None:
                    raise ValueError(f"Could not find item schema for model: {item.__class__.__name__}")

                # Add key attribute if key field is set
                item_attrs = None
                if self.key is not None:
                    item_attrs = {self.key: camel_to_snake(item.__class__.__name__)}

                item_block = item_schema._inst_from_dict(
                    item.model_dump(),
                    style=style,
                    attrs=item_attrs,
                )

            else:
                # Scalar value
                item_schema = self.get_item_schema()
                if item_schema is not None:
                    item_block = item_schema.instantiate(item, style=style)
                else:
                    item_block = Block(
                        content=item,
                        tags=[self.item_name],
                        style=style,
                    )

            block_list.append_child(item_block)

        return block_list

    # -------------------------------------------------------------------------
    # Child Management
    # -------------------------------------------------------------------------

    def _raw_append_child(self, child: Block | None = None, content: str | None = None, to_body: bool = True) -> Block:
        """Override to maintain list_schemas when adding schema children."""
        result = super()._raw_append_child(child, content, to_body=to_body)
        if isinstance(result, BlockSchema):
            self.list_schemas.append(result)
        return result

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy(self, deep: bool = True, copy_id: bool = False) -> BlockListSchema:
        """Copy this list schema.

        Args:
            deep: If True, recursively copy children
            copy_id: If True, copy the id from this schema. If False, generate a new id.
        """
        new_schema = BlockListSchema(
            item_name=self.item_name,
            name=self.name,
            key=self.key,
            type=self._type,
            style=list(self.style),
            tags=list(self.tags),
            role=self._role,
            is_required=self.is_required,
            attrs=dict(self.attrs),
            id=self.id if copy_id else None,
        )

        if deep:
            # Deep copy children - _raw_append_child auto-populates list_schemas
            for child in self.children:
                child_copy = child.copy(deep=True, copy_id=copy_id)
                new_schema._raw_append_child(child_copy)

        new_schema.list_models = self.list_models.copy()

        return new_schema

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def model_dump(self) -> dict[str, Any]:
        """Serialize BlockListSchema to dict."""
        result = super().model_dump()
        result["item_name"] = self.item_name
        if self.key:
            result["key"] = self.key
        return result

    @classmethod
    def model_load(cls, data: dict[str, Any]) -> BlockListSchema:
        """Deserialize dict to BlockListSchema."""
        schema = cls(
            item_name=data.get("item_name", "item"),
            name=data.get("name"),
            key=data.get("key"),
            type=None,
            style=data.get("style", []),
            tags=data.get("tags", []),
            role=data.get("role"),
            is_required=data.get("is_required", True),
            attrs=data.get("attrs", {}),
        )
        
        schema = cls._load_mutator(schema, data)

        # Restore ID and text
        schema.id = data.get("id", schema.id)
        schema._text = data.get("text", "")

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

        # Recursively load children - Block.model_load handles class dispatch
        # _raw_append_child auto-populates list_schemas
        for child_data in data.get("children", []):
            child = Block.model_load(child_data)
            schema._raw_append_child(child)

        return schema

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"item_name={self.item_name!r}"]
        if self.name != f"{self.item_name}_list":
            parts.append(f"name={self.name!r}")
        if self.key:
            parts.append(f"key={self.key!r}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        if not self.is_required:
            parts.append("optional")
        return f"BlockListSchema({', '.join(parts)})"
