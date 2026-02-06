"""
Block Builder - Streaming Block Instantiation from Schema

This module provides utilities for incrementally constructing Block instances
from BlockSchema definitions. It supports two primary use cases:

1. **Streaming XML Parsing** (via Parser in fbp_process.py):
   - `StreamingBlockBuilder` maintains a stack of open blocks
   - XML start/end events map to open/close operations
   - Character data is appended to the current block

2. **Dict Instantiation** (via `instantiate_from_dict`):
   - Recursively converts a dictionary to Block structure
   - Uses schema to determine types and structure

## Architecture

```
BlockSchema (template)
    |
    v
StreamingBlockBuilder
    ├── root_schema (BlockSchema tree)
    ├── root (BlockBuilder for root block)
    └── stack[] (BlockBuilder for nested blocks)
           |
           v instantiate()
        Block (instances)
```

## Example Usage

### Streaming (for Parser):
```python
builder = StreamingBlockBuilder(schema)
builder.open_view("response", chunks)
builder.append(chunk)
builder.close_view(postfix_chunks)
result = builder.result
```

### Dict instantiation:
```python
block = instantiate_from_dict(schema, {"key": "value", "nested": {"foo": "bar"}})
```
"""

from .block import Block, BlockSchema, BlockSent, BlockListSchema, BlockChunk
from .base_blocks import BaseContent
from typing import Any
from pydantic import BaseModel


def traverse_dict(target: dict, path: list[int] | None = None, label_path: list[str] | None = None):
    """
    Traverse a nested dictionary, yielding each key-value pair with path info.

    For nested dicts, yields the dict key with value=None first, then recurses.
    For lists, yields each item separately.
    For leaf values, yields the key-value pair.

    Args:
        target: The dictionary to traverse
        path: Current numeric path (indices)
        label_path: Current string path (keys)

    Yields:
        Tuple of (key, value, path, label_path)
        - For intermediate dict nodes: value is None
        - For leaf nodes: value is the actual value
    """
    if path is None:
        path = []
    if label_path is None:
        label_path = []

    for i, (k, v) in enumerate(target.items()):
        if type(v) is dict:
            yield k, None, [*path, i], [*label_path, k]
            yield from traverse_dict(v, [*path, i], [*label_path, k])
        elif type(v) is list:
            for item in v:
                yield k, item, [*path, i], [*label_path, k]
        else:
            yield k, v, [*path, i], [*label_path, k]


def instantiate_from_dict(schema: BlockSchema, data: dict[str, Any]) -> Block:
    """
    Recursively instantiate a Block tree from a dictionary using a schema.

    This is a non-streaming alternative to StreamingBlockBuilder for cases
    where the full data is available upfront.

    Args:
        schema: The BlockSchema defining the structure
        data: The dictionary data to instantiate

    Returns:
        A Block instance populated with the data

    Example:
        >>> schema = BlockSchema("response")
        >>> schema.view("message", type=str)
        >>> block = instantiate_from_dict(schema, {"message": "hello"})
    """
    from ...utils.string_utils import camel_to_snake

    # Create root block
    root_block = schema.instantiate()

    def _instantiate_value(parent_schema: BlockSchema, key: str, value: Any) -> Block:
        """Instantiate a single value within a parent schema."""
        child_schema = parent_schema.get_one(key)

        if child_schema is None:
            raise ValueError(f"Schema '{key}' not found in parent")
        if not isinstance(child_schema, BlockSchema):
            raise ValueError(f"Schema '{key}' is not a BlockSchema")

        if isinstance(child_schema, BlockListSchema):
            # Handle list schema
            return _instantiate_list_item(child_schema, value)
        elif isinstance(value, dict):
            # Recurse for nested dicts
            return _instantiate_dict(child_schema, value)
        elif isinstance(value, BaseModel):
            # Handle Pydantic models
            return _instantiate_pydantic(child_schema, value)
        else:
            # Leaf value
            block = child_schema.instantiate()
            if value is not None:
                block.append(value)
            return block

    def _instantiate_dict(parent_schema: BlockSchema, data: dict) -> Block:
        """Instantiate a dict as a Block with children."""
        block = parent_schema.instantiate()
        for key, value in data.items():
            child = _instantiate_value(parent_schema, key, value)
            block.append(child)
        return block

    def _instantiate_list_item(list_schema: BlockListSchema, value: Any) -> Block:
        """Instantiate an item within a BlockListSchema."""
        if isinstance(value, BaseModel):
            model_name = camel_to_snake(value.__class__.__name__)
            if list_schema.key is None:
                raise ValueError("BlockListSchema requires a key field")
            attrs = {list_schema.key: model_name}
            item_schema = list_schema.get_one(model_name)
            if item_schema is None:
                raise ValueError(f"Schema for '{model_name}' not found in list")
            block = item_schema.instantiate(attrs=attrs)
            # Add pydantic fields
            for field_name, field_value in value.model_dump().items():
                field_schema = item_schema.get_one_or_none(field_name)
                if field_schema and isinstance(field_schema, BlockSchema):
                    child = field_schema.instantiate()
                    if field_value is not None:
                        child.append(field_value)
                    block.append(child)
            return block
        else:
            return list_schema.instantiate_item(value=value)

    def _instantiate_pydantic(schema: BlockSchema, model: BaseModel) -> Block:
        """Instantiate a Pydantic model as a Block."""
        block = schema.instantiate()
        for field_name, field_value in model.model_dump().items():
            field_schema = schema.get_one_or_none(field_name)
            if field_schema and isinstance(field_schema, BlockSchema):
                child = field_schema.instantiate()
                if field_value is not None:
                    child.append(field_value)
                block.append(child)
        return block

    # Process top-level data
    for key, value in data.items():
        child = _instantiate_value(schema, key, value)
        root_block.append(child)

    return root_block


class BlockBuilder:
    """
    Builds a single Block from a BlockSchema.

    This is a wrapper that tracks the lifecycle of a single block instantiation:
    - Not started → Started (has block) → Finished

    Used internally by StreamingBlockBuilder to manage each level of nesting.

    Attributes:
        schema: The BlockSchema being instantiated
        block: The Block instance (None until started)
        is_list: Whether this is a BlockListSchema
    """

    def __init__(self, schema: BlockSchema):
        """
        Initialize a BlockBuilder.

        Args:
            schema: The BlockSchema to instantiate
        """
        self.schema = schema
        self.block: Block | None = None
        self._is_started = False
        self._is_finished = False

    @property
    def is_list(self) -> bool:
        """Check if this builder wraps a BlockListSchema."""
        return isinstance(self.schema, BlockListSchema)

    @property
    def is_started(self) -> bool:
        """Check if the block has been instantiated."""
        return self._is_started

    @property
    def is_finished(self) -> bool:
        """Check if the block has been finalized."""
        return self._is_finished

    def start(
        self,
        value: str | list[BlockChunk] | None = None,
        content: str | None = None,
        attrs: dict[str, str] | None = None,
        ignore_style: bool = False,
        ignore_tags: bool = False,
    ) -> Block:
        """
        Initialize the block from schema.

        Args:
            value: Optional initial value/content
            content: Optional content string
            attrs: Optional attributes dict
            ignore_style: Skip style from schema
            ignore_tags: Skip tags from schema

        Returns:
            The created Block instance

        Raises:
            ValueError: If already started
        """
        if self._is_started:
            raise ValueError("Block already started")
        self.block = self.schema.instantiate(
            value=value,
            content=content,
            attrs=attrs,
            ignore_style=ignore_style,
            ignore_tags=ignore_tags
        )
        self._is_started = True
        return self.block

    def start_list_item(
        self,
        value: str | list[BlockChunk] | None = None,
        content: str | None = None,
        attrs: dict[str, str] | None = None,
        ignore_style: bool = False,
        ignore_tags: bool = False,
    ) -> Block:
        """
        Instantiate a list item (for BlockListSchema).

        The list wrapper must already be started.

        Args:
            value: Optional initial value
            content: Optional content string
            attrs: Attributes including the key field
            ignore_style: Skip style from schema
            ignore_tags: Skip tags from schema

        Returns:
            The created list item Block

        Raises:
            ValueError: If not started or not a list schema
        """
        if not self._is_started:
            raise ValueError("Block not started - call start() first")
        if self.block is None:
            raise ValueError("Block is None")
        if not isinstance(self.schema, BlockListSchema):
            raise ValueError("Schema is not a BlockListSchema")
        self.block = self.schema.instantiate_item(
            value=value,
            content=content,
            attrs=attrs,
            ignore_style=ignore_style,
            ignore_tags=ignore_tags
        )
        return self.block

    def append(self, value: Block | BaseContent) -> Block:
        """
        Append a child block or content to the current block.

        Args:
            value: Block or content to append

        Returns:
            The current block

        Raises:
            ValueError: If not started
        """
        if not self._is_started:
            raise ValueError("Block not started")
        if self.block is None:
            raise ValueError("Block is None")
        self.block.append(value)
        return self.block

    def inline_append(self, value: BlockChunk | BaseContent) -> Block:
        """
        Append inline content to the last child's content.

        Used for streaming character data within a tag.

        Args:
            value: Content to append inline

        Returns:
            The current block

        Raises:
            ValueError: If not started or no children
        """
        if not self._is_started:
            raise ValueError("Block not started")
        if self.block is None:
            raise ValueError("Block is None")
        if len(self.block) == 0:
            self.block.append(Block())
        self.block.inline_append(value)
        return self.block

    def get_child_schema(self, view_name: str) -> BlockSchema:
        """
        Get a child schema by name.

        Args:
            view_name: Name of the child schema

        Returns:
            The child BlockSchema

        Raises:
            ValueError: If not found or not a BlockSchema
        """
        schema = self.schema.get_one(view_name)
        if schema is None:
            raise ValueError(f"Schema '{view_name}' not found")
        if not isinstance(schema, BlockSchema):
            raise ValueError(f"Schema '{view_name}' is not a BlockSchema")
        return schema

    def finish(self, postfix: str | list[BlockChunk] | None = None) -> Block:
        """
        Finalize the block with optional postfix.

        Args:
            postfix: Optional postfix content (e.g., closing tag chunks)

        Returns:
            The finalized Block

        Raises:
            ValueError: If not started
        """
        if not self._is_started:
            raise ValueError("Block not started")
        if self.block is None:
            raise ValueError("Block is None")
        if postfix is not None:
            self.block.postfix = BlockSent(postfix)
        self._is_finished = True
        return self.block


class StreamingBlockBuilder:
    """
    Incrementally builds nested Block structure from schema.

    Maintains a stack of open BlockBuilders, allowing nested block construction
    as XML-like events arrive (open tag, content, close tag).

    This is the primary builder for streaming scenarios like Parser where
    content arrives in chunks and structure is determined by events.

    Attributes:
        root_schema: The root BlockSchema (with only BlockSchema nodes)
        root: The root BlockBuilder (set on first open)
        stack: Stack of currently open BlockBuilders

    Example:
        >>> builder = StreamingBlockBuilder(response_schema)
        >>> builder.open_view("message", chunks, attrs={"type": "text"})
        >>> builder.append(content_chunk)
        >>> builder.close_view(closing_chunks)
        >>> result = builder.result
    """

    def __init__(self, schema: BlockSchema | Block, role: str = "assistant", tags: list[str] | None = None):
        """
        Initialize StreamingBlockBuilder.

        Args:
            schema: The schema defining allowed structure
            role: Role for the root block (default: "assistant")
            tags: Optional tags for the root block
        """
        self.root_schema: BlockSchema = schema.copy_kind(BlockSchema)
        self.root: BlockBuilder | None = None
        self.stack: list[BlockBuilder] = []
        self._role = role
        self._tags = tags

    @property
    def result(self) -> Block:
        """
        Get the built Block tree.

        Returns:
            The root Block

        Raises:
            ValueError: If no root block or unclosed blocks remain
        """
        if self.root is None:
            raise ValueError("No root block - nothing was built")
        return self.root.block

    @property
    def current(self) -> BlockBuilder | None:
        """Get the current (top of stack) BlockBuilder, or None if empty."""
        if not self.stack:
            return None
        return self.stack[-1]

    def current_path(self) -> list[str]:
        """
        Get the current tag path as a list of tag names.

        Returns:
            List of tag names from root to current
        """
        return [b.block.tags[0] for b in self.stack if b.block and b.block.tags]

    def _get_child_schema(self, view_name: str) -> BlockSchema:
        """
        Get child schema from current context.

        If stack is empty, looks in root_schema.
        Otherwise, looks in current builder's schema.

        Args:
            view_name: Name of the child schema

        Returns:
            The child BlockSchema
        """
        if not self.stack:
            schema = self.root_schema.get_one(view_name)
            if schema is None:
                raise ValueError(f"Schema '{view_name}' not found in root")
            if not isinstance(schema, BlockSchema):
                raise ValueError(f"Schema '{view_name}' is not a BlockSchema")
            return schema
        return self.stack[-1].get_child_schema(view_name)

    def _get_list_item_schema(self, attrs: dict[str, str]) -> BlockSchema:
        """
        Get the schema for a list item based on key attribute.

        Args:
            attrs: Attributes dict containing the key field

        Returns:
            The item's BlockSchema

        Raises:
            ValueError: If stack is empty or current is not a list
        """
        if not self.stack:
            raise ValueError("Stack is empty")
        builder = self.stack[-1]
        if not isinstance(builder.schema, BlockListSchema) or builder.schema.key is None:
            raise ValueError("Current schema is not a BlockListSchema or key is not set")
        key_value = attrs.get(builder.schema.key)
        if key_value is None:
            raise ValueError(f"Key field '{builder.schema.key}' not in attrs")
        return builder.get_child_schema(key_value)

    def _push(self, builder: BlockBuilder) -> BlockBuilder:
        """
        Push a builder onto the stack and link to parent.

        Args:
            builder: The BlockBuilder to push

        Returns:
            The pushed builder
        """
        if self.stack:
            # Add as child of current top
            self.stack[-1].append(builder.block)
        else:
            # This is the root
            self.root = builder
        self.stack.append(builder)
        return builder

    def _pop(self) -> BlockBuilder:
        """
        Pop the current builder from the stack.

        Returns:
            The popped BlockBuilder
        """
        return self.stack.pop()

    def _is_list_already_open(self, view_name: str) -> bool:
        """
        Check if a list with the given view_name is already open on the stack.

        This is used to determine whether to create a new list wrapper or
        add another item to an existing list.
        """
        if not self.stack:
            return False
        current = self.stack[-1]
        return (
            current.is_list and
            current.is_started and
            view_name in current.schema.tags
        )

    def open_view(
        self,
        view_name: str,
        value: str | list[BlockChunk] | None = None,
        attrs: dict[str, str] | None = None,
        ignore_style: bool = False,
        ignore_tags: bool = False,
    ) -> list[Block]:
        """
        Open a new nested view/block.

        For BlockListSchema, this handles both the list wrapper and item creation.

        Args:
            view_name: Name of the view to open
            value: Initial content (typically chunks from tag)
            attrs: Attributes from the tag

        Returns:
            List of created blocks (usually 1, but 2 for first list item)
        """
        # Check if we're adding another item to an already-open list
        if self._is_list_already_open(view_name):
            if not attrs:
                raise ValueError("Attributes required for list item (must include key field)")

            # Get item schema and create item builder
            item_schema = self._get_list_item_schema(attrs)
            item_builder = BlockBuilder(item_schema)
            block = item_builder.start(content=value, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
            self._push(item_builder)
            return [block]

        schema = self._get_child_schema(view_name)
        builder = BlockBuilder(schema)

        if builder.is_list:
            # BlockListSchema: first create wrapper, then item
            builder.start(None)  # Empty wrapper
            self._push(builder)

            if not attrs:
                raise ValueError("Attributes required for list item (must include key field)")

            # Get item schema and create item builder
            item_schema = self._get_list_item_schema(attrs)
            item_builder = BlockBuilder(item_schema)
            block = item_builder.start(content=value, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
            self._push(item_builder)
        else:
            block = builder.start(content=value, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
            self._push(builder)

        if block is None:
            raise ValueError("Block was not created")
        return [block]

    def append(self, value: BlockChunk | BaseContent) -> Block:
        """
        Append inline content to the current block.

        Args:
            value: Content to append

        Returns:
            The current block

        Raises:
            ValueError: If no block is open
        """
        if not self.stack:
            raise ValueError("No open block to append to")
        return self.stack[-1].inline_append(value)

    def close_view(self, postfix: str | list[BlockChunk] | None = None) -> Block:
        """
        Close the current view/block.

        Args:
            postfix: Optional closing content (e.g., closing tag chunks)

        Returns:
            The closed Block
        """
        builder = self._pop()
        return builder.finish(postfix)

    def build_from_dict(self, payload: dict[str, Any]) -> Block:
        """
        Build block tree from a dictionary.

        This uses the streaming interface internally but processes
        the entire dict at once. For non-streaming use cases, prefer
        `instantiate_from_dict()` instead.

        Args:
            payload: Dictionary data to build from

        Returns:
            The root Block
        """
        from ...utils.string_utils import camel_to_snake

        # First, create the root block from root_schema
        root_builder = BlockBuilder(self.root_schema)
        root_builder.start(None)  # Empty root block
        self.root = root_builder
        self.stack.append(root_builder)

        # Track which lists are open so we know when to close them
        open_lists: dict[str, bool] = {}

        items = list(traverse_dict(payload))
        for idx, (key, value, path, label_path) in enumerate(items):
            # Check if this key corresponds to a list schema
            current = self.current
            is_list_item = False
            list_key = None

            # Determine if we're dealing with a list
            if current is not None and self._is_list_already_open(key):
                # We're adding another item to an already-open list
                is_list_item = True
                list_key = key
            else:
                # Check if the schema for this key is a list
                try:
                    schema = self._get_child_schema(key)
                    if isinstance(schema, BlockListSchema):
                        is_list_item = True
                        list_key = key
                except ValueError:
                    pass

            # Determine attrs for list items with Pydantic models
            attrs = None
            if is_list_item and isinstance(value, BaseModel):
                # Get the list schema to find the key field
                if self._is_list_already_open(key):
                    list_schema = self.stack[-1].schema
                else:
                    list_schema = self._get_child_schema(key)
                if isinstance(list_schema, BlockListSchema) and list_schema.key is not None:
                    model_name = camel_to_snake(value.__class__.__name__)
                    attrs = {list_schema.key: model_name}
                    open_lists[key] = True

            self.open_view(key, key, attrs=attrs)
            if value is not None:
                if isinstance(value, BaseModel):
                    # For Pydantic models, add each field as content
                    for field_name, field_value in value.model_dump().items():
                        if field_value is not None:
                            # Try to find a child schema for this field
                            try:
                                self.open_view(field_name, field_name)
                                self.stack[-1].append(str(field_value))
                                self.close_view()
                            except ValueError:
                                # No schema for this field, skip it
                                pass
                else:
                    self.stack[-1].append(value)
            self.close_view()

            # Check if we need to close the list
            # A list should be closed when the next item is not for the same list key
            if list_key and list_key in open_lists:
                # Look ahead to see if next item is for the same list
                next_is_same_list = False
                if idx + 1 < len(items):
                    next_key = items[idx + 1][0]
                    next_is_same_list = (next_key == list_key)

                if not next_is_same_list:
                    # Close the list
                    self.close_view()
                    del open_lists[list_key]

        return self.result


# Backward compatibility aliases
BlockBuildContext = BlockBuilder
SchemaBuildContext = StreamingBlockBuilder
