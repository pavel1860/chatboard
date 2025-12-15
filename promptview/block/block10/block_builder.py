# """
# Block Builder - Streaming Block Instantiation from Schema

# This module provides utilities for incrementally constructing Block instances
# from BlockSchema definitions. It supports two primary use cases:

# 1. **Streaming XML Parsing** (via Parser):
#    - `StreamingBlockBuilder` maintains a stack of open blocks
#    - XML start/end events map to open/close operations
#    - Character data is appended to the current block

# 2. **Dict Instantiation** (via `instantiate_from_dict`):
#    - Recursively converts a dictionary to Block structure
#    - Uses schema to determine types and structure

# ## Architecture

# ```
# BlockSchema (template)
#     |
#     v
# StreamingBlockBuilder
#     ├── root_schema (BlockSchema tree - extracted schemas only)
#     ├── root (BlockBuilder for root block)
#     └── stack[] (BlockBuilder for nested blocks)
#            |
#            v instantiate()
#         Block (instances)
# ```

# ## Example Usage

# ### Streaming (for Parser):
# ```python
# builder = StreamingBlockBuilder(schema)
# builder.open_view("response", chunks)
# builder.append(chunk)
# builder.close_view(postfix_chunks)
# result = builder.result
# ```

# ### Dict instantiation:
# ```python
# block = instantiate_from_dict(schema, {"key": "value", "nested": {"foo": "bar"}})
# ```
# """

# from typing import Any, Type
# from pydantic import BaseModel

# from .block import Block, BlockSchema, BlockBase
# from .chunk import BlockChunk, BlockText


# def traverse_dict(target: dict, path: list[int] | None = None, label_path: list[str] | None = None):
#     """
#     Traverse a nested dictionary, yielding each key-value pair with path info.

#     For nested dicts, yields the dict key with value=None first, then recurses.
#     For lists, yields each item separately.
#     For leaf values, yields the key-value pair.

#     Args:
#         target: The dictionary to traverse
#         path: Current numeric path (indices)
#         label_path: Current string path (keys)

#     Yields:
#         Tuple of (key, value, path, label_path)
#         - For intermediate dict nodes: value is None
#         - For leaf nodes: value is the actual value
#     """
#     if path is None:
#         path = []
#     if label_path is None:
#         label_path = []

#     for i, (k, v) in enumerate(target.items()):
#         if type(v) is dict:
#             yield k, None, [*path, i], [*label_path, k]
#             yield from traverse_dict(v, [*path, i], [*label_path, k])
#         elif type(v) is list:
#             for item in v:
#                 yield k, item, [*path, i], [*label_path, k]
#         else:
#             yield k, v, [*path, i], [*label_path, k]


# def instantiate_from_dict(schema: BlockSchema, data: dict[str, Any]) -> Block:
#     """
#     Recursively instantiate a Block tree from a dictionary using a schema.

#     This is a non-streaming alternative to StreamingBlockBuilder for cases
#     where the full data is available upfront.

#     Args:
#         schema: The BlockSchema defining the structure
#         data: The dictionary data to instantiate

#     Returns:
#         A Block instance populated with the data

#     Example:
#         >>> schema = BlockSchema("response")
#         >>> schema.view("message", type=str)
#         >>> block = instantiate_from_dict(schema, {"message": "hello"})
#     """
#     # Create root block
#     root_block = schema.instantiate()

#     def _get_child_schema(parent_schema: BlockSchema, key: str) -> BlockSchema | None:
#         """Find a child schema by name."""
#         for child in parent_schema.children:
#             if isinstance(child, BlockSchema) and child.name == key:
#                 return child
#         return None

#     def _instantiate_value(parent_schema: BlockSchema, key: str, value: Any) -> Block:
#         """Instantiate a single value within a parent schema."""
#         child_schema = _get_child_schema(parent_schema, key)

#         if child_schema is None:
#             raise ValueError(f"Schema '{key}' not found in parent '{parent_schema.name}'")

#         if isinstance(value, dict):
#             # Recurse for nested dicts
#             return _instantiate_dict(child_schema, value)
#         elif isinstance(value, BaseModel):
#             # Handle Pydantic models
#             return _instantiate_pydantic(child_schema, value)
#         else:
#             # Leaf value
#             block = child_schema.instantiate()
#             if value is not None:
#                 block /= str(value)
#             return block

#     def _instantiate_dict(parent_schema: BlockSchema, data: dict) -> Block:
#         """Instantiate a dict as a Block with children."""
#         block = parent_schema.instantiate()
#         for key, value in data.items():
#             child = _instantiate_value(parent_schema, key, value)
#             block /= child
#         return block

#     def _instantiate_pydantic(schema: BlockSchema, model: BaseModel) -> Block:
#         """Instantiate a Pydantic model as a Block."""
#         block = schema.instantiate()
#         for field_name, field_value in model.model_dump().items():
#             child_schema = _get_child_schema(schema, field_name)
#             if child_schema:
#                 child = child_schema.instantiate()
#                 if field_value is not None:
#                     child /= str(field_value)
#                 block /= child
#         return block

#     # Process top-level data
#     for key, value in data.items():
#         child = _instantiate_value(schema, key, value)
#         root_block /= child

#     return root_block


# class BlockBuilder:
#     """
#     Builds a single Block from a BlockSchema.

#     This is a wrapper that tracks the lifecycle of a single block instantiation:
#     - Not started → Started (has block) → Finished

#     Used internally by StreamingBlockBuilder to manage each level of nesting.

#     Attributes:
#         schema: The BlockSchema being instantiated
#         block: The Block instance (None until started)
#         parent: Parent BlockBuilder (None for root)
#     """

#     def __init__(self, schema: BlockSchema, parent: "BlockBuilder | None" = None):
#         """
#         Initialize a BlockBuilder.

#         Args:
#             schema: The BlockSchema to instantiate
#             parent: Parent BlockBuilder (None for root)
#         """
#         self.schema = schema
#         self.parent = parent
#         self.block: Block | None = None
#         self._is_started = False
#         self._is_finished = False

#     @property
#     def is_started(self) -> bool:
#         """Check if the block has been instantiated."""
#         return self._is_started

#     @property
#     def is_finished(self) -> bool:
#         """Check if the block has been finalized."""
#         return self._is_finished

#     def start(
#         self,
#         content: str | list[BlockChunk] | None = None,
#         attrs: dict[str, str] | None = None,
#         ignore_style: bool = False,
#         ignore_tags: bool = False,
#     ) -> Block:
#         """
#         Initialize the block from schema.

#         Args:
#             content: Optional initial content
#             attrs: Optional attributes dict

#         Returns:
#             The created Block instance

#         Raises:
#             ValueError: If already started
#         """
#         if self._is_started:
#             raise ValueError("Block already started")

#         self.block = self.schema.instantiate(ignore_style=ignore_style, ignore_tags=ignore_tags)

#         # Add initial content if provided
#         if content is not None:
#             if isinstance(content, str):
#                 self.block /= content
#             else:
#                 for chunk in content:
#                     self.block /= chunk

#         self._is_started = True
#         return self.block

#     def append(self, value: Block | str | BlockChunk, copy: bool = True) -> Block:
#         """
#         Append a child block or content to the current block.

#         Args:
#             value: Block or content to append
#             copy: If True (default), copy blocks before appending.
#                   If False, move blocks directly (caller loses ownership).

#         Returns:
#             The current block

#         Raises:
#             ValueError: If not started
#         """
#         if not self._is_started:
#             raise ValueError("Block not started")
#         if self.block is None:
#             raise ValueError("Block is None")

#         if isinstance(value, Block):
#             self.block.append_child(value, copy=copy)
#         else:
#             self.block.append(value)
#         return self.block

#     def append_chunk(self, chunk: BlockChunk) -> Block:
#         """
#         Append a chunk to the current block's last child.

#         Used for streaming character data within a tag.

#         Args:
#             chunk: BlockChunk to append

#         Returns:
#             The current block

#         Raises:
#             ValueError: If not started
#         """
#         if not self._is_started:
#             raise ValueError("Block not started")
#         if self.block is None:
#             raise ValueError("Block is None")

#         # Append chunk to the block's content
#         if self.block.children:
#             # Append to last child
#             last_child = self.block.children[-1]
#             last_child.append(chunk)
#         else:
#             # Append directly to this block
#             self.block.append(chunk)

#         return self.block

#     def get_child_schema(self, view_name: str) -> BlockSchema:
#         """
#         Get a child schema by name.

#         Args:
#             view_name: Name of the child schema

#         Returns:
#             The child BlockSchema

#         Raises:
#             ValueError: If not found
#         """
#         for child in self.schema.children:
#             if isinstance(child, BlockSchema) and child.name == view_name:
#                 return child
#         raise ValueError(f"Schema '{view_name}' not found in '{self.schema.name}'")

#     def finish(self, postfix: str | list[BlockChunk] | None = None) -> Block:
#         """
#         Finalize the block with optional postfix.

#         Args:
#             postfix: Optional postfix content (e.g., closing tag chunks)

#         Returns:
#             The finalized Block

#         Raises:
#             ValueError: If not started
#         """
#         if not self._is_started:
#             raise ValueError("Block not started")
#         if self.block is None:
#             raise ValueError("Block is None")

#         if postfix is not None:
#             if isinstance(postfix, str):
#                 self.block.postfix_append(postfix)
#             else:
#                 for chunk in postfix:
#                     self.block.postfix_append(chunk)

#         self._is_finished = True
#         return self.block


# class StreamingBlockBuilder:
#     """
#     Incrementally builds nested Block structure from schema.

#     Maintains a stack of open BlockBuilders, allowing nested block construction
#     as XML-like events arrive (open tag, content, close tag).

#     This is the primary builder for streaming scenarios like Parser where
#     content arrives in chunks and structure is determined by events.

#     Attributes:
#         root_schema: The root BlockSchema (with only BlockSchema nodes)
#         root: The root BlockBuilder (set on first open)
#         stack: Stack of currently open BlockBuilders

#     Example:
#         >>> builder = StreamingBlockBuilder(response_schema)
#         >>> builder.open_view("message", chunks, attrs={"type": "text"})
#         >>> builder.append(content_chunk)
#         >>> builder.close_view(closing_chunks)
#         >>> result = builder.result
#     """

#     def __init__(self, schema: BlockSchema, role: str = "assistant", tags: list[str] | None = None):
#         """
#         Initialize StreamingBlockBuilder.

#         Args:
#             schema: The schema defining allowed structure
#             role: Role for the root block (default: "assistant")
#             tags: Optional tags for the root block
#         """
#         # Extract only BlockSchema nodes from the tree
#         self.root_schema: BlockSchema = schema.extract_schema()
#         self.root: BlockBuilder | None = None
#         self.stack: list[BlockBuilder] = []
#         self._role = role
#         self._tags = tags

#     @property
#     def result(self) -> Block:
#         """
#         Get the built Block tree.

#         Returns:
#             The root Block

#         Raises:
#             ValueError: If no root block or unclosed blocks remain
#         """
#         if self.root is None:
#             raise ValueError("No root block - nothing was built")
#         if self.root.block is None:
#             raise ValueError("Root block is None")
#         return self.root.block

#     @property
#     def current(self) -> BlockBuilder | None:
#         """Get the current (top of stack) BlockBuilder, or None if empty."""
#         if not self.stack:
#             return None
#         return self.stack[-1]

#     def current_path(self) -> list[str]:
#         """
#         Get the current tag path as a list of tag names.

#         Returns:
#             List of tag names from root to current
#         """
#         return [b.schema.name for b in self.stack if b.block]

#     def _get_child_schema(self, view_name: str) -> BlockSchema:
#         """
#         Get child schema from current context.

#         If stack is empty, checks if view_name matches root_schema name first,
#         then looks in root_schema children.
#         Otherwise, looks in current builder's schema.

#         Args:
#             view_name: Name of the child schema

#         Returns:
#             The child BlockSchema
#         """
#         if not self.stack:
#             # Check if opening the root schema itself
#             if self.root_schema.name == view_name:
#                 return self.root_schema
#             # Look in root's children
#             for child in self.root_schema.children:
#                 if isinstance(child, BlockSchema) and child.name == view_name:
#                     return child
#             raise ValueError(f"Schema '{view_name}' not found in root")
#         return self.stack[-1].get_child_schema(view_name)

#     def open_root(self, content: str | list[BlockChunk] | None = None) -> Block:
#         """
#         Explicitly open the root schema as the container block.

#         This should be called before opening any child views if you want
#         the root schema to be part of the output tree.

#         Args:
#             content: Optional initial content

#         Returns:
#             The root Block
#         """
#         if self.root is not None:
#             raise ValueError("Root already opened")

#         builder = BlockBuilder(self.root_schema)
#         block = builder.start(content=content)
#         self.root = builder
#         self.stack.append(builder)
#         return block

#     def _push(self, builder: BlockBuilder) -> BlockBuilder:
#         """
#         Push a builder onto the stack and link to parent.

#         Args:
#             builder: The BlockBuilder to push

#         Returns:
#             The pushed builder
#         """
#         if builder.block is None:
#             raise ValueError("Cannot push builder with no block")

#         if self.stack:
#             # Add as child of current top using copy=False
#             # The builder owns this block exclusively, no copy needed
#             self.stack[-1].append(builder.block, copy=False)
#         else:
#             # This is the root
#             self.root = builder

#         self.stack.append(builder)
#         return builder

#     def _pop(self) -> BlockBuilder:
#         """
#         Pop the current builder from the stack.

#         Returns:
#             The popped BlockBuilder
#         """
#         return self.stack.pop()

#     def open_view(
#         self,
#         view_name: str,
#         content: str | list[BlockChunk] | None = None,
#         attrs: dict[str, str] | None = None,
#         ignore_style: bool = False,
#         ignore_tags: bool = False,
#     ) -> Block:
#         """
#         Open a new nested view/block.

#         Args:
#             view_name: Name of the view to open
#             content: Initial content (typically chunks from tag)
#             attrs: Attributes from the tag

#         Returns:
#             The created Block
#         """
#         schema = self._get_child_schema(view_name)
#         parent = self.stack[-1] if self.stack else None
#         builder = BlockBuilder(schema, parent=parent)
#         block = builder.start(content=content, attrs=attrs, ignore_style=ignore_style, ignore_tags=ignore_tags)
#         self._push(builder)
#         return block

#     def append(self, value: str | BlockChunk) -> Block:
#         """
#         Append content to the current block.

#         Args:
#             value: Content to append

#         Returns:
#             The current block

#         Raises:
#             ValueError: If no block is open
#         """
#         if not self.stack:
#             raise ValueError("No open block to append to")
#         return self.stack[-1].append(value)

#     def append_chunk(self, chunk: BlockChunk) -> Block:
#         """
#         Append a chunk to the current block's content stream.

#         Args:
#             chunk: BlockChunk to append

#         Returns:
#             The current block

#         Raises:
#             ValueError: If no block is open
#         """
#         if not self.stack:
#             raise ValueError("No open block to append to")
#         return self.stack[-1].append_chunk(chunk)

#     def close_view(self, postfix: str | list[BlockChunk] | None = None) -> Block:
#         """
#         Close the current view/block.

#         Args:
#             postfix: Optional closing content (e.g., closing tag chunks)

#         Returns:
#             The closed Block
#         """
#         builder = self._pop()
#         return builder.finish(postfix)

#     def build_from_dict(self, payload: dict[str, Any]) -> Block:
#         """
#         Build block tree from a dictionary.

#         This uses the streaming interface internally but processes
#         the entire dict at once.

#         Args:
#             payload: Dictionary data to build from

#         Returns:
#             The root Block
#         """
#         # First, create the root block from root_schema
#         root_builder = BlockBuilder(self.root_schema)
#         root_builder.start(None)
#         self.root = root_builder
#         self.stack.append(root_builder)

#         for key, value, path, label_path in traverse_dict(payload):
#             self.open_view(key, content=None)
#             if value is not None:
#                 if isinstance(value, BaseModel):
#                     # For Pydantic models, add each field as content
#                     for field_name, field_value in value.model_dump().items():
#                         if field_value is not None:
#                             try:
#                                 self.open_view(field_name, content=None)
#                                 self.append(str(field_value))
#                                 self.close_view()
#                             except ValueError:
#                                 # No schema for this field, skip it
#                                 pass
#                 else:
#                     self.append(str(value))
#             self.close_view()

#         return self.result

from .block import BlockSchema, BlockListSchema, BlockList, BlockBase, ContentType, BlockChunk






class BlockBuilderContext:
    
    def __init__(self, schema: "BlockSchema | None"):
        self.schema = schema.extract_schema() if schema is not None else None
        self._stack = []
        self._root = None
        self._block_text = None

    
    def _push(self, block: "BlockBase"):
        if self._root is None:
            self._root = block
        else:
            block = self._top().append_child(block)
        self._stack.append(block)
        return block
        
    def _pop(self):
        if len(self._stack) == 0:
            raise RuntimeError("No block to pop")
        return self._stack.pop()
    
    def _top(self):
        if len(self._stack) == 0:
            raise RuntimeError("No block on top")
        return self._stack[-1]
    
    def _top_or_none(self):
        if len(self._stack) == 0:
            return None
        return self._stack[-1]
    
    
    def instantiate(
        self, 
        name: str, 
        content: ContentType | None = None, 
        attrs: dict | None = None, 
        style: str | None = None, 
        role: str | None = None, 
        tags: list[str] | None = None,
        ignore_style: bool = False,
        ignore_tags: bool = False,
        ignore_name: bool = False,
        
    ):
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        block_schema = self.schema.get_one(name)
        if isinstance(block_schema.parent, BlockListSchema):
            if not isinstance(self._top_or_none(), BlockList):
                list_schema = block_schema.parent.instantiate()
                self._push(list_schema)
        else:
            if isinstance(self._top_or_none(), BlockList):
                self._pop()
        block = block_schema.instantiate(
            style=style, 
            role=role, 
            tags=tags, 
            ignore_style=ignore_style, 
            ignore_tags=ignore_tags,
            ignore_name=ignore_name
        )
        if content is not None:
            block.append(content, sep="")
        self._push(block)
        return block
        
    
    def commit(self, content: ContentType | None = None):
        if len(self._stack) == 0:
            raise RuntimeError("No block to commit")
        self._stack.pop()
        
    
    def append(self, chunk: "BlockChunk"):
        if len(self._stack) == 0:
            raise RuntimeError("No block to append to")
        self._top().append(chunk, sep="")
