"""
Object helpers for converting Pydantic models to Block schemas.

This module provides utilities for:
- Converting Pydantic BaseModel classes to BlockSchema
- Traversing nested dictionaries
- Converting blocks back to Pydantic objects
"""

from __future__ import annotations
from typing import Any, Generator, Literal, Type
from types import UnionType, NoneType
from typing import get_origin, get_args, Union
import json

from pydantic import BaseModel

from .block import Block
from .schema import BlockSchema
from ..utils.string_utils import camel_to_snake


DictTraverseAction = Literal["open", "close", "open-close"]
DictFieldType = Literal["field", "dict", "list", "list-item", "model", "model-list-item"]


def traverse_dict(
    target: dict,
    path: list[int] | None = None,
    label_path: list[str] | None = None
) -> Generator[tuple[str, Any, list[int], list[str], DictTraverseAction, DictFieldType], None, None]:
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
        Tuple of (key, value, path, label_path, action, field_type)
        - For intermediate dict nodes: value is None
        - For leaf nodes: value is the actual value
    """
    if path is None:
        path = []
    if label_path is None:
        label_path = []

    for i, (k, v) in enumerate(target.items()):
        if type(v) is dict:
            yield k, None, [*path, i], [*label_path, k], "open", "dict"
            yield from traverse_dict(v, [*path, i], [*label_path, k])
            yield k, None, [*path, i], [*label_path, k], "close", "dict"
        elif type(v) is list:
            for item in v:
                if isinstance(item, BaseModel):
                    dump = item.model_dump()
                    model_name = camel_to_snake(item.__class__.__name__)
                    yield model_name, None, [*path, i], [*label_path, k], "open", "model-list-item"
                    yield from traverse_dict(dump, [*path, i, 0], [*label_path, k, model_name])
                    yield model_name, None, [*path, i], [*label_path, k], "close", "model-list-item"
                else:
                    yield k, item, [*path, i], [*label_path, k], "open-close", "list-item"
        else:
            yield k, v, [*path, i], [*label_path, k], "open-close", "field"


def pydantic_to_schema(
    name: str,
    cls: Type[BaseModel],
    key_field: str,
    style: str | list[str] | None = None,
) -> BlockSchema:
    """
    Convert a Pydantic model class to a BlockSchema.

    Creates a schema structure that includes:
    - The model's docstring as a description
    - All model fields with their descriptions
    - Field type annotations

    Args:
        name: Name for the schema (used as XML tag)
        cls: Pydantic BaseModel class
        key_field: Field name used to identify this model type
        style: Optional style for the schema

    Returns:
        BlockSchema representing the Pydantic model

    Example:
        class SearchTool(BaseModel):
            '''Search the web for information.'''
            query: str = Field(description="The search query")

        schema = pydantic_to_schema("tool", SearchTool, "type")
        # Creates schema like:
        # <tool type="search_tool">
        #   <Description>Search the web for information.</Description>
        #   <Parameters>
        #     <query>The search query</query>
        #   </Parameters>
        # </tool>
    """
    if key_field is None:
        raise ValueError("key_field is required")

    tool_name = camel_to_snake(cls.__name__)

    # Create main schema with key attribute
    with BlockSchema(
        name,
        type=cls,
        tags=[tool_name, "model-schema"],
        attrs={key_field: tool_name},
        style=style,
    ) as schema:
        # Add description from docstring
        if not cls.__doc__:
            raise ValueError(f"description is required for model {cls.__name__}")

        with schema("Description", tags=["description"]) as desc:
            desc /= cls.__doc__

        # Add parameters section
        with schema("Parameters", tags=["parameters"]) as params:
            for field_name, field_info in cls.model_fields.items():
                is_required = field_info.is_required()

                if not field_info.description:
                    raise ValueError(
                        f"description is required for field '{field_name}' in model {cls.__name__}"
                    )

                with params.view(
                    field_name,
                    type=field_info.annotation,
                    tags=[field_name, "field"],
                    is_required=is_required,
                ) as field_schema:
                    field_schema /= field_info.description

    return schema


def parse_content(content: str, type: Type) -> Any:
    """
    Parse string content to a specific type.

    Args:
        content: String content to parse
        type: Target type

    Returns:
        Parsed value
    """
    if type == int:
        return int(content)
    elif type == float:
        return float(content)
    elif type == bool:
        return content.lower() in ("true", "1", "yes")
    elif type == str:
        return content
    elif type == list:
        return content.split(",")
    elif type == dict:
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported type: {type}")
    
    
def parse_union_content(content, content_type):
    is_optional=False
    non_none_args = [content_type]
    origin = get_origin(content_type)
    if origin is Union or isinstance(content_type, UnionType):
        args = get_args(content_type)
        non_none_args = [arg for arg in args if arg is not NoneType]
        is_optional = len(non_none_args) < len(args) 
    if is_optional:
        if not content:
            return None
        elif content.lower().strip() == "none":
            return None    
    for args in non_none_args:
        try:
            return parse_content(content, args)
        except ValueError:
            continue
    raise ValueError(f"Failed to parse content: {content}")



def block_to_object(block: Block, target_type: Type) -> Any:
    """
    Convert a Block back to a Python object.

    For Pydantic models, extracts field values from block children
    and constructs the model instance.

    For simple types, parses the block content.

    Args:
        block: Block to convert
        target_type: Target type (Pydantic model or simple type)

    Returns:
        Converted object
    """
    if target_type is None:
        return block.render()

    if issubclass(target_type, BaseModel):
        # Build dict from block structure
        def extract_values(blk: Block, result: dict) -> None:
            for child in blk.body:
                if child.tags:
                    field_name = child.tags[0]
                    # Check if it's a container or leaf
                    if child.body:
                        # Nested structure
                        nested = {}
                        extract_values(child, nested)
                        if nested:
                            result[field_name] = nested
                    else:
                        # Leaf value - get rendered content
                        result[field_name] = child.content.strip()

        target = {}
        extract_values(block, target)
        return target_type(**target)
    else:
        # Simple type - parse content
        content = block.render().strip()
        return parse_content(content, target_type)


def block_to_dict(block: Block) -> dict[str, Any]:
    """
    Convert a Block tree to a dictionary.

    Uses block tags as keys and content as values.

    Args:
        block: Block to convert

    Returns:
        Dictionary representation
    """
    result = {}

    for child in block.body:
        if not child.tags:
            continue

        field_name = child.tags[0]

        if child.body:
            # Nested structure
            result[field_name] = block_to_dict(child)
        else:
            # Leaf value
            result[field_name] = child.content.strip()

    return result
