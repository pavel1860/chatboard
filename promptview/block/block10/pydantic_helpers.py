from ...utils.string_utils import camel_to_snake
from .block import Block, BlockBase, BlockSchema
from pydantic import BaseModel
from typing import Any, Generator, Literal, Type




DictTraverseAction = Literal["open", "close", "open-close"] 
DictFieldType = Literal["field", "dict", "list", "list-item", "model", "model-list-item"] 
    

def traverse_dict(target: dict, path: list[int] | None = None, label_path: list[str] | None = None) -> Generator[tuple[str, Any, list[int], list[str], DictTraverseAction, DictFieldType], None, None]:
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




def pydantic_to_block(name: str, cls: Type[BaseModel], key_field: str) -> BlockBase:
    if key_field is None:
        raise ValueError("key_field is required")
    tool_name = camel_to_snake(cls.__name__)
    with Block("Tool Name: " + tool_name) as tool:
        with tool.view(name, type=cls, tags=[tool_name, "model-schema"], style="xml") as b:
            # b.field(key_field, tool_name, type=key_type)
            if not cls.__doc__:
                raise ValueError(f"description is required for Tool {cls.__name__}")
            b(cls.__doc__, tags=["description"])
            
            with b("Parameters:") as params:
                for field_name, field_info in cls.model_fields.items():
                    if not field_info.description:
                        raise ValueError(f"description is required for field '{field_name}' in Tool {cls.__name__}")
                    with params.view(field_name, type=field_info.annotation, tags=[field_name, "field"]) as bf:
                        bf /= field_info.description
    return tool