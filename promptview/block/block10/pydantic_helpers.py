from ...utils.string_utils import camel_to_snake
from .block import Block, BlockBase, BlockSchema
from pydantic import BaseModel
from typing import Type








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