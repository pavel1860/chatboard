from __future__ import annotations
from ..prompt import Block
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field, PrivateAttr
from ..parsers import XmlOutputParser




class OutputModel(BaseModel):
    _tool_calls: list[BaseModel] = PrivateAttr(default=[])
    _tools: BaseModel
    
    @classmethod
    def render(cls) -> Block | None:
        return None
    
    @classmethod
    def to_block(cls) -> Block:
        with Block("Output format", tags=["output_format"]) as of:
            of += "you should use the following format for your output:"
            for field, field_info in cls.model_fields.items():
                with of(field, type=field_info.annotation.__name__, style=["xml"]):
                    if field_info.description:
                        of += field_info.description
            cls.render()
            # if render_output is not None:
                # output_format.append(render_output)
            return of
        
    @classmethod
    def parse(cls, text: str):
        xml_parser = XmlOutputParser()
        fmt_res, fmt_actions = xml_parser.parse(f"<root>{text}</root>", [], cls)
        cls._tool_calls = fmt_actions
        return fmt_res
