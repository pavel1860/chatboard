
from typing import TYPE_CHECKING, Generic, List, Type, TypeVar
from uuid import uuid4
from pydantic import BaseModel, Field

import xml.etree.ElementTree as ET
# if TYPE_CHECKING:
    # from .prompt.prompt import ToolCall




OUTPUT_MODEL = TypeVar("OUTPUT_MODEL", bound=BaseModel)

class XmlOutputParser(Generic[OUTPUT_MODEL]):
    
    # def __init__(self, xml_string, actions):
    #     self.root = ET.fromstring(xml_string)
    #     self.actions = actions
        
        
    def find_tools(self, tools, root, tool_tag="tool", param_tag="param") -> "List[ToolCall]":
        from ..prompt import ToolCall
        tool_calls = []
        tool_lookup = {tool.__name__: tool for tool in tools}        
        for tool in root.findall(tool_tag):
            tool_cls = tool_lookup.get(tool.attrib["name"], None)
            if not tool_cls:
                from ..llms import ErrorMessage
                raise ErrorMessage(tool.attrib["name"])
            # params = {param.attrib["name"]: param.text for param in tool.findall(param_tag)}
            tool_inst = tool_cls.model_validate_json(tool.text)
            tool_calls.append(
                ToolCall(
                    id=f"tool_call_{uuid4()}"[:40], 
                    name=tool.attrib["name"], 
                    tool=tool_inst
                )
            )
        return tool_calls
        
        
    def find(self, root, tag):
        res = root.find(tag)
        if res is None:
            return None
        return res.text.strip()

    def get_model_fields(self, model_cls):
        if not issubclass(model_cls, BaseModel):
            raise ValueError("model_cls must be a subclass of pydantic.BaseModel")
        parent_cls = model_cls.__bases__[0]
        if not issubclass(parent_cls, BaseModel):
            raise ValueError("model_cls Parent must be a subclass of pydantic.BaseModel")
        parent_fields = parent_cls.model_fields
        child_fields = model_cls.model_fields
        child_only_fields = {k: v for k, v in child_fields.items() if k not in parent_fields}
        return child_only_fields
    
    def parse(self, content: str, tools: List[Type[BaseModel]], ai_model_cls: Type[OUTPUT_MODEL]) -> "tuple[OUTPUT_MODEL, List[ToolCall]]":
        # try:
        root = ET.fromstring(content)
        # except:
        #     raise
        fields = self.get_model_fields(ai_model_cls)
        params = {k: self.find(root, k) for k in fields}
        tool_calls = self.find_tools(tools, root)
        try:
            # output = ai_model_cls(
            #     # id=response.id,
            #     content=response.content,
            #     model=response.model,
            #     raw=response.raw,
            #     usage=response.usage,
            #     **(params | action_calls)
            # )
            # output = ai_model_cls(**(response.model_dump() | params | action_calls))
            return ai_model_cls(**params), tool_calls
            
        except TypeError as e:
            print(params)
            print(tool_calls)
            raise ValueError(f"Error parsing response: {e}")
        