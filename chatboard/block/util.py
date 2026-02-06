import json
import textwrap
from typing import Literal
from pydantic import BaseModel


BlockRole = Literal["assistant", "user", "system", "tool"]

class LlmUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCall(BaseModel):
    id: str
    name: str
    tool: dict | BaseModel
    extra: dict
    
    def __init__(self, id: str, name: str, tool: dict | BaseModel, extra: dict | None = None):
        super().__init__(id=id, name=name, tool=tool, extra=extra or {})
        self.tool = tool
    
    @property
    def type(self):
        return type(self.tool)
    
    def __repr__(self) -> str:
        return f"ToolCall(id='{self.id}', name='{self.name}', tool={self.tool.__class__.__name__})"
    
    def to_json(self):
        return self.tool.model_dump_json() if isinstance(self.tool, BaseModel) else json.dumps(self.tool)

    def model_dump(self, *args, **kwargs):
        dump = super().model_dump(*args, **kwargs)
        if isinstance(self.tool, BaseModel):
            dump["tool"] = self.tool.model_dump(*args, **kwargs)
        elif isinstance(self.tool, dict):
            dump["tool"] = self.tool
        else:
            raise ValueError(f"Tool is not a BaseModel or dict: {type(self.tool)}")
        return dump




StreamStatus = Literal["stream_start", "stream_success", "stream_error"]



class StreamEvent:
    event: StreamStatus | None = None
    metadata: dict | None = None





class LLMEvent(StreamEvent):
    type: Literal["stream_start", "stream_success", "stream_error"]
    data: dict | None = None
    
    def __init__(self, type: Literal["stream_start", "stream_success", "stream_error"], data: dict | None = None):
        self.type = type
        self.data = data
        
    def __repr__(self) -> str:
        return f"LLMEvent(type='{self.type}', data={self.data})"
        
        
        
        
        
        
def strip(text: str):
    return textwrap.dedent(text).strip()


def diff(block_str: str, target: str):
    import difflib
    diff = difflib.unified_diff(
        strip(target).splitlines(),
        strip(block_str).splitlines(),
        fromfile='target',
        tofile='actual',
        lineterm=''
    )
    return "\n".join(diff)



def assert_render(block, target):
    render_out = block.render()
    print(render_out)
    try:
        assert render_out == strip(target)
    except AssertionError as e:
        print(diff(block, target))
        raise e