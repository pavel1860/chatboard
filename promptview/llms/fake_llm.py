from datetime import datetime
import json
from typing import List, Type
import openai
import os

from pydantic import BaseModel
from ..block import BlockChunk, BlockList
from ..block.util import LLMEvent, ToolCall
from ..context.execution_context import ExecutionContext
from .llm2 import LLMStreamController, LlmConfig
from openai.types.chat import ChatCompletionMessageParam


from ..utils.model_utils import schema_to_function



class FakeLLM(LLMStreamController):
    name: str = "OpenAiLLM"    
    model: str | None = "pirate_stream.json"
    models = ["pirate_stream.json", "pirate_xml_reasoning_stream.json"]
    
    
    def __init__(self, blocks: BlockList, config: LlmConfig):
        super().__init__(blocks, config)
        self.client = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))
    
    def to_message(self, content: str, role: str, tool_calls: List[ToolCall] | None = None, tool_call_id: str | None = None, name: str | None = None) -> ChatCompletionMessageParam:
        message = {
            "role": role or "user",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                        "type": "function",
                        "function": {
                            # "arguments": json.dumps(action_call.action.model_dump()),
                            "arguments": tool_call.to_json(),
                            "name": tool_call.name
                        }                      
                } for tool_call in tool_calls
            ]
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name:
            message["name"] = name        
        return message
    
    def to_tool(self, tool: Type[BaseModel]) -> dict:
        schema = schema_to_function(tool)
        return schema
    
    async def stream(self):
        with open(f"__tests__/data/{self.model}", "r") as f:
            for line in f:
                block = BlockChunk.model_validate(json.loads(line))
                yield block