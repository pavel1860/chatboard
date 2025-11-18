from datetime import datetime
from typing import List, Type
import openai
import os

from pydantic import BaseModel
from ..block import BlockChunk, BlockList
from ..block.util import LLMEvent, ToolCall
from ..context.execution_context import ExecutionContext
from ..llms.llm2 import BaseLLM, LLMStream, LlmConfig, llm_stream
from openai.types.chat import ChatCompletionMessageParam
from ..utils.model_utils import schema_to_function
from ..tracer.langsmith_tracer import Tracer


class OpenAiLLM(BaseLLM):
    name: str = "OpenAiLLM"
    default_model: str= "gpt-4o"
    models = ["gpt-4o", "gpt-4o-mini"]
    client: openai.AsyncClient
    
    def __init__(self, config: LlmConfig | None = None):
        super().__init__(config)
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
    
    @llm_stream(name="openai_llm")
    async def stream(
        self, 
        blocks: BlockList,
        config: LlmConfig,
        tools: List[Type[BaseModel]] | None = None
    ):
        
            
        messages = [
            # self.to_message(b.render(), role=b.role or "user", tool_calls=b.tool_calls, tool_call_id=b.id)
            self.to_message(b.render(), role=b.role or "user", tool_call_id=b.id)
            for b in blocks
        ]
        llm_tools = None
        tool_choice = None
        if tools:
            llm_tools = [self.to_tool(tool) for tool in tools]        
            tool_choice = config.tool_choice

        with Tracer(
            run_type="llm",
            name=self.__class__.__name__,
            inputs={"messages": messages},
            metadata={
                "ls_model_name": config.model,
                "ls_model_version": "openai",
            },
        ) as llm_run: 
            full_content = ""
            try:
                
                res_stream = await self.client.chat.completions.create(
                        messages=messages,
                        tools=llm_tools,
                        model=config.model,
                        tool_choice=tool_choice,
                        stream=True,
                        logprobs=True,                
                    )
                            
                async for chunk in res_stream:                
                    if chunk.choices[0].delta:
                        choice = chunk.choices[0]                   
                        content = choice.delta.content
                        if content is not None:
                            full_content += content
                        if content is None:
                            continue
                        try:
                            if choice.logprobs and choice.logprobs.content:                
                                logprob = choice.logprobs.content[0].logprob
                            else:
                                logprob = 0  
                        except:
                            raise ValueError("No logprobs")        
                        blk_chunk = BlockChunk(content, logprob=logprob)
                        yield blk_chunk
                llm_run.end(outputs={"content": full_content})
                return
            except Exception as e:
                llm_run.end(outputs={"content": full_content}, errors=str(e))
                raise e
                    
                    
        # except Exception as e:
            # yield = Block(str(e))
            # event = Event(type="stream_error", payload=block, timestamp=int(datetime.now().timestamp()), request_id=request_id)
            # yield event
            # raise e
