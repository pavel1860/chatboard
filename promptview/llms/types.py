from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Type, Union
from pydantic import BaseModel, Field


# ToolChoice = Literal['auto', 'required', 'none']




ToolChoice = Literal['auto', 'required', 'none']

class LlmConfig(BaseModel):
    model: str | None = Field(default=None, description="The model to use")
    temperature: float = Field(default=0, ge=0, le=1, description="The temperature of the response")
    max_tokens: int | None = Field(default=None, ge=0, description="The maximum number of tokens in the response")
    stop_sequences: List[str] | None = Field(default=None, description="The stop sequences of the response")
    stream: bool = Field(default=False, description="If the response should be streamed")
    logit_bias: Dict[str, int] | None = Field(default=None, description="The logit bias of the response")
    top_p: float = Field(default=1, ge=0, le=1, description="The top p of the response")
    presence_penalty: float | None = Field(default=None, ge=-2, le=2, description="The presence penalty of the response")
    logprobs: bool | None = Field(default=None, description="If the logprobs should be returned")
    seed: int | None = Field(default=None, description="The seed of the response")
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2, description="The frequency penalty of the response")
    tools: List[Type[BaseModel]] | None = Field(default=None, description="The tools of the response")
    retries: int = Field(default=3, ge=1, description="The number of retries")
    parallel_tool_calls: bool = Field(default=False, description="If the tool calls should be parallel")
    tool_choice: ToolChoice | None = Field(default=None, description="The tool choice of the response")



class LLMResponse(BaseModel):
    id: str
    item_id: str


class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class ErrorMessage(Exception):
    
    def __init__(self, error_content: str, should_retry: bool = True) -> None:
        self.error_content = error_content
        self.should_retry = should_retry
        super().__init__(f"Output parsing error: {error_content}")
        
    def to_block(self, output_model: Type[BaseModel] | None = None, role: str = "user", tags: List[str] = ["error"]) -> "BlockChunk":
        from ..block import BlockChunk
        with BlockChunk(tags=tags, role=role) as b:
            b /= self.error_content
            if output_model:
                b /= "do not add any other text or apologies"
                b /= "use the output format as provided to generate your answer"                
        return b.root
    