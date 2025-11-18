import json
from typing import Any, List, Type, TypedDict, cast, Literal
import anthropic
from pydantic import Field, BaseModel
from ..llms.llm import LLM, BaseLlmClient
from ..llms.messages import ActionCall, LlmUsage
from ..llms.utils.action_manager import Actions
from ..prompt.legacy.block1 import BaseBlock, ResponseBlock, ActionBlock
import os

class TextBlock(TypedDict):
    type: Literal["text"]
    text: str

class ToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    parameters: dict[str, Any]

ContentBlock = TextBlock | ToolUseBlock

class AnthropicLLM2(LLM[anthropic.AsyncAnthropic, Any, anthropic.types.message.Message]):
    name: str = "AnthropicLLM2"
    client: anthropic.AsyncAnthropic = Field(default_factory=lambda: anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY")))
    model: str = "claude-3-5-sonnet-20240620"
    
    beta: list[str] | None = None

    def to_message(self, block: BaseBlock) -> dict[str, Any]:
        if block.role == "user" or block.role == "system":
            return {
                "role": block.role,
                "content": block.render()
            }
        elif block.role == "assistant":
            msg: dict[str, Any] = {"role": block.role}
            if isinstance(block, ResponseBlock) and block.action_calls:
                content_blocks: List[ContentBlock] = [{"type": "text", "text": block.render()}]
                for action_call in block.action_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": action_call.id,
                        "name": action_call.name,
                        "parameters": action_call.action.model_dump() if isinstance(action_call.action, BaseModel) else action_call.action
                    })
                msg["content"] = content_blocks
            else:
                msg["content"] = block.render()
            return msg
        elif block.role == "tool":
            tool_block = cast(ActionBlock, block)
            return {
                "role": "tool",
                "content": tool_block.render(),
                "tool_call_id": tool_block.tool_call_id
            }
        raise ValueError(f"Unsupported role: {block.role}")

    def to_tools(self, actions: List[Type[BaseModel]]) -> List[dict] | None:
        if not actions:
            return None
        return Actions(actions=actions).to_anthropic()

    async def complete(
        self,
        messages: List[dict],
        tools: List[dict] | None = None,
        model: str | None = None,
        tool_choice: str | None = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> anthropic.types.message.Message:
        system = anthropic.NOT_GIVEN
        if messages and messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages = messages[1:]

        completion_args = {
            "model": model or self.model,
            "messages": messages,
            "system": system,
            "tools": tools or anthropic.NOT_GIVEN,
            "tool_choice": tool_choice or anthropic.NOT_GIVEN,
            "max_tokens": max_tokens,
            **kwargs
        }
        if self.beta:
            completion_args["betas"] = self.beta

        return await self.client.messages.create(**completion_args)

    def parse_response(self, response: anthropic.types.message.Message, actions: List[Type[BaseModel]] | None) -> ResponseBlock:
        content = ""
        action_calls = []
        
        for content_block in response.content:
            if content_block.type == "text":
                content = content_block.text
            elif content_block.type == "tool_use" and actions:
                action_instance = Actions(actions=actions).from_anthropic(content_block)
                action_calls.append(
                    ActionCall(
                        id=content_block.id,
                        name=content_block.name,
                        action=action_instance
                    )
                )

        return ResponseBlock(
            id=response.id,
            model=response.model,
            content=content,
            action_calls=action_calls,
            raw=response,
            usage=LlmUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens
            )
        )
