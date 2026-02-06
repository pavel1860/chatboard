
import pytest
from pydantic import Field, BaseModel
import pytest_asyncio
from chatboard.llms.clients.openai_client import OpenAiLlmClient
from chatboard.llms.clients.anthropic_client import AnthropicLlmClient

from chatboard.llms.messages import HumanMessage



class CatAction(BaseModel):
    """use this if the user asks about cats"""
    text: str = Field(..., title="Text", description="write a funny sentence about cats")



class NumAction(BaseModel):
    """use this if the user asks about numbers"""
    num: int = Field(..., title="Text", description="a number between 1 and 10", gt=0, lt=11)
    
    


@pytest.mark.asyncio
async def test_openai_client():
    client = OpenAiLlmClient()
    content = "hello" 
    response = await client.complete([HumanMessage(content=content)], actions=[CatAction, NumAction])
    assert len(response.content) > 0
    assert len(response.action_calls) == 0
    content = "tell me about cats"
    response = await client.complete([HumanMessage(content=content)], actions=[CatAction, NumAction])
    assert len(response.action_calls) == 1
    assert isinstance(response.output, CatAction) 
    



@pytest.mark.asyncio
async def test_anthropic_client():
    client = AnthropicLlmClient()
    content = "hello" 
    response = await client.complete([HumanMessage(content=content)], actions=[CatAction, NumAction])
    assert len(response.content) > 0
    assert len(response.action_calls) == 0
    content = "tell me about cats"
    response = await client.complete([HumanMessage(content=content)], actions=[CatAction, NumAction])
    assert len(response.action_calls) == 1
    assert isinstance(response.output, CatAction) 
    
    