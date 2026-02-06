import pytest
from typing import AsyncGenerator, Callable, Union, Optional, Any

from chatboard.prompt.stream2 import StreamController, stream  # Adjust the import
from chatboard.prompt import component, Depends
from chatboard.llms import LLMRegistry
from chatboard.block import Block, BlockList, BlockSent, BlockChunk
from chatboard.model3.versioning.models import Branch
from chatboard.model3 import NamespaceManager

@pytest.mark.asyncio
async def test_basic_stream():
    async def gen():
        yield "A"
        yield "B"

    wrapper = StreamController(gen(), accumulator=str)
    chunks = []
    async for chunk in wrapper:
        chunks.append(chunk)

    assert chunks == ["A", "B"]
    assert await StreamController(gen(), accumulator=str) == "AB"

@pytest.mark.asyncio
async def test_nested_stream():
    async def inner():
        yield "X"
        yield "Y"

    async def outer():
        yield "Start"
        yield inner()
        yield "End"

    wrapper = StreamController(outer(), accumulator=str)
    result = []
    async for token in wrapper:
        result.append(token)

    assert result == ["Start", "X", "Y", "End"]
    assert await StreamController(outer(), accumulator=str) == "StartXYEnd"

@pytest.mark.asyncio
async def test_stream_events():
    async def gen():
        yield "A"
        yield "B"

    stream = StreamController(gen(), accumulator=str)
    events = []
    async for event in stream.stream_events():
        events.append(event)

    assert events[0]["type"] == "stream_start"
    assert events[1]["type"] == "stream_delta"
    assert events[1]["value"] == "A"
    assert events[2]["type"] == "stream_delta"
    assert events[2]["value"] == "B"
    assert events[3]["type"] == "stream_end"
    assert events[3]["final_value"] == "AB"

@pytest.mark.asyncio
async def test_custom_accumulator():
    async def gen():
        yield "x"
        yield "y"

    class MyCollector:
        def __init__(self):
            self.data = []

        def append(self, x):
            self.data.append(x)

    stream = StreamController(gen(), accumulator=MyCollector)
    result = await stream
    assert isinstance(result, MyCollector)
    assert result.data == ["x", "y"]

@pytest.mark.asyncio
async def test_stream_decorator():
    @stream(str)
    async def llm():
        yield "Hi"
        response = yield " there"
        yield response + "!"  # simulate return

    chunks = []
    async for token in llm():
        chunks.append(token)

    assert chunks == ["Hi", " there", "Hi there!"]
    assert await llm() == "Hi there!"

@pytest.mark.asyncio
async def test_stream_non_appendable_handling():
    async def gen():
        yield 42  # not appendable to str
        yield "B"

    stream = StreamController(gen(), accumulator=str)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [42, "B"]
    assert await StreamController(gen(), accumulator=str) == "B"









@pytest.mark.asyncio
async def test_basic_component():
    await NamespaceManager.initialize_all()
    @component()
    async def test_city_prompt(message: Block, llm: LLMRegistry = Depends(LLMRegistry)):
        with Block(role="system") as s:
            s /= "answer as short as possible"
            s /= "no unneccesary words"
            with s.response_schema() as r:
                r /= "you must use the following format"
                with r.field("output", str) as o:
                    o /= "your answer goes here"
                    
        res = yield llm.stream(s, message).load("test_city_prompt", "__tests__/data").parse(r)
        yield res
        

    branch = await Branch.get_main()
    message = Block("what is the capital of France, Italy and Germany?")
    events = []
    with branch:
        async with branch.start_turn() as turn:
            async for e in test_city_prompt(message=message).stream_events():
                print(e)
                events.append(e)
                
                
    assert events[0].type == "span_start"
    assert events[0].path == [0]
    assert events[0].name == "test_city_prompt"
    assert events[1].type == "stream_start"
    assert events[1].path == [0, 0]
    assert events[1].name == "openai_llm"
    assert events[2].type == "stream_delta"   
    assert isinstance(events[2].payload,  Block)
    assert events[3].type == "stream_delta"    
    assert isinstance(events[3].payload, BlockSent)
    assert events[3].payload.path == [1, 1]
    assert events[4].type == "stream_delta"    
    assert isinstance(events[4].payload, BlockChunk)
    assert events[5].type == "stream_delta"    
    assert isinstance(events[5].payload, BlockChunk)
    assert events[6].type == "stream_delta"    
    assert isinstance(events[6].payload, BlockChunk)
    assert events[7].type == "stream_delta"    
    assert isinstance(events[7].payload, BlockChunk)
    assert events[8].type == "stream_delta"
    assert isinstance(events[8].payload, BlockChunk)
    assert events[9].type == "stream_delta"
    assert isinstance(events[9].payload, BlockSent)
    assert events[9].path == [0, 0]    
    assert events[10].type == "stream_end"    
    assert events[10].path == [0, 0]
    assert events[11].type == "span_value"    
    assert events[11].path == [1]
    assert events[12].type == "span_end" 
    assert events[12].path == [1]   


        
    
        
        
        
            