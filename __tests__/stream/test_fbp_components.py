import pytest
import asyncio
import json
from pydantic import BaseModel

from chatboard.block import BlockChunk, BlockList
from chatboard.prompt.flow_components import StreamController, PipeController, Stream, Parser, Accumulator
from chatboard.prompt.parser import StreamEvent


class Response(BaseModel):
    content: str
    tools: list[str]


# Test data generators
async def hw_stream():
    """Simple hello world stream for testing"""
    yield BlockChunk("Hello")
    yield BlockChunk("World")


async def pirate_stream():
    """Pirate stream that reads from test data file"""
    with open("__tests__/data/pirate_xml_reasoning_stream.json", "r") as f:
        index = 0
        for line in f:
            block = BlockChunk.model_validate(json.loads(line))
            yield block
            await asyncio.sleep(0.02)
            index += 1


# Test prompt functions
async def prompt_span():
    """Test prompt that yields a StreamController and returns a Response"""
    response = yield StreamController(hw_stream)
    yield Response(content=str(response), tools=["attack"])


async def sub_agent():
    """Test sub-agent that processes multiple prompt spans"""
    for i in range(2):
        response = yield PipeController(prompt_span)
        for tool in response.tools:
            if tool == "attack":
                yield BlockChunk("Attack")
            elif tool == "defend":
                yield BlockChunk("Defend")
            elif tool == "heal":
                yield BlockChunk("Heal")


async def master_agent():
    """Test master agent that processes sub-agents"""
    for i in range(3):
        response = yield PipeController(sub_agent)
        yield BlockChunk(f"Master: {response.content}")



def assert_stream(chunks):
    assert len(chunks) == 4
    assert chunks[0].type == "stream_start"
    assert chunks[1].type == "stream_delta"
    assert chunks[2].type == "stream_delta"
    assert chunks[3].type == "stream_end"
    assert chunks[1].name == "hw_stream"
    assert chunks[1].payload == BlockChunk("Hello")
    assert chunks[2].payload == BlockChunk("World")
    assert chunks[3].payload is None
    
    
def assert_prompt_span(chunks):
    assert len(chunks) == 6
    assert chunks[0].type == "span_start"
    assert chunks[-1].type == "span_end"
    assert chunks[-1].payload == Response(content = "Hello World", tools = ["attack"])
    assert_stream(chunks[1:5])



def assert_sub_agent(chunks):
    assert len(chunks) == 15
    assert chunks[0].type == "span_start"
    assert chunks[0].name == "sub_agent"
    assert_prompt_span(chunks[1:7])
    assert chunks[7].type == "span_value"
    assert chunks[7].name == "sub_agent"
    assert_prompt_span(chunks[8:14])
    assert chunks[14].type == "span_end"
    assert chunks[14].name == "sub_agent"
    assert chunks[14].payload == BlockChunk("Attack")
    
    

class TestStreamController:
    """Test cases for StreamController"""

    @pytest.mark.asyncio
    async def test_stream_controller_basic(self):
        """Test basic StreamController functionality"""
        chunks = [chunk async for chunk in StreamController(hw_stream)]
        
        assert len(chunks) == 4
        assert chunks[0].type == "stream_start"
        assert chunks[1].type == "stream_delta"
        assert chunks[2].type == "stream_delta"
        assert chunks[3].type == "stream_end"
        assert chunks[1].name == "hw_stream"
        assert chunks[1].payload == BlockChunk("Hello")
        assert chunks[2].payload == BlockChunk("World")
        assert chunks[3].payload is None

    @pytest.mark.asyncio
    async def test_stream_controller_accumulator(self):
        """Test that StreamController properly accumulates results"""
        controller = StreamController(hw_stream)
        chunks = [chunk async for chunk in controller]
        
        # Verify chunks are emitted correctly
        assert len(chunks) == 4
        assert chunks[0].type == "stream_start"
        assert chunks[3].type == "stream_end"
        
        # Verify accumulator contains the expected blocks
        assert len(controller.acc) == 2
        assert controller.acc[0] == BlockChunk("Hello")
        assert controller.acc[1] == BlockChunk("World")

    @pytest.mark.asyncio
    async def test_stream_controller_index(self):
        """Test that StreamController tracks index correctly"""
        controller = StreamController(hw_stream)
        chunks = [chunk async for chunk in controller]
        
        # Index should be 2 after processing 2 blocks
        assert controller.index == 2


class TestPipeController:
    """Test cases for PipeController"""

    @pytest.mark.asyncio
    async def test_pipe_controller_prompt_span(self):
        """Test PipeController with a simple prompt span"""
        chunks = [chunk async for chunk in PipeController(prompt_span)]
        assert_prompt_span(chunks)

    @pytest.mark.asyncio
    async def test_pipe_controller_sub_agent(self):
        """Test PipeController with sub-agent that has multiple iterations"""
        chunks = [chunk async for chunk in PipeController(sub_agent)]        
        assert_sub_agent(chunks)

    @pytest.mark.asyncio
    async def test_pipe_controller_master_agent(self):
        """Test PipeController with master agent"""
        chunks = [chunk async for chunk in PipeController(master_agent)]
        
        # Should have span_start, multiple nested spans, and span_end
        assert len(chunks) > 0
        assert chunks[0].type == "span_start"
        assert chunks[0].name == "master_agent"
        assert chunks[-1].type == "span_end"
        assert chunks[-1].name == "master_agent"


class TestStreamComponents:
    """Test cases for individual stream components"""

    @pytest.mark.asyncio
    async def test_stream_component(self):
        """Test the Stream component directly"""
        stream = Stream(hw_stream())
        chunks = []
        
        async for chunk in stream:
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == BlockChunk("Hello")
        assert chunks[1] == BlockChunk("World")
        assert stream._index == 2

    @pytest.mark.asyncio
    async def test_accumulator_component(self):
        """Test the Accumulator component directly"""
        stream = Stream(hw_stream())
        accumulator = Accumulator(BlockList())
        accumulator._gen = stream
        
        chunks = []
        async for chunk in accumulator:
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert len(accumulator.result) == 2
        assert accumulator.result[0] == BlockChunk("Hello")
        assert accumulator.result[1] == BlockChunk("World")


class TestIntegration:
    """Integration tests for the complete flow"""

    @pytest.mark.asyncio
    async def test_complete_flow_with_pirate_stream(self):
        """Test complete flow with actual pirate stream data"""
        # This test uses the real pirate stream data
        controller = StreamController(pirate_stream)
        chunks = []
        
        async for chunk in controller:
            chunks.append(chunk)
        
        # Should have start, multiple deltas, and end
        assert len(chunks) > 2
        assert chunks[0].type == "stream_start"
        assert chunks[-1].type == "stream_end"
        
        # Should have accumulated blocks
        assert len(controller.acc) > 0

    @pytest.mark.asyncio
    async def test_nested_controllers(self):
        """Test nested StreamController and PipeController interactions"""
        async def nested_prompt():
            inner_response = yield StreamController(hw_stream)
            yield Response(content=f"Nested: {str(inner_response)}", tools=["nested"])
        
        chunks = [chunk async for chunk in PipeController(nested_prompt)]
        
        assert len(chunks) > 0
        assert chunks[0].type == "span_start"
        assert chunks[-1].type == "span_end"
        
        # Should contain nested stream events
        stream_events = [chunk for chunk in chunks if chunk.type in ["stream_start", "stream_delta", "stream_end"]]
        assert len(stream_events) > 0


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_stream_controller_with_empty_stream(self):
        """Test StreamController with an empty stream"""
        async def empty_stream():
            if False:
                yield BlockChunk("This won't be yielded")
        
        controller = StreamController(empty_stream)
        chunks = [chunk async for chunk in controller]
        
        assert len(chunks) == 2
        assert chunks[0].type == "stream_start"
        assert chunks[1].type == "stream_end"
        assert len(controller.acc) == 0

    @pytest.mark.asyncio
    async def test_pipe_controller_with_empty_prompt(self):
        """Test PipeController with an empty prompt"""
        async def empty_prompt():
            yield Response(content="Empty", tools=[])
        
        chunks = [chunk async for chunk in PipeController(empty_prompt)]
        
        assert len(chunks) == 2
        assert chunks[0].type == "span_start"
        assert chunks[1].type == "span_end"
        assert chunks[1].payload == Response(content="Empty", tools=[])


# Helper functions for assertions
def assert_stream_structure(chunks):
    """Helper to assert basic stream structure"""
    assert len(chunks) >= 2
    assert chunks[0].type == "stream_start"
    assert chunks[-1].type == "stream_end"


def assert_span_structure(chunks, name):
    """Helper to assert basic span structure"""
    assert len(chunks) >= 3
    assert chunks[0].type == "span_start"
    assert chunks[0].name == name
    assert chunks[-1].type == "span_end"
    assert chunks[-1].name == name
