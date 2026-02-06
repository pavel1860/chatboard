import pytest
from typing import AsyncGenerator, Callable, Union, Optional, Any

from chatboard.prompt import component, stream

class TestException(Exception):
    pass

@stream()
async def basic_stream():
    yield "Hello "
    yield "world "
    raise TestException("test")
    yield "!"



@component()
async def basic_component():
    response = yield basic_stream()
    print(">>>",response)
    yield "Hello world"

@component()
async def basic_component_safe():
    
    try:
        response = yield basic_stream()
        print(">>>",response)
    except Exception as e:
        print("ERROR", e)
    
    yield "Hello world"


@pytest.mark.asyncio
async def test_basic_stream_exception():
    
    try:
        async for chunk in basic_component().stream_events():
            print(chunk)
        assert False
    except Exception as e:
        assert isinstance(e, TestException)
    
    
    try:
        async for chunk in basic_component_safe().stream_events():
            print(chunk)
        assert True
    except Exception as e:
        assert False