import pytest

from chatboard.text.llms.mvc import View, Field, Action
from chatboard.text.llms.view_renderer import ViewRenderer



@pytest.mark.asyncio
async def test_basic_view_rendering():
    class TestView(View):
        name: str

    renderer = ViewRenderer()

    view = TestView(name="test")
    output = await renderer.render_view(view) 
    assert output == """{"name": "test"}"""


@pytest.mark.asyncio
async def test_basic_system_rendering():
    class TestView(View):
        name: str
    

    renderer = ViewRenderer()

    view = TestView(name="test")
    output = await renderer.render_view(view, render_system=True) 
    assert output.system_prompt == '{"type": "function", "function": {"name": "TestView", "description": "", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}\n'


    class TestView(View):    
        name: str = Field(..., description="The name of the user")

    view = TestView(name="test")

    renderer = ViewRenderer(system_indent=None)

    output = await renderer.render_view(view, render_system=True) 
    output
    assert "The name of the user" in output.system_prompt

    class TestView(View):
        """A test view"""    
        name: str = Field(..., description="The name of the user")

    
    output = await renderer.render_view(view, render_system=True) 
    output
    assert "A test view" in output.system_prompt

    renderer = ViewRenderer(system_indent=4)

    output = await renderer.render_view(view, render_system=True) 
    assert output.system_prompt == """{
        "type": "function",
        "function": {
            "name": "TestView",
            "description": "A test view",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "description": "The name of the user",
                        "type": "string"
                    }
                },
                "required": [
                    "name"
                ]
            }
        }
    }"""



@pytest.mark.asyncio
async def test_basic_actions_rendering():
    class TestAction1(Action):
        age1: int = Field(..., description="The age of the user")
        is_male1: bool = Field(..., description="Whether the the user is male")

    class TestAction2(Action):
        age2: int
        is_male2: bool


    class TestView(View):
        """this is a test"""
        name: str = Field(..., description="The name of the user")

        _actions = [TestAction1, TestAction2]

        class Config:
            title = "Car Model"


    view = TestView(name="test")

    renderer = ViewRenderer(
            system_indent=4, 
            # system_to_prompt=True
        )
    
    output = await renderer.render_view(view, render_tool=True) 
    assert output == """{
    "type": "function",
    "function": {
        "name": "Car Model",
        "description": "this is a test",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "description": "The name of the user",
                    "type": "string"
                }
            },
            "required": [
                "name"
            ]
        }
    }
}"""
