"""Tests for BlockSchema."""
import pytest
from chatboard.block.block12 import Block, BlockSchema
from pydantic import BaseModel, Field


class TestBlockSchemaCreation:
    """Tests for BlockSchema creation."""

    def test_create_schema(self):
        schema = BlockSchema('response', style='xml')
        assert schema is not None

    def test_create_schema_with_children(self):
        response_schema = BlockSchema('response', style='xml')
        response_schema /= BlockSchema('thinking')
        response_schema /= BlockSchema('answer')

        assert len(response_schema.children) == 2


class TestBlockSchemaInstantiate:
    """Tests for BlockSchema instantiation."""

    def test_instantiate_with_string(self):
        response_schema = BlockSchema('response', style='xml')
        response_schema /= BlockSchema('thinking')
        response_schema /= BlockSchema('answer')

        block = response_schema.instantiate('Hello World')

        assert block.text is not None
        assert block.tags is not None

    def test_instantiate_with_dict(self):
        response_schema = BlockSchema('response', style='xml')
        response_schema /= BlockSchema('thinking')
        response_schema /= BlockSchema('answer')

        block = response_schema.instantiate({
            'thinking': 'Let me think...',
            'answer': 'The answer is 42'
        })

        assert len(block.children) == 2


class TestBlockSchemaView:
    """Tests for BlockSchema view method."""

    def test_schema_view(self):
        with BlockSchema('response') as schema:
            with schema.view("thinking") as thinking:
                thinking /= "Let me think..."
            with schema.view("answer") as answer:
                answer /= "The answer is 42"

        blk = schema.transform()

        assert blk is not None


class TestBlockSchemaWithToolsIntegration:
    """Tests for BlockSchema with Pydantic tools."""

    def test_register_pydantic_model(self):
        class Attack(BaseModel):
            """Use this tool to attack."""
            weapon: str = Field(description="the weapon to use")
            target: str = Field(description="the target to attack")

        with Block.schema_view(tags=["output"]) as schema:
            with schema.view("thought", str) as t:
                t /= "think step by step"
            with schema.view("answer", str) as r:
                r /= "the answer goes here"
            with schema.view_list("tool", key="name") as tools:
                tools.register(Attack)

        assert tools is not None

    def test_instantiate_with_pydantic_tool(self):
        class Attack(BaseModel):
            """Use this tool to attack."""
            weapon: str = Field(description="the weapon to use")
            target: str = Field(description="the target to attack")

        with Block.schema_view(tags=["output"]) as schema:
            with schema.view("thought", str) as t:
                t /= "think step by step"
            with schema.view("answer", str) as r:
                r /= "the answer goes here"
            with schema.view_list("tool", key="name") as tools:
                tools.register(Attack)

        inst = tools.get_schema("attack").instantiate({
            "weapon": "sword",
            "target": "head"
        }, style="xml")

        assert inst is not None


class TestBlockSchemaComplexInstantiation:
    """Tests for complex schema instantiation."""

    def test_instantiate_with_tool_list(self):
        class DeleteBlock(BaseModel):
            """Use this action to delete a block."""
            block_path: str = Field(description="the path to delete")

        class EditBlock(BaseModel):
            """Use this action to edit a block."""
            block_path: str = Field(description="the path to edit")
            content: str = Field(description="the new content")

        with Block.schema_view(tags=["output"]) as schema:
            with schema.view("thought", str) as t:
                t /= "think step by step"
            with schema.view("answer", str) as r:
                r /= "the answer goes here"
            with schema.view_list("tool", key="name") as actions:
                actions.register(EditBlock)
                actions.register(DeleteBlock)

        inst = schema.instantiate({
            "thought": "the user wants to delete block 1.1",
            "answer": "here you go I have edited the document",
            "tool_list": [
                DeleteBlock(block_path="1.1"),
                EditBlock(block_path="1.2", content="the cat sat on the mat"),
            ]
        }, style="xml")

        expected = """<thought>
  the user wants to delete block 1.1
</thought>
<answer>
  here you go I have edited the document
</answer>
<tool name="delete_block">
  <block_path>
    1.1
  </block_path>
</tool>
<tool name="edit_block">
  <block_path>
    1.2
  </block_path>
  <content>
    the cat sat on the mat
  </content>
</tool>"""

        assert inst.render() == expected


class TestBlockWithViews:
    """Tests for block with nested views."""

    def test_nested_views_access(self):
        with Block("User Details", style="md li-num") as todos:
            with todos("Name", tags=["name"]) as name:
                name /= "John Doe"
            with todos("Email", tags=["email"]) as email:
                email /= "john.doe@example.com"

        # Access children by index
        assert todos.children[0].text == "Name"
        assert todos.children[0].children[0].text == "John Doe"
