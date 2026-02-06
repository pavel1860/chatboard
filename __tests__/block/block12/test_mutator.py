"""Tests for Block mutator system."""
import pytest
from chatboard.block.block12 import Block, MutatorMeta


class TestMutatorRegistry:
    """Tests for mutator registry."""

    def test_list_styles(self):
        styles = MutatorMeta.list_styles()
        assert "xml" in styles

    def test_get_mutator(self):
        xml_mutator = MutatorMeta.get_mutator("xml")
        assert xml_mutator is not None

    def test_get_unknown_mutator_returns_base(self):
        # Unknown styles return the base Mutator class
        from chatboard.block.block12.mutator import Mutator
        unknown = MutatorMeta.get_mutator("unknown")
        assert unknown is Mutator


class TestXmlMutator:
    """Tests for XML mutator via high-level Block API."""

    def test_xml_block_rendering(self):
        # Test XML rendering via transform
        with Block("Item", tags=["item"], style="xml") as b:
            b /= "Hello World"

        blk = b.transform()
        rendered = blk.render()

        assert "<item>" in rendered
        assert "</item>" in rendered
        assert "Hello World" in rendered

    def test_xml_nested_structure(self):
        with Block("outer", tags=["outer"], style="xml") as b:
            with b("inner", tags=["inner"], style="xml") as inner:
                inner /= "Content"

        blk = b.transform()
        rendered = blk.render()

        assert "<outer>" in rendered
        assert "<inner>" in rendered
        assert "Content" in rendered

    def test_xml_schema_transform(self):
        from chatboard.block.block12 import BlockSchema

        with BlockSchema('response', style='xml') as schema:
            with schema.view("thinking", style="xml") as thinking:
                thinking /= "Let me think..."
            with schema.view("answer", style="xml") as answer:
                answer /= "The answer is 42"

        blk = schema.transform()
        rendered = blk.render()

        assert "<thinking>" in rendered
        assert "<answer>" in rendered


class TestStyleApplication:
    """Tests for style application."""

    def test_list_style(self):
        with Block("ToDos", style="li") as todos:
            todos /= "Buy groceries"
            todos /= "Finish project"
            todos /= "Call the bank"

        assert len(todos.children) == 3

    def test_md_list_num_style(self):
        with Block("ToDos", style="md li-num") as todos:
            todos /= "Buy groceries"
            todos /= "Finish project"
            todos /= "Call the bank"

        assert len(todos.children) == 3

    def test_banner_style(self):
        with Block() as block:
            with block("Cats", style="banner") as cats:
                cats /= "Black Cats"
                cats /= "White Cats"
            with block("Dogs", style="banner") as dogs:
                dogs /= "Black Dogs"
                dogs /= "White Dogs"

        assert len(block.children) == 2
        assert len(cats.children) == 2
        assert len(dogs.children) == 2
