"""Tests for XML rendering."""
import pytest
from chatboard.block.block12 import Block, BlockSchema


class TestXmlRendering:
    """Tests for XML style rendering."""

    def test_xml_basic(self):
        with Block("Item", tags=["item"], style="xml") as b:
            b /= "Hello World"

        blk = b.transform()

        assert blk is not None

    def test_xml_nested(self):
        with Block("Item", tags=["item"], style="xml") as b:
            with b("name", tags=["name"], style="xml") as name:
                name /= "Hello World"

        blk = b.transform()

        assert blk is not None


class TestXmlSchema:
    """Tests for XML schema rendering."""

    def test_schema_xml_style(self):
        with BlockSchema('response', role="assistant", style='xml') as schema:
            with schema.view("thinking", tags=["thinking"], style="xml") as thinking:
                thinking /= "Let me analyze this problem..."
            with schema.view("answer", tags=["answer"], style="xml") as answer:
                answer /= "The answer is 42"

        blk = schema.transform()

        assert blk is not None
        assert len(schema.children) == 2

    def test_schema_print_debug(self):
        with BlockSchema('response', role="assistant", style='xml') as schema:
            with schema.view("thinking", tags=["thinking"], style="xml") as thinking:
                thinking /= "Let me analyze this problem..."
            with schema.view("answer", tags=["answer"], style="xml") as answer:
                answer /= "The answer is 42"

        blk = schema.transform()

        assert blk.children[0].text is not None
        assert blk.children[1].text is not None


class TestBlockExtract:
    """Tests for extract functionality."""

    def test_extract_basic(self):
        with Block("Item", tags=["item"], style="xml") as b:
            b /= "Hello World"

        tb = b.transform()
        extracted = tb.extract()

        assert extracted is not None
