"""Tests for XmlParser class."""
import pytest
from promptview.block.block11 import (
    Block,
    BlockSchema,
    BlockListSchema,
    XmlParser,
    ParserError,
    Chunk,
)


class TestXmlParserCreation:
    """Tests for XmlParser creation."""

    def test_create_parser(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        assert parser.schema is schema
        assert parser.result is None

    def test_parser_with_wrapper_schema(self):
        schema = BlockSchema("response")
        schema /= BlockSchema("thinking")
        schema /= BlockSchema("answer")
        parser = XmlParser(schema)

        assert parser._has_synthetic_root is True


class TestXmlParserBasicParsing:
    """Tests for basic XML parsing."""

    def test_parse_simple_tag(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed_str("<response>Hello World</response>")
        parser.close()

        assert parser.result is not None
        assert parser.result.content == "Hello World"

    def test_parse_nested_tags(self):
        schema = BlockSchema("response")
        schema /= BlockSchema("thinking")
        schema /= BlockSchema("answer")
        parser = XmlParser(schema)

        parser.feed_str("<thinking>Let me think</thinking>")
        parser.feed_str("<answer>42</answer>")
        parser.close()

        result = parser.result
        assert result is not None
        assert len(result.children) == 2
        assert result.children[0].content == "Let me think"
        assert result.children[1].content == "42"

    def test_parse_with_attributes(self):
        schema = BlockSchema("item")
        parser = XmlParser(schema)

        parser.feed_str('<item id="1">Content</item>')
        parser.close()

        result = parser.result
        assert result is not None
        assert result.attrs.get("id") == "1"


class TestXmlParserStreaming:
    """Tests for streaming XML parsing."""

    def test_parse_chunked_content(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed_str("<response>")
        parser.feed_str("Hello ")
        parser.feed_str("World")
        parser.feed_str("</response>")
        parser.close()

        assert parser.result is not None
        assert parser.result.content == "Hello World"

    def test_parse_chunked_tag(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed_str("<res")
        parser.feed_str("ponse>Content</re")
        parser.feed_str("sponse>")
        parser.close()

        assert parser.result is not None
        assert parser.result.content == "Content"

    def test_parse_with_chunk_objects(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed(Chunk(content="<response>"))
        parser.feed(Chunk(content="Hello"))
        parser.feed(Chunk(content="</response>"))
        parser.close()

        assert parser.result is not None
        assert parser.result.content == "Hello"


class TestXmlParserErrors:
    """Tests for parser error handling."""

    def test_unknown_tag_raises_error(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        with pytest.raises(ParserError) as exc_info:
            parser.feed_str("<unknown>Content</unknown>")
            parser.close()

        assert "Unknown tag" in str(exc_info.value)

    def test_mismatched_closing_tag_raises_error(self):
        schema = BlockSchema("response")
        schema /= BlockSchema("thinking")
        parser = XmlParser(schema)

        with pytest.raises(ParserError) as exc_info:
            parser.feed_str("<thinking>Content</answer>")
            parser.close()

        assert "Mismatched" in str(exc_info.value)

    def test_invalid_xml_raises_error(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        with pytest.raises(ParserError):
            parser.feed_str("<response><unclosed>")
            parser.close()


class TestXmlParserStack:
    """Tests for parser stack management."""

    def test_current_block(self):
        schema = BlockSchema("response")
        schema /= BlockSchema("thinking")
        parser = XmlParser(schema)

        parser.feed_str("<thinking>")
        assert parser.current_block is not None
        assert parser.current_schema.name == "thinking"

        parser.feed_str("</thinking>")
        assert parser.current_schema is None or parser.current_schema.name == "response"

    def test_nested_stack(self):
        schema = BlockSchema("response")
        child = BlockSchema("section")
        child /= BlockSchema("content")
        schema /= child
        parser = XmlParser(schema)

        parser.feed_str("<section>")
        assert parser.current_schema.name == "section"

        parser.feed_str("<content>")
        assert parser.current_schema.name == "content"

        parser.feed_str("</content>")
        assert parser.current_schema.name == "section"

        parser.feed_str("</section>")


class TestXmlParserWithListSchema:
    """Tests for parsing with BlockListSchema."""

    def test_parse_list_items(self):
        list_schema = BlockListSchema("tool", name="tools")
        schema = BlockSchema("response")
        schema.append_child(list_schema)

        parser = XmlParser(schema)
        parser.feed_str("<tools>")
        parser.feed_str("<tool>Hammer</tool>")
        parser.feed_str("<tool>Wrench</tool>")
        parser.feed_str("</tools>")
        parser.close()

        result = parser.result
        assert result is not None


class TestXmlParserWhitespace:
    """Tests for whitespace handling."""

    def test_skip_whitespace_between_tags(self):
        schema = BlockSchema("response")
        schema /= BlockSchema("a")
        schema /= BlockSchema("b")
        parser = XmlParser(schema)

        parser.feed_str("<a>First</a>")
        parser.feed_str("   \n\t   ")  # Whitespace between tags
        parser.feed_str("<b>Second</b>")
        parser.close()

        result = parser.result
        assert len(result.children) == 2
        assert result.children[0].content == "First"
        assert result.children[1].content == "Second"

    def test_preserve_content_whitespace(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed_str("<response>  Hello  World  </response>")
        parser.close()

        # Internal whitespace should be preserved
        assert "Hello" in parser.result.content
        assert "World" in parser.result.content


class TestXmlParserAlternativeTags:
    """Tests for schemas with alternative tags."""

    def test_parse_with_alternative_tag(self):
        schema = BlockSchema("response", tags=["reply", "answer"])
        parser = XmlParser(schema)

        parser.feed_str("<reply>Hello</reply>")
        parser.close()

        assert parser.result is not None
        assert parser.result.content == "Hello"


class TestXmlParserProperties:
    """Tests for parser properties."""

    def test_result_property(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        assert parser.result is None

        parser.feed_str("<response>Test</response>")
        parser.close()

        assert parser.result is not None

    def test_current_block_property(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        assert parser.current_block is None

        parser.feed_str("<response>")
        assert parser.current_block is not None

    def test_current_schema_property(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        assert parser.current_schema is None

        parser.feed_str("<response>")
        assert parser.current_schema is not None
        assert parser.current_schema.name == "response"


class TestXmlParserEmptyContent:
    """Tests for empty content handling."""

    def test_parse_empty_tag(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        parser.feed_str("<response></response>")
        parser.close()

        assert parser.result is not None
        assert parser.result.content == ""

    def test_parse_self_closing_tag(self):
        schema = BlockSchema("response")
        parser = XmlParser(schema)

        # Note: expat handles self-closing tags as start + end
        parser.feed_str("<response/>")
        parser.close()

        assert parser.result is not None
