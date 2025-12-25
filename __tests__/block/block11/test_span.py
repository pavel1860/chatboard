"""Tests for Chunk and Span classes."""
import pytest
from promptview.block.block11 import Chunk, Span


class TestChunk:
    """Tests for Chunk class."""

    def test_create_chunk(self):
        chunk = Chunk("Hello")
        assert chunk.content == "Hello"
        assert chunk.logprob is None
        assert len(chunk) == 5

    def test_chunk_with_logprob(self):
        chunk = Chunk("token", logprob=-0.5)
        assert chunk.content == "token"
        assert chunk.logprob == -0.5

    def test_chunk_is_line_end(self):
        chunk1 = Chunk("Hello")
        chunk2 = Chunk("World\n")
        assert chunk1.is_line_end is False
        assert chunk2.is_line_end is True

    def test_chunk_copy(self):
        original = Chunk("test", logprob=-0.1)
        copy = original.copy()

        assert copy.content == original.content
        assert copy.logprob == original.logprob
        assert copy.id != original.id  # New ID
        assert copy is not original

    def test_chunk_repr(self):
        chunk = Chunk("Hello")
        assert repr(chunk) == "Chunk('Hello')"


class TestSpan:
    """Tests for Span class."""

    def test_empty_span(self):
        span = Span()
        assert span.is_empty is True
        assert span.text == ""
        assert span.prefix_text == ""
        assert span.content_text == ""
        assert span.postfix_text == ""
        assert len(span) == 0

    def test_span_with_content(self):
        span = Span(content=[Chunk("Hello")])
        assert span.is_empty is False
        assert span.content_text == "Hello"
        assert span.text == "Hello"

    def test_span_with_all_parts(self):
        span = Span(
            prefix=[Chunk(">>")],
            content=[Chunk("Hello")],
            postfix=[Chunk("\n")],
        )
        assert span.prefix_text == ">>"
        assert span.content_text == "Hello"
        assert span.postfix_text == "\n"
        assert span.text == ">>Hello\n"

    def test_span_has_end_of_line_postfix(self):
        span = Span(
            content=[Chunk("Hello")],
            postfix=[Chunk("\n")],
        )
        assert span.has_end_of_line() is True

    def test_span_has_end_of_line_content(self):
        span = Span(content=[Chunk("Hello\n")])
        assert span.has_end_of_line() is True

    def test_span_no_end_of_line(self):
        span = Span(content=[Chunk("Hello")])
        assert span.has_end_of_line() is False

    def test_span_chunks_iteration(self):
        span = Span(
            prefix=[Chunk("A")],
            content=[Chunk("B"), Chunk("C")],
            postfix=[Chunk("D")],
        )
        chunks = list(span.chunks())
        assert len(chunks) == 4
        assert [c.content for c in chunks] == ["A", "B", "C", "D"]

    def test_span_append_content(self):
        span = Span()
        span.append_content([Chunk("Hello")])
        span.append_content([Chunk(" World")])
        assert span.content_text == "Hello World"

    def test_span_prepend_content(self):
        span = Span(content=[Chunk("World")])
        span.prepend_content([Chunk("Hello ")])
        assert span.content_text == "Hello World"

    def test_span_append_prefix(self):
        span = Span(content=[Chunk("text")])
        span.append_prefix([Chunk("  ")])
        span.append_prefix([Chunk("- ")])
        assert span.prefix_text == "  - "

    def test_span_prepend_prefix(self):
        span = Span(prefix=[Chunk("- ")])
        span.prepend_prefix([Chunk("  ")])
        assert span.prefix_text == "  - "

    def test_span_append_postfix(self):
        span = Span(content=[Chunk("text")])
        span.append_postfix([Chunk("\n")])
        assert span.postfix_text == "\n"
        assert span.has_end_of_line() is True

    def test_span_prepend_postfix(self):
        span = Span(postfix=[Chunk("\n")])
        span.prepend_postfix([Chunk(";")])
        assert span.postfix_text == ";\n"

    def test_span_copy(self):
        original = Span(
            prefix=[Chunk(">>")],
            content=[Chunk("Hello")],
            postfix=[Chunk("\n")],
        )
        copy = original.copy()

        assert copy.text == original.text
        assert copy is not original
        assert copy.content[0] is not original.content[0]
        assert copy.owner is None
        assert copy.prev is None
        assert copy.next is None

    def test_span_copy_isolation(self):
        original = Span(content=[Chunk("Hello")])
        copy = original.copy()

        copy.append_content([Chunk(" World")])

        assert original.content_text == "Hello"
        assert copy.content_text == "Hello World"

    def test_span_method_chaining(self):
        span = (
            Span()
            .append_prefix([Chunk("  ")])
            .append_content([Chunk("item")])
            .append_postfix([Chunk("\n")])
        )
        assert span.text == "  item\n"
