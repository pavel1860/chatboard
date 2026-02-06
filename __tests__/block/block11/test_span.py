"""Tests for Chunk and Span classes."""
import pytest
from chatboard.block.block11 import BlockChunk, Span


class TestChunk:
    """Tests for Chunk class."""

    def test_create_chunk(self):
        chunk = BlockChunk("Hello")
        assert chunk.content == "Hello"
        assert chunk.logprob is None
        assert len(chunk) == 5

    def test_chunk_with_logprob(self):
        chunk = BlockChunk("token", logprob=-0.5)
        assert chunk.content == "token"
        assert chunk.logprob == -0.5

    def test_chunk_is_line_end(self):
        chunk1 = BlockChunk("Hello")
        chunk2 = BlockChunk("World\n")
        assert chunk1.is_line_end is False
        assert chunk2.is_line_end is True

    def test_chunk_copy(self):
        original = BlockChunk("test", logprob=-0.1)
        copy = original.copy()

        assert copy.content == original.content
        assert copy.logprob == original.logprob
        assert copy.id != original.id  # New ID
        assert copy is not original

    def test_chunk_repr(self):
        chunk = BlockChunk("Hello")
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
        span = Span(chunks=[BlockChunk("Hello")])
        assert span.is_empty is False
        assert span.content_text == "Hello"
        assert span.text == "Hello"

    def test_span_with_all_parts(self):
        span = Span(
            prefix=[BlockChunk(">>")],
            chunks=[BlockChunk("Hello")],
            postfix=[BlockChunk("\n")],
        )
        assert span.prefix_text == ">>"
        assert span.content_text == "Hello"
        assert span.postfix_text == "\n"
        assert span.text == ">>Hello\n"

    def test_span_has_end_of_line_postfix(self):
        span = Span(
            chunks=[BlockChunk("Hello")],
            postfix=[BlockChunk("\n")],
        )
        assert span.has_newline() is True

    def test_span_has_end_of_line_content(self):
        span = Span(chunks=[BlockChunk("Hello\n")])
        assert span.has_newline() is True

    def test_span_no_end_of_line(self):
        span = Span(chunks=[BlockChunk("Hello")])
        assert span.has_newline() is False

    def test_span_chunks_iteration(self):
        span = Span(
            prefix=[BlockChunk("A")],
            chunks=[BlockChunk("B"), BlockChunk("C")],
            postfix=[BlockChunk("D")],
        )
        chunks = list(span.chunks())
        assert len(chunks) == 4
        assert [c.content for c in chunks] == ["A", "B", "C", "D"]

    def test_span_append_content(self):
        span = Span()
        span.append([BlockChunk("Hello")])
        span.append([BlockChunk(" World")])
        assert span.content_text == "Hello World"

    def test_span_prepend_content(self):
        span = Span(chunks=[BlockChunk("World")])
        span.prepend([BlockChunk("Hello ")])
        assert span.content_text == "Hello World"

    def test_span_append_prefix(self):
        span = Span(chunks=[BlockChunk("text")])
        span.append_prefix([BlockChunk("  ")])
        span.append_prefix([BlockChunk("- ")])
        assert span.prefix_text == "  - "

    def test_span_prepend_prefix(self):
        span = Span(prefix=[BlockChunk("- ")])
        span.prepend_prefix([BlockChunk("  ")])
        assert span.prefix_text == "  - "

    def test_span_append_postfix(self):
        span = Span(chunks=[BlockChunk("text")])
        span.append_postfix([BlockChunk("\n")])
        assert span.postfix_text == "\n"
        assert span.has_newline() is True

    def test_span_prepend_postfix(self):
        span = Span(postfix=[BlockChunk("\n")])
        span.prepend_postfix([BlockChunk(";")])
        assert span.postfix_text == ";\n"

    def test_span_copy(self):
        original = Span(
            prefix=[BlockChunk(">>")],
            chunks=[BlockChunk("Hello")],
            postfix=[BlockChunk("\n")],
        )
        copy = original.copy()

        assert copy.text == original.text
        assert copy is not original
        assert copy.chunks[0] is not original.chunks[0]
        assert copy.owner is None
        assert copy.prev is None
        assert copy.next is None

    def test_span_copy_isolation(self):
        original = Span(chunks=[BlockChunk("Hello")])
        copy = original.copy()

        copy.append([BlockChunk(" World")])

        assert original.content_text == "Hello"
        assert copy.content_text == "Hello World"

    def test_span_method_chaining(self):
        span = (
            Span()
            .append_prefix([BlockChunk("  ")])
            .append_content([BlockChunk("item")])
            .append_postfix([BlockChunk("\n")])
        )
        assert span.text == "  item\n"
