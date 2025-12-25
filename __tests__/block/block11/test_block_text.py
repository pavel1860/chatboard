"""Tests for BlockText class."""
import pytest
from promptview.block.block11 import Chunk, Span, BlockText


class TestBlockTextCreation:
    """Tests for BlockText creation and basic operations."""

    def test_empty_block_text(self):
        bt = BlockText()
        assert len(bt) == 0
        assert bt.first is None
        assert bt.last is None
        assert bt.text() == ""
        assert list(bt) == []

    def test_bool_empty(self):
        bt = BlockText()
        assert bool(bt) is False

    def test_bool_with_spans(self):
        bt = BlockText()
        bt.create_span("Hello")
        assert bool(bt) is True


class TestBlockTextCreateSpan:
    """Tests for BlockText.create_span()."""

    def test_create_span_empty(self):
        bt = BlockText()
        span = bt.create_span()

        assert len(bt) == 1
        assert span.owner is bt
        assert span.is_empty is True

    def test_create_span_with_string(self):
        bt = BlockText()
        span = bt.create_span("Hello")

        assert len(bt) == 1
        assert span.content_text == "Hello"
        assert span.owner is bt

    def test_create_span_with_chunks(self):
        bt = BlockText()
        chunks = [Chunk("Hello"), Chunk(" World")]
        span = bt.create_span(chunks)

        assert span.content_text == "Hello World"

    def test_create_multiple_spans(self):
        bt = BlockText()
        span1 = bt.create_span("First")
        span2 = bt.create_span("Second")

        assert len(bt) == 2
        assert bt.first is span1
        assert bt.last is span2
        assert bt.text() == "FirstSecond"


class TestBlockTextInsertion:
    """Tests for BlockText insertion operations."""

    def test_append(self):
        bt = BlockText()
        span1 = Span(content=[Chunk("First")])
        span2 = Span(content=[Chunk("Second")])

        bt.append(span1)
        bt.append(span2)

        assert len(bt) == 2
        assert list(bt) == [span1, span2]
        assert span1.owner is bt
        assert span2.owner is bt

    def test_prepend(self):
        bt = BlockText()
        span1 = Span(content=[Chunk("First")])
        span2 = Span(content=[Chunk("Second")])

        bt.append(span1)
        bt.prepend(span2)

        assert list(bt) == [span2, span1]
        assert bt.text() == "SecondFirst"

    def test_insert_after(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span3 = bt.create_span("C")

        span2 = Span(content=[Chunk("B")])
        bt.insert_after(span1, span2)

        assert list(bt) == [span1, span2, span3]
        assert bt.text() == "ABC"

    def test_insert_before(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span3 = bt.create_span("C")

        span2 = Span(content=[Chunk("B")])
        bt.insert_before(span3, span2)

        assert list(bt) == [span1, span2, span3]
        assert bt.text() == "ABC"


class TestBlockTextRemoval:
    """Tests for BlockText removal operations."""

    def test_remove(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span2 = bt.create_span("B")
        span3 = bt.create_span("C")

        removed = bt.remove(span2)

        assert removed is span2
        assert span2.owner is None
        assert span2.prev is None
        assert span2.next is None
        assert len(bt) == 2
        assert bt.text() == "AC"

    def test_remove_not_owned_raises(self):
        bt1 = BlockText()
        bt2 = BlockText()
        span = bt2.create_span("Test")

        with pytest.raises(ValueError, match="not owned"):
            bt1.remove(span)

    def test_clear(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span2 = bt.create_span("B")

        bt.clear()

        assert len(bt) == 0
        assert bt.first is None
        assert bt.last is None
        assert span1.owner is None
        assert span2.owner is None


class TestBlockTextQuery:
    """Tests for BlockText query operations."""

    def test_first_last(self):
        bt = BlockText()
        span1 = bt.create_span("First")
        span2 = bt.create_span("Middle")
        span3 = bt.create_span("Last")

        assert bt.first is span1
        assert bt.last is span3

    def test_contains(self):
        bt = BlockText()
        span1 = bt.create_span("Test")
        span2 = Span(content=[Chunk("Other")])

        assert bt.contains(span1) is True
        assert bt.contains(span2) is False

    def test_spans_between(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span2 = bt.create_span("B")
        span3 = bt.create_span("C")
        span4 = bt.create_span("D")

        spans = bt.spans_between(span2, span3)

        assert spans == [span2, span3]


class TestBlockTextCopyFork:
    """Tests for BlockText copy/fork operations."""

    def test_fork_all(self):
        bt = BlockText()
        bt.create_span("Hello ")
        bt.create_span("World")

        forked = bt.fork()

        assert len(forked) == 2
        assert forked.text() == "Hello World"
        assert forked is not bt

        # Spans are copies
        original_spans = list(bt)
        forked_spans = list(forked)
        assert forked_spans[0] is not original_spans[0]
        assert forked_spans[0].owner is forked

    def test_fork_partial(self):
        bt = BlockText()
        span1 = bt.create_span("A")
        span2 = bt.create_span("B")
        span3 = bt.create_span("C")

        forked = bt.fork([span1, span3])

        assert len(forked) == 2
        assert forked.text() == "AC"

    def test_fork_isolation(self):
        bt = BlockText()
        bt.create_span("Original")

        forked = bt.fork()
        list(forked)[0].content = [Chunk("Modified")]

        assert bt.text() == "Original"
        assert forked.text() == "Modified"

    def test_copy_alias(self):
        bt = BlockText()
        bt.create_span("Test")

        copied = bt.copy()

        assert copied.text() == "Test"
        assert copied is not bt


class TestBlockTextExtend:
    """Tests for BlockText extend operations."""

    def test_extend(self):
        bt = BlockText()
        bt.create_span("First")

        spans = [
            Span(content=[Chunk("Second")]),
            Span(content=[Chunk("Third")]),
        ]
        bt.extend(spans)

        assert len(bt) == 3
        assert bt.text() == "FirstSecondThird"

    def test_extend_from_copy(self):
        bt1 = BlockText()
        bt1.create_span("A")

        bt2 = BlockText()
        bt2.create_span("B")
        bt2.create_span("C")

        bt1.extend_from(bt2, copy=True)

        assert bt1.text() == "ABC"
        assert bt2.text() == "BC"  # Original unchanged

    def test_extend_from_move(self):
        bt1 = BlockText()
        bt1.create_span("A")

        bt2 = BlockText()
        bt2.create_span("B")
        bt2.create_span("C")

        bt1.extend_from(bt2, copy=False)

        assert bt1.text() == "ABC"
        assert len(bt2) == 0  # Original cleared
