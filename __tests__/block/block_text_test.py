"""Tests for BlockText (block10) - linked list chunk storage."""

import pytest
from promptview.block.block10.chunk import Chunk, BlockText


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        chunk = Chunk(content="hello")
        assert chunk.content == "hello"
        assert chunk.logprob is None
        assert chunk.prev is None
        assert chunk.next is None
        assert len(chunk.id) == 8

    def test_chunk_with_logprob(self):
        chunk = Chunk(content="test", logprob=-0.5)
        assert chunk.logprob == -0.5

    def test_chunk_len(self):
        chunk = Chunk(content="hello")
        assert len(chunk) == 5

    def test_chunk_equality_by_id(self):
        c1 = Chunk(content="hello", id="abc123")
        c2 = Chunk(content="world", id="abc123")
        c3 = Chunk(content="hello", id="xyz789")
        assert c1 == c2  # Same ID
        assert c1 != c3  # Different ID

    def test_chunk_hash(self):
        c1 = Chunk(content="hello", id="abc123")
        c2 = Chunk(content="world", id="abc123")
        assert hash(c1) == hash(c2)

    def test_chunk_copy(self):
        original = Chunk(content="hello", logprob=-0.5)
        copied = original.copy()
        assert copied.content == "hello"
        assert copied.logprob == -0.5
        assert copied.id != original.id  # New ID
        assert copied.prev is None
        assert copied.next is None

    def test_chunk_split(self):
        chunk = Chunk(content="hello world")
        left, right = chunk.split(5)
        assert left.content == "hello"
        assert right.content == " world"
        assert left.id != chunk.id
        assert right.id != chunk.id

    def test_chunk_split_at_start(self):
        chunk = Chunk(content="hello")
        left, right = chunk.split(0)
        assert left.content == ""
        assert right.content == "hello"

    def test_chunk_split_at_end(self):
        chunk = Chunk(content="hello")
        left, right = chunk.split(5)
        assert left.content == "hello"
        assert right.content == ""

    def test_chunk_split_out_of_bounds(self):
        chunk = Chunk(content="hello")
        with pytest.raises(ValueError):
            chunk.split(-1)
        with pytest.raises(ValueError):
            chunk.split(10)

    def test_chunk_is_line_end(self):
        assert Chunk(content="hello\n").is_line_end is True
        assert Chunk(content="hello").is_line_end is False
        assert Chunk(content="\n").is_line_end is True

    def test_chunk_model_dump_and_validate(self):
        chunk = Chunk(content="hello", logprob=-0.5, id="test123")
        data = chunk.model_dump()
        restored = Chunk.model_validate(data)
        assert restored.content == "hello"
        assert restored.logprob == -0.5
        assert restored.id == "test123"


class TestBlockText:
    """Tests for BlockText linked list."""

    def test_empty_block_text(self):
        bt = BlockText()
        assert len(bt) == 0
        assert bt.is_empty is True
        assert bt.head is None
        assert bt.tail is None
        assert bool(bt) is False

    def test_append_single_chunk(self):
        bt = BlockText()
        chunk = Chunk(content="hello")
        bt.append(chunk)
        assert len(bt) == 1
        assert bt.head is chunk
        assert bt.tail is chunk
        assert chunk._owner is bt
        assert bool(bt) is True

    def test_append_multiple_chunks(self):
        bt = BlockText()
        c1 = Chunk(content="hello")
        c2 = Chunk(content=" ")
        c3 = Chunk(content="world")
        bt.append(c1)
        bt.append(c2)
        bt.append(c3)

        assert len(bt) == 3
        assert bt.head is c1
        assert bt.tail is c3
        assert c1.next is c2
        assert c2.prev is c1
        assert c2.next is c3
        assert c3.prev is c2

    def test_init_with_chunks(self):
        chunks = [Chunk(content="a"), Chunk(content="b"), Chunk(content="c")]
        bt = BlockText(chunks)
        assert len(bt) == 3
        assert bt.text() == "abc"

    def test_prepend(self):
        bt = BlockText()
        c1 = Chunk(content="world")
        c2 = Chunk(content="hello ")
        bt.append(c1)
        bt.prepend(c2)

        assert len(bt) == 2
        assert bt.head is c2
        assert bt.tail is c1
        assert bt.text() == "hello world"

    def test_insert_after(self):
        bt = BlockText()
        c1 = Chunk(content="hello")
        c2 = Chunk(content="world")
        c3 = Chunk(content=" ")
        bt.append(c1)
        bt.append(c2)
        bt.insert_after(c1, c3)

        assert len(bt) == 3
        assert bt.text() == "hello world"
        assert c1.next is c3
        assert c3.prev is c1
        assert c3.next is c2

    def test_insert_before(self):
        bt = BlockText()
        c1 = Chunk(content="hello")
        c2 = Chunk(content="world")
        c3 = Chunk(content=" ")
        bt.append(c1)
        bt.append(c2)
        bt.insert_before(c2, c3)

        assert len(bt) == 3
        assert bt.text() == "hello world"

    def test_remove(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        middle = bt.head.next
        bt.remove(middle)

        assert len(bt) == 2
        assert bt.text() == "ac"
        assert middle._owner is None

    def test_remove_head(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b")])
        head = bt.head
        bt.remove(head)
        assert len(bt) == 1
        assert bt.head.content == "b"

    def test_remove_tail(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b")])
        tail = bt.tail
        bt.remove(tail)
        assert len(bt) == 1
        assert bt.tail.content == "a"

    def test_iteration(self):
        chunks = [Chunk(content="a"), Chunk(content="b"), Chunk(content="c")]
        bt = BlockText(chunks)
        result = [c.content for c in bt]
        assert result == ["a", "b", "c"]

    def test_reversed_iteration(self):
        chunks = [Chunk(content="a"), Chunk(content="b"), Chunk(content="c")]
        bt = BlockText(chunks)
        result = [c.content for c in reversed(bt)]
        assert result == ["c", "b", "a"]

    def test_contains(self):
        bt = BlockText()
        c1 = Chunk(content="hello")
        c2 = Chunk(content="world")
        bt.append(c1)

        assert c1 in bt
        assert c2 not in bt

    def test_get_by_id(self):
        bt = BlockText()
        chunk = Chunk(content="hello", id="test123")
        bt.append(chunk)

        assert bt.get_by_id("test123") is chunk
        assert bt.get_by_id("nonexistent") is None

    def test_text(self):
        bt = BlockText([Chunk(content="hello"), Chunk(content=" "), Chunk(content="world")])
        assert bt.text() == "hello world"

    def test_chunks_list(self):
        chunks = [Chunk(content="a"), Chunk(content="b")]
        bt = BlockText(chunks)
        result = bt.chunks_list()
        assert len(result) == 2
        assert result[0].content == "a"

    def test_append_already_owned_chunk_raises(self):
        bt1 = BlockText()
        bt2 = BlockText()
        chunk = Chunk(content="hello")
        bt1.append(chunk)

        with pytest.raises(ValueError, match="already belongs"):
            bt2.append(chunk)

    def test_insert_after_wrong_blocktext_raises(self):
        bt1 = BlockText([Chunk(content="a")])
        bt2 = BlockText([Chunk(content="b")])

        with pytest.raises(ValueError, match="not in this BlockText"):
            bt1.insert_after(bt2.head, Chunk(content="c"))


class TestBlockTextExtend:
    """Tests for extend methods."""

    def test_extend_appends_chunks(self):
        bt = BlockText([Chunk(content="a")])
        chunks = [Chunk(content="b"), Chunk(content="c")]
        bt.extend(chunks)

        assert len(bt) == 3
        assert bt.text() == "abc"

    def test_extend_after_specific_chunk(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="c")])
        first = bt.head
        bt.extend([Chunk(content="b")], after=first)

        assert bt.text() == "abc"

    def test_left_extend_prepends_chunks(self):
        bt = BlockText([Chunk(content="c")])
        chunks = [Chunk(content="a"), Chunk(content="b")]
        bt.left_extend(chunks)

        assert bt.text() == "abc"

    def test_insert_chunks_after(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="d")])
        first = bt.head
        bt.insert_chunks_after(first, [Chunk(content="b"), Chunk(content="c")])

        assert bt.text() == "abcd"

    def test_insert_chunks_before(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="d")])
        last = bt.tail
        bt.insert_chunks_before(last, [Chunk(content="b"), Chunk(content="c")])

        assert bt.text() == "abcd"


class TestBlockTextExtendBlockText:
    """Tests for extend_block_text and left_extend_block_text methods."""

    def test_extend_block_text_copy_mode(self):
        bt1 = BlockText([Chunk(content="hello ")])
        bt2 = BlockText([Chunk(content="world")])

        result = bt1.extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello world"
        assert bt2.text() == "world"  # Source unchanged
        assert len(result) == 1
        assert result[0].content == "world"
        assert result[0] is not bt2.head  # Different chunk object

    def test_extend_block_text_move_mode(self):
        bt1 = BlockText([Chunk(content="hello ")])
        bt2 = BlockText([Chunk(content="world")])
        original_chunk = bt2.head

        result = bt1.extend_block_text(bt2, copy=False)

        assert bt1.text() == "hello world"
        assert bt2.is_empty  # Source emptied
        assert len(bt2) == 0
        assert result[0] is original_chunk  # Same chunk object moved
        assert original_chunk._owner is bt1

    def test_extend_block_text_after_specific_chunk(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="c")])
        bt2 = BlockText([Chunk(content="b")])
        first = bt1.head

        bt1.extend_block_text(bt2, after=first, copy=True)

        assert bt1.text() == "abc"

    def test_extend_block_text_empty_source(self):
        bt1 = BlockText([Chunk(content="hello")])
        bt2 = BlockText()

        result = bt1.extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello"
        assert result == []

    def test_extend_block_text_to_empty_target(self):
        bt1 = BlockText()
        bt2 = BlockText([Chunk(content="hello")])

        bt1.extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello"
        assert bt1.head is not None
        assert bt1.tail is not None

    def test_extend_block_text_move_clears_by_id(self):
        bt1 = BlockText()
        bt2 = BlockText([Chunk(content="hello", id="test123")])

        bt1.extend_block_text(bt2, copy=False)

        assert bt1.get_by_id("test123") is not None
        assert bt2.get_by_id("test123") is None

    def test_extend_block_text_multiple_chunks_move(self):
        bt1 = BlockText([Chunk(content="start ")])
        bt2 = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])

        bt1.extend_block_text(bt2, copy=False)

        assert bt1.text() == "start abc"
        assert len(bt1) == 4
        assert bt2.is_empty

    def test_left_extend_block_text_copy_mode(self):
        bt1 = BlockText([Chunk(content="world")])
        bt2 = BlockText([Chunk(content="hello ")])

        result = bt1.left_extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello world"
        assert bt2.text() == "hello "  # Source unchanged
        assert len(result) == 1

    def test_left_extend_block_text_move_mode(self):
        bt1 = BlockText([Chunk(content="world")])
        bt2 = BlockText([Chunk(content="hello ")])
        original_chunk = bt2.head

        result = bt1.left_extend_block_text(bt2, copy=False)

        assert bt1.text() == "hello world"
        assert bt2.is_empty
        assert result[0] is original_chunk
        assert bt1.head is original_chunk

    def test_left_extend_block_text_before_specific_chunk(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="c")])
        bt2 = BlockText([Chunk(content="b")])
        last = bt1.tail

        bt1.left_extend_block_text(bt2, before=last, copy=True)

        assert bt1.text() == "abc"

    def test_left_extend_block_text_empty_source(self):
        bt1 = BlockText([Chunk(content="hello")])
        bt2 = BlockText()

        result = bt1.left_extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello"
        assert result == []

    def test_left_extend_block_text_to_empty_target(self):
        bt1 = BlockText()
        bt2 = BlockText([Chunk(content="hello")])

        bt1.left_extend_block_text(bt2, copy=True)

        assert bt1.text() == "hello"
        assert bt1.head is not None
        assert bt1.tail is not None

    def test_linked_list_integrity_after_move(self):
        """Verify linked list pointers are correct after move operation."""
        bt1 = BlockText([Chunk(content="a"), Chunk(content="d")])
        bt2 = BlockText([Chunk(content="b"), Chunk(content="c")])
        first = bt1.head

        bt1.extend_block_text(bt2, after=first, copy=False)

        # Verify forward links
        chunks_forward = [c.content for c in bt1]
        assert chunks_forward == ["a", "b", "c", "d"]

        # Verify backward links
        chunks_backward = [c.content for c in reversed(bt1)]
        assert chunks_backward == ["d", "c", "b", "a"]

        # Verify head/tail
        assert bt1.head.content == "a"
        assert bt1.tail.content == "d"


class TestBlockTextReplace:
    """Tests for replace and replace_block_text methods."""

    def test_replace_single_chunk(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        middle = bt.head.next

        removed = bt.replace(middle, middle, [Chunk(content="X")])

        assert bt.text() == "aXc"
        assert len(removed) == 1
        assert removed[0].content == "b"
        assert removed[0]._owner is None

    def test_replace_range(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c"), Chunk(content="d")])
        start = bt.head.next      # "b"
        end = bt.tail.prev        # "c"

        removed = bt.replace(start, end, [Chunk(content="X")])

        assert bt.text() == "aXd"
        assert len(removed) == 2
        assert [c.content for c in removed] == ["b", "c"]

    def test_replace_with_multiple_chunks(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="old"), Chunk(content="d")])
        middle = bt.head.next

        bt.replace(middle, middle, [Chunk(content="b"), Chunk(content="c")])

        assert bt.text() == "abcd"
        assert len(bt) == 4

    def test_replace_delete_only(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        middle = bt.head.next

        removed = bt.replace(middle, middle, None)

        assert bt.text() == "ac"
        assert len(bt) == 2
        assert removed[0].content == "b"

    def test_replace_delete_with_empty_list(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        middle = bt.head.next

        removed = bt.replace(middle, middle, [])

        assert bt.text() == "ac"
        assert len(bt) == 2

    def test_replace_at_head(self):
        bt = BlockText([Chunk(content="old"), Chunk(content="b"), Chunk(content="c")])
        head = bt.head

        bt.replace(head, head, [Chunk(content="new")])

        assert bt.text() == "newbc"
        assert bt.head.content == "new"

    def test_replace_at_tail(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="old")])
        tail = bt.tail

        bt.replace(tail, tail, [Chunk(content="new")])

        assert bt.text() == "abnew"
        assert bt.tail.content == "new"

    def test_replace_entire_blocktext(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])

        removed = bt.replace(bt.head, bt.tail, [Chunk(content="X")])

        assert bt.text() == "X"
        assert len(bt) == 1
        assert len(removed) == 3

    def test_replace_entire_blocktext_with_empty(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b")])

        bt.replace(bt.head, bt.tail, [])

        assert bt.is_empty
        assert bt.head is None
        assert bt.tail is None

    def test_replace_wrong_order_raises(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        first = bt.head
        last = bt.tail

        with pytest.raises(ValueError, match="does not come after"):
            bt.replace(last, first, [Chunk(content="X")])

    def test_replace_chunk_not_in_blocktext_raises(self):
        bt = BlockText([Chunk(content="a")])
        other_chunk = Chunk(content="other")

        with pytest.raises(ValueError, match="not in this BlockText"):
            bt.replace(other_chunk, bt.head, [])

    def test_replace_updates_by_id(self):
        bt = BlockText([Chunk(content="a", id="id_a"), Chunk(content="b", id="id_b")])

        bt.replace(bt.tail, bt.tail, [Chunk(content="c", id="id_c")])

        assert bt.get_by_id("id_a") is not None
        assert bt.get_by_id("id_b") is None  # Removed
        assert bt.get_by_id("id_c") is not None  # Added

    def test_replace_linked_list_integrity(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c"), Chunk(content="d")])
        start = bt.head.next
        end = bt.tail.prev

        bt.replace(start, end, [Chunk(content="X"), Chunk(content="Y")])

        # Forward iteration
        assert [c.content for c in bt] == ["a", "X", "Y", "d"]
        # Backward iteration
        assert [c.content for c in reversed(bt)] == ["d", "Y", "X", "a"]

    def test_replace_block_text_copy_mode(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="old"), Chunk(content="d")])
        bt2 = BlockText([Chunk(content="b"), Chunk(content="c")])
        middle = bt1.head.next

        removed = bt1.replace_block_text(middle, middle, bt2, copy=True)

        assert bt1.text() == "abcd"
        assert bt2.text() == "bc"  # Source unchanged
        assert len(removed) == 1
        assert removed[0].content == "old"

    def test_replace_block_text_move_mode(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="old"), Chunk(content="d")])
        bt2 = BlockText([Chunk(content="b"), Chunk(content="c")])
        middle = bt1.head.next

        bt1.replace_block_text(middle, middle, bt2, copy=False)

        assert bt1.text() == "abcd"
        assert bt2.is_empty  # Source emptied

    def test_replace_block_text_empty_source(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        bt2 = BlockText()
        middle = bt1.head.next

        removed = bt1.replace_block_text(middle, middle, bt2, copy=True)

        assert bt1.text() == "ac"  # Just deletes
        assert len(removed) == 1

    def test_replace_block_text_at_head(self):
        bt1 = BlockText([Chunk(content="old"), Chunk(content="c")])
        bt2 = BlockText([Chunk(content="a"), Chunk(content="b")])

        bt1.replace_block_text(bt1.head, bt1.head, bt2, copy=True)

        assert bt1.text() == "abc"

    def test_replace_block_text_at_tail(self):
        bt1 = BlockText([Chunk(content="a"), Chunk(content="old")])
        bt2 = BlockText([Chunk(content="b"), Chunk(content="c")])

        bt1.replace_block_text(bt1.tail, bt1.tail, bt2, copy=True)

        assert bt1.text() == "abc"

    def test_replace_block_text_entire(self):
        bt1 = BlockText([Chunk(content="old")])
        bt2 = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])

        bt1.replace_block_text(bt1.head, bt1.tail, bt2, copy=True)

        assert bt1.text() == "abc"


class TestBlockTextSplitChunk:
    """Tests for split_chunk method."""

    def test_split_chunk_middle(self):
        bt = BlockText([Chunk(content="hello world")])
        original = bt.head

        left, right = bt.split_chunk(original, 5)

        assert len(bt) == 2
        assert left.content == "hello"
        assert right.content == " world"
        assert bt.head is left
        assert bt.tail is right
        assert left.next is right
        assert right.prev is left

    def test_split_chunk_preserves_neighbors(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="bc"), Chunk(content="d")])
        middle = bt.head.next

        left, right = bt.split_chunk(middle, 1)

        assert bt.text() == "abcd"
        assert len(bt) == 4


class TestBlockTextFork:
    """Tests for fork method."""

    def test_fork_entire_blocktext(self):
        bt = BlockText([Chunk(content="hello"), Chunk(content=" world")])

        forked = bt.fork()

        assert forked.text() == "hello world"
        assert len(forked) == 2
        assert forked.head is not bt.head  # Different objects
        assert forked.head.id != bt.head.id  # Different IDs

    def test_fork_range(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c"), Chunk(content="d")])
        start = bt.head.next  # "b"
        end = bt.tail.prev    # "c"

        forked = bt.fork(start=start, end=end)

        assert forked.text() == "bc"
        assert len(forked) == 2

    def test_fork_from_start(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        end = bt.head.next  # "b"

        forked = bt.fork(end=end)

        assert forked.text() == "ab"

    def test_fork_to_end(self):
        bt = BlockText([Chunk(content="a"), Chunk(content="b"), Chunk(content="c")])
        start = bt.head.next  # "b"

        forked = bt.fork(start=start)

        assert forked.text() == "bc"

    def test_fork_empty_blocktext(self):
        bt = BlockText()
        forked = bt.fork()
        assert forked.is_empty


class TestBlockTextSerialization:
    """Tests for model_dump and model_validate."""

    def test_model_dump(self):
        bt = BlockText([Chunk(content="hello", id="c1"), Chunk(content=" world", id="c2")])
        data = bt.model_dump()

        assert "chunks" in data
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["content"] == "hello"
        assert data["chunks"][0]["id"] == "c1"

    def test_model_validate(self):
        data = {
            "chunks": [
                {"id": "c1", "content": "hello", "logprob": -0.1},
                {"id": "c2", "content": " world", "logprob": None},
            ]
        }

        bt = BlockText.model_validate(data)

        assert len(bt) == 2
        assert bt.text() == "hello world"
        assert bt.head.id == "c1"
        assert bt.head.logprob == -0.1

    def test_roundtrip_serialization(self):
        original = BlockText([
            Chunk(content="hello", logprob=-0.5),
            Chunk(content=" world", logprob=-0.3),
        ])

        data = original.model_dump()
        restored = BlockText.model_validate(data)

        assert restored.text() == original.text()
        assert len(restored) == len(original)
