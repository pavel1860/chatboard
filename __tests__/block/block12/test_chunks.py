"""Tests for Block chunks and metadata."""
import pytest
from promptview.block.block12 import Block, BlockChunk


class TestBlockChunkMetadata:
    """Tests for chunk metadata."""

    def test_raw_append_with_style(self):
        block = Block()
        block._raw_append("<tag>", style="xml-open")
        block._raw_append("content", style="content")
        block._raw_append("</tag>", style="xml-close")

        assert block.text == "<tag>content</tag>"
        assert block.get_region_text('xml-open') == "<tag>"
        assert block.get_region_text('content') == "content"
        assert block.get_region_text('xml-close') == "</tag>"


class TestBlockFromChunks:
    """Tests for creating blocks from chunks."""

    def test_create_from_chunks(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        assert block.text == "<item>\n"

    def test_split_prefix(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        prefix, suffix = block.split_prefix("<")

        assert "<" == prefix
        assert "item>\n" == suffix

    def test_split_postfix(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        prefix, suffix = block.split_prefix("<")
        content, postfix = suffix.split_postfix(">")

        assert "item" == content
        assert ">\n" == postfix

    def test_split_prefix_not_found(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        prefix, suffix = block.split_prefix("#")
        assert not prefix

    def test_split_prefix_create_on_empty_false(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        prefix, suffix = block.split_prefix("#", create_on_empty=False)
        assert not prefix

    def test_reconstruct_from_chunks(self):
        chunks = [
            BlockChunk("<", logprob=0.5),
            BlockChunk("item", logprob=0.6),
            BlockChunk(">", logprob=0.7),
            BlockChunk("\n", logprob=0.8),
        ]

        block = Block(chunks)
        prefix, suffix = block.split_prefix("<")
        content, postfix = suffix.split_postfix(">")

        with Block(content) as item:
            item.prepend(prefix or "<", style="xml")
            item.append(postfix or ">", style="xml")

        assert "<item>\n" == item

        item_chunks = item.get_chunks()
        assert len(item_chunks) == 3
        assert item_chunks[0] == "<"
        assert item_chunks[1] == "item"
        assert item_chunks[2] == ">\n"


class TestBlockChunkObject:
    """Tests for BlockChunk object."""

    def test_chunk_creation(self):
        chunk = BlockChunk("content", logprob=0.9)
        assert chunk.content == "content"
        assert chunk.logprob == 0.9

    def test_chunk_equality(self):
        chunk1 = BlockChunk("test")
        assert chunk1 == "test"
