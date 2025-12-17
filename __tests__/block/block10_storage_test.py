"""
Unit tests for Block10 storage layer.

Tests cover:
1. dump_block - serializing block trees
2. load_block_dump - deserializing block trees
3. Round-trip (dump -> load -> verify)
4. Shared BlockText on load
5. Content hashing and deduplication
"""

import pytest
from promptview.block.block10 import Block, BlockChunk, BlockText, Span
from promptview.block.block10.block import BlockSchema
from promptview.model.block_models.block_log import (
    dump_block,
    load_block_dump,
    dump_block_content,
    dump_span_chunks,
    block_content_hash,
    signature_hash,
)


# =============================================================================
# dump_block Tests
# =============================================================================

class TestDumpBlock:
    """Tests for dump_block function."""

    def test_dump_simple_block(self):
        """dump_block serializes a simple block."""
        block = Block("hello world")
        dumps = dump_block(block)

        assert len(dumps) == 1
        assert dumps[0]["content"] == "hello world"
        assert dumps[0]["path"] == "0"

    def test_dump_block_with_children(self):
        """dump_block serializes block tree."""
        root = Block("root")
        root /= "child1"
        root /= "child2"

        dumps = dump_block(root)

        assert len(dumps) == 3
        paths = [d["path"] for d in dumps]
        assert "0" in paths
        assert "0.0" in paths
        assert "0.1" in paths

    def test_dump_nested_tree(self):
        """dump_block handles deeply nested trees."""
        root = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child /= grandchild
        root /= child

        dumps = dump_block(root)

        assert len(dumps) == 3
        paths = [d["path"] for d in dumps]
        assert "0" in paths
        assert "0.0" in paths
        assert "0.0.0" in paths

    def test_dump_preserves_tags(self):
        """dump_block preserves tags."""
        block = Block("content", tags=["tag1", "tag2"])
        dumps = dump_block(block)

        assert dumps[0]["tags"] == ["tag1", "tag2"]

    def test_dump_preserves_role(self):
        """dump_block preserves role."""
        block = Block("content", role="system")
        dumps = dump_block(block)

        assert dumps[0]["role"] == "system"

    def test_dump_preserves_styles(self):
        """dump_block preserves styles."""
        block = Block("content", style="xml markdown")
        dumps = dump_block(block)

        assert "xml" in dumps[0]["styles"]
        assert "markdown" in dumps[0]["styles"]

    def test_dump_wrapper_block(self):
        """dump_block handles wrapper blocks."""
        wrapper = Block(tags=["wrapper"])
        wrapper /= "child"

        dumps = dump_block(wrapper)

        assert len(dumps) == 2
        # Wrapper has empty content
        assert dumps[0]["content"] == ""
        assert dumps[0]["tags"] == ["wrapper"]

    def test_dump_json_content_has_chunks(self):
        """dump_block includes chunk data in json_content."""
        block = Block("hello world")
        dumps = dump_block(block)

        json_content = dumps[0]["json_content"]
        assert "chunks" in json_content
        assert len(json_content["chunks"]) > 0
        assert json_content["chunks"][0]["content"] == "hello world"


# =============================================================================
# load_block_dump Tests
# =============================================================================

class TestLoadBlockDump:
    """Tests for load_block_dump function."""

    def test_load_simple_block(self):
        """load_block_dump reconstructs simple block."""
        dumps = [{
            "path": "0",
            "content": "hello world",
            "json_content": {
                "chunks": [{"content": "hello world", "logprob": None}],
                "prefix_chunks": [],
                "postfix_chunks": [],
            },
            "styles": [],
            "tags": [],
            "role": None,
        }]

        block = load_block_dump(dumps)

        assert block.span.text() == "hello world"

    def test_load_block_with_children(self):
        """load_block_dump reconstructs tree structure."""
        dumps = [
            {
                "path": "0",
                "content": "root",
                "json_content": {"chunks": [{"content": "root"}], "prefix_chunks": [], "postfix_chunks": []},
                "styles": [], "tags": [], "role": None,
            },
            {
                "path": "0.0",
                "content": "child1",
                "json_content": {"chunks": [{"content": "child1"}], "prefix_chunks": [], "postfix_chunks": []},
                "styles": [], "tags": [], "role": None,
            },
            {
                "path": "0.1",
                "content": "child2",
                "json_content": {"chunks": [{"content": "child2"}], "prefix_chunks": [], "postfix_chunks": []},
                "styles": [], "tags": [], "role": None,
            },
        ]

        block = load_block_dump(dumps)

        assert len(block.children) == 2
        assert block.children[0].span.text() == "child1"
        assert block.children[1].span.text() == "child2"

    def test_load_preserves_tags(self):
        """load_block_dump preserves tags."""
        dumps = [{
            "path": "0",
            "content": "content",
            "json_content": {"chunks": [{"content": "content"}], "prefix_chunks": [], "postfix_chunks": []},
            "styles": [],
            "tags": ["tag1", "tag2"],
            "role": None,
        }]

        block = load_block_dump(dumps)

        assert block.tags == ["tag1", "tag2"]

    def test_load_preserves_role(self):
        """load_block_dump preserves role."""
        dumps = [{
            "path": "0",
            "content": "content",
            "json_content": {"chunks": [{"content": "content"}], "prefix_chunks": [], "postfix_chunks": []},
            "styles": [],
            "tags": [],
            "role": "system",
        }]

        block = load_block_dump(dumps)

        assert block.role == "system"

    def test_load_preserves_styles(self):
        """load_block_dump preserves styles."""
        dumps = [{
            "path": "0",
            "content": "content",
            "json_content": {"chunks": [{"content": "content"}], "prefix_chunks": [], "postfix_chunks": []},
            "styles": ["xml", "markdown"],
            "tags": [],
            "role": None,
        }]

        block = load_block_dump(dumps)

        assert "xml" in block.styles
        assert "markdown" in block.styles

    def test_load_wrapper_block(self):
        """load_block_dump handles wrapper blocks (empty chunks)."""
        dumps = [
            {
                "path": "0",
                "content": "",
                "json_content": {"chunks": [], "prefix_chunks": [], "postfix_chunks": []},
                "styles": [], "tags": ["wrapper"], "role": None,
            },
            {
                "path": "0.0",
                "content": "child",
                "json_content": {"chunks": [{"content": "child"}], "prefix_chunks": [], "postfix_chunks": []},
                "styles": [], "tags": [], "role": None,
            },
        ]

        block = load_block_dump(dumps)

        assert block.is_wrapper
        assert len(block.children) == 1


# =============================================================================
# Round-trip Tests
# =============================================================================

class TestRoundTrip:
    """Tests for dump -> load round-trip."""

    def test_roundtrip_simple_block(self):
        """Simple block survives round-trip."""
        original = Block("hello world")

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        assert loaded.span.text() == original.span.text()

    def test_roundtrip_block_with_children(self):
        """Block with children survives round-trip."""
        original = Block("root")
        original /= "child1"
        original /= "child2"

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        assert len(loaded.children) == len(original.children)
        assert loaded.render() == original.render()

    def test_roundtrip_nested_tree(self):
        """Deeply nested tree survives round-trip."""
        original = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")

        child /= grandchild
        original /= child

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        assert len(list(loaded.traverse())) == 3
        assert loaded.render() == original.render()

    def test_roundtrip_preserves_metadata(self):
        """Round-trip preserves tags, role, styles."""
        original = Block("content", tags=["tag1"], role="system", style="xml")

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        assert loaded.tags == original.tags
        assert loaded.role == original.role
        assert loaded.styles == original.styles

    def test_roundtrip_wrapper_block(self):
        """Wrapper block survives round-trip."""
        original = Block(tags=["wrapper"])
        original /= "child1"
        original /= "child2"

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        assert loaded.is_wrapper
        assert len(loaded.children) == 2

    def test_roundtrip_complex_tree(self):
        """Complex tree structure survives round-trip."""
        root = Block("root", role="system")
        root /= Block("section1", tags=["section"])
        root /= Block("section2", tags=["section"])
        root.children[0] /= Block("item1")
        root.children[0] /= Block("item2")
        root.children[1] /= Block("item3")

        dumps = dump_block(root)
        loaded = load_block_dump(dumps)

        original_blocks = list(root.traverse())
        loaded_blocks = list(loaded.traverse())

        assert len(loaded_blocks) == len(original_blocks)
        assert loaded.render() == root.render()


# =============================================================================
# Shared BlockText Tests
# =============================================================================

class TestSharedBlockTextOnLoad:
    """Tests for shared BlockText after loading."""

    def test_loaded_blocks_share_block_text(self):
        """All loaded blocks share single BlockText."""
        original = Block("root")
        original /= "child1"
        original /= "child2"

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        # Collect all BlockText instances
        block_texts = set()
        for block in loaded.traverse():
            block_texts.add(id(block.block_text))

        assert len(block_texts) == 1

    def test_loaded_nested_tree_shares_block_text(self):
        """Deeply nested loaded tree shares BlockText."""
        original = Block("root")
        child = Block("child")
        grandchild = Block("grandchild")
        child /= grandchild
        original /= child

        dumps = dump_block(original)
        loaded = load_block_dump(dumps)

        root_bt = loaded.block_text
        for block in loaded.traverse():
            assert block.block_text is root_bt


# =============================================================================
# Hash Functions Tests
# =============================================================================

class TestHashFunctions:
    """Tests for content and signature hashing."""

    def test_block_content_hash_same_content(self):
        """Same content produces same hash."""
        chunks1 = [{"content": "hello", "logprob": None}]
        chunks2 = [{"content": "hello", "logprob": None}]

        assert block_content_hash(chunks1) == block_content_hash(chunks2)

    def test_block_content_hash_different_content(self):
        """Different content produces different hash."""
        chunks1 = [{"content": "hello"}]
        chunks2 = [{"content": "world"}]

        assert block_content_hash(chunks1) != block_content_hash(chunks2)

    def test_block_content_hash_empty(self):
        """Empty chunks list produces consistent hash."""
        hash1 = block_content_hash([])
        hash2 = block_content_hash([])

        assert hash1 == hash2

    def test_signature_hash_same_signature(self):
        """Same block_id + styling produces same hash."""
        sig1 = signature_hash("block1", ["xml"], "system", ["tag1"])
        sig2 = signature_hash("block1", ["xml"], "system", ["tag1"])

        assert sig1 == sig2

    def test_signature_hash_different_block_id(self):
        """Different block_id produces different hash."""
        sig1 = signature_hash("block1", ["xml"], "system", ["tag1"])
        sig2 = signature_hash("block2", ["xml"], "system", ["tag1"])

        assert sig1 != sig2

    def test_signature_hash_different_styles(self):
        """Different styles produces different hash."""
        sig1 = signature_hash("block1", ["xml"], "system", ["tag1"])
        sig2 = signature_hash("block1", ["markdown"], "system", ["tag1"])

        assert sig1 != sig2

    def test_signature_hash_different_role(self):
        """Different role produces different hash."""
        sig1 = signature_hash("block1", ["xml"], "system", ["tag1"])
        sig2 = signature_hash("block1", ["xml"], "user", ["tag1"])

        assert sig1 != sig2

    def test_signature_hash_different_tags(self):
        """Different tags produces different hash."""
        sig1 = signature_hash("block1", ["xml"], "system", ["tag1"])
        sig2 = signature_hash("block1", ["xml"], "system", ["tag2"])

        assert sig1 != sig2


# =============================================================================
# dump_span_chunks Tests
# =============================================================================

class TestDumpSpanChunks:
    """Tests for dump_span_chunks function."""

    def test_dump_span_chunks_simple(self):
        """dump_span_chunks extracts chunk content."""
        block = Block("hello world")
        chunks = dump_span_chunks(block, block.span)

        assert len(chunks) == 1
        assert chunks[0]["content"] == "hello world"

    def test_dump_span_chunks_none_span(self):
        """dump_span_chunks returns empty for None span."""
        block = Block("content")
        chunks = dump_span_chunks(block, None)

        assert chunks == []

    def test_dump_span_chunks_preserves_logprob(self):
        """dump_span_chunks preserves logprob metadata."""
        # Create block with chunk that has logprob
        bt = BlockText()
        chunk = BlockChunk(content="hello", logprob=-0.5)
        bt.append(chunk)

        block = Block(block_text=bt, _skip_content=True)
        block.span = Span.from_chunks([chunk])

        chunks = dump_span_chunks(block, block.span)

        assert chunks[0]["logprob"] == -0.5


# =============================================================================
# dump_block_content Tests
# =============================================================================

class TestDumpBlockContent:
    """Tests for dump_block_content function."""

    def test_dump_block_content_simple(self):
        """dump_block_content returns content and json_content."""
        block = Block("hello world")
        result = dump_block_content(block)

        assert result["content"] == "hello world"
        assert "chunks" in result["json_content"]
        assert "prefix_chunks" in result["json_content"]
        assert "postfix_chunks" in result["json_content"]

    def test_dump_block_content_wrapper(self):
        """dump_block_content handles wrapper blocks."""
        wrapper = Block()
        result = dump_block_content(wrapper)

        assert result["content"] == ""
        assert result["json_content"]["chunks"] == []
