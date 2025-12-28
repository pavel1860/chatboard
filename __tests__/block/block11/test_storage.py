"""Tests for Block11 Merkle tree storage."""

import pytest
from promptview.block.block11 import Block, BlockSchema, Chunk, Span
from promptview.model.block_models.block11_storage import (
    compute_span_hash,
    compute_block_hash,
    dump_span,
    dump_block,
    dump_chunks,
    load_block,
)


class TestHashComputation:
    """Test hash computation functions."""

    def test_span_hash_deterministic(self):
        """Same content produces same hash."""
        hash1 = compute_span_hash("", "hello", "", [], [{"content": "hello", "logprob": None}], [])
        hash2 = compute_span_hash("", "hello", "", [], [{"content": "hello", "logprob": None}], [])
        assert hash1 == hash2

    def test_span_hash_different_content(self):
        """Different content produces different hash."""
        hash1 = compute_span_hash("", "hello", "", [], [{"content": "hello", "logprob": None}], [])
        hash2 = compute_span_hash("", "world", "", [], [{"content": "world", "logprob": None}], [])
        assert hash1 != hash2

    def test_span_hash_includes_logprob(self):
        """Logprob affects hash."""
        hash1 = compute_span_hash("", "hello", "", [], [{"content": "hello", "logprob": None}], [])
        hash2 = compute_span_hash("", "hello", "", [], [{"content": "hello", "logprob": -0.5}], [])
        assert hash1 != hash2

    def test_span_hash_includes_prefix_postfix(self):
        """Prefix and postfix affect hash."""
        hash1 = compute_span_hash("pre", "content", "post", [], [], [])
        hash2 = compute_span_hash("", "content", "", [], [], [])
        assert hash1 != hash2

    def test_block_hash_deterministic(self):
        """Same block data produces same hash."""
        hash1 = compute_block_hash("span1", "user", ["tag1"], ["xml"], None, None, {}, ["child1", "child2"])
        hash2 = compute_block_hash("span1", "user", ["tag1"], ["xml"], None, None, {}, ["child1", "child2"])
        assert hash1 == hash2

    def test_block_hash_different_children(self):
        """Different children produce different hash."""
        hash1 = compute_block_hash("span1", "user", [], [], None, None, {}, ["child1", "child2"])
        hash2 = compute_block_hash("span1", "user", [], [], None, None, {}, ["child1", "child3"])
        assert hash1 != hash2

    def test_block_hash_children_order_matters(self):
        """Children order affects hash."""
        hash1 = compute_block_hash("span1", None, [], [], None, None, {}, ["child1", "child2"])
        hash2 = compute_block_hash("span1", None, [], [], None, None, {}, ["child2", "child1"])
        assert hash1 != hash2

    def test_block_hash_tags_sorted(self):
        """Tags are sorted before hashing (order doesn't matter)."""
        hash1 = compute_block_hash(None, None, ["a", "b"], [], None, None, {}, [])
        hash2 = compute_block_hash(None, None, ["b", "a"], [], None, None, {}, [])
        assert hash1 == hash2

    def test_block_hash_includes_schema_fields(self):
        """Schema-specific fields affect hash."""
        hash1 = compute_block_hash(None, None, [], [], "thought", "str", {"key": "value"}, [])
        hash2 = compute_block_hash(None, None, [], [], "answer", "str", {"key": "value"}, [])
        assert hash1 != hash2


class TestDumpChunks:
    """Test chunk serialization."""

    def test_dump_empty_chunks(self):
        """Empty list returns empty list."""
        result = dump_chunks([])
        assert result == []

    def test_dump_single_chunk(self):
        """Single chunk is serialized correctly."""
        chunk = Chunk(content="hello", logprob=-0.5)
        result = dump_chunks([chunk])
        assert result == [{"content": "hello", "logprob": -0.5}]

    def test_dump_multiple_chunks(self):
        """Multiple chunks are serialized in order."""
        chunks = [
            Chunk(content="hello", logprob=-0.1),
            Chunk(content=" world", logprob=-0.2),
        ]
        result = dump_chunks(chunks)
        assert result == [
            {"content": "hello", "logprob": -0.1},
            {"content": " world", "logprob": -0.2},
        ]

    def test_dump_chunk_none_logprob(self):
        """Chunk with None logprob is handled."""
        chunk = Chunk(content="test")
        result = dump_chunks([chunk])
        assert result == [{"content": "test", "logprob": None}]


class TestDumpSpan:
    """Test span serialization."""

    def test_dump_empty_span(self):
        """Empty span produces valid output."""
        span = Span()
        result = dump_span(span)

        assert "id" in result
        assert result["prefix_text"] == ""
        assert result["content_text"] == ""
        assert result["postfix_text"] == ""

    def test_dump_span_with_content(self):
        """Span with content is serialized correctly."""
        span = Span(
            content=[Chunk(content="hello"), Chunk(content=" world")]
        )
        result = dump_span(span)

        assert result["content_text"] == "hello world"
        assert len(result["content_chunks"]) == 2

    def test_dump_span_with_prefix_postfix(self):
        """Span with prefix/postfix is serialized correctly."""
        span = Span(
            prefix=[Chunk(content="<tag>")],
            content=[Chunk(content="content")],
            postfix=[Chunk(content="</tag>")],
        )
        result = dump_span(span)

        assert result["prefix_text"] == "<tag>"
        assert result["content_text"] == "content"
        assert result["postfix_text"] == "</tag>"


class TestDumpBlock:
    """Test block tree serialization."""

    def test_dump_simple_block(self):
        """Simple block without children."""
        block = Block("hello world", role="user")
        root_data, all_blocks, all_spans = dump_block(block)

        assert len(all_blocks) == 1
        assert len(all_spans) == 1
        assert root_data["role"] == "user"

    def test_dump_block_with_children(self):
        """Block with children is serialized correctly."""
        with Block() as root:
            root /= "child 1"
            root /= "child 2"

        root_data, all_blocks, all_spans = dump_block(root)

        assert len(all_blocks) == 3  # root + 2 children
        assert len(root_data["children"]) == 2

    def test_dump_block_schema(self):
        """BlockSchema fields are included."""
        schema = BlockSchema("thought", tags=["thinking"])
        root_data, all_blocks, all_spans = dump_block(schema)

        assert root_data["name"] == "thought"
        assert "thinking" in root_data["tags"]

    def test_dump_nested_tree(self):
        """Deeply nested tree is serialized correctly."""
        with Block() as root:
            with root("level1") as level1:
                with level1("level2") as level2:
                    level2 /= "level3"

        root_data, all_blocks, all_spans = dump_block(root)

        assert len(all_blocks) == 4  # root + 3 levels

    def test_dump_identical_subtrees_same_hash(self):
        """Identical subtrees produce same hash (deduplication)."""
        subtree1 = Block("same content", role="user", tags=["tag1"])
        subtree2 = Block("same content", role="user", tags=["tag1"])

        _, blocks1, spans1 = dump_block(subtree1)
        _, blocks2, spans2 = dump_block(subtree2)

        assert list(blocks1.keys())[0] == list(blocks2.keys())[0]


class TestLoadBlock:
    """Test block tree deserialization."""

    def test_load_simple_block(self):
        """Load simple block from storage format."""
        spans = {
            "span1": {
                "id": "span1",
                "prefix_text": "",
                "content_text": "hello",
                "postfix_text": "",
                "prefix_chunks": [],
                "content_chunks": [{"content": "hello", "logprob": None}],
                "postfix_chunks": [],
            }
        }
        blocks = {
            "block1": {
                "id": "block1",
                "span_id": "span1",
                "role": "user",
                "tags": ["greeting"],
                "styles": [],
                "name": None,
                "type_name": None,
                "attrs": {},
                "children": [],
            }
        }

        block = load_block("block1", blocks, spans)

        assert block.role == "user"
        assert "greeting" in block.tags
        assert block.content == "hello"

    def test_load_block_with_children(self):
        """Load block with children."""
        spans = {
            "span1": {"id": "span1", "prefix_text": "", "content_text": "parent", "postfix_text": "",
                      "prefix_chunks": [], "content_chunks": [{"content": "parent", "logprob": None}], "postfix_chunks": []},
            "span2": {"id": "span2", "prefix_text": "", "content_text": "child", "postfix_text": "",
                      "prefix_chunks": [], "content_chunks": [{"content": "child", "logprob": None}], "postfix_chunks": []},
        }
        blocks = {
            "root": {"id": "root", "span_id": "span1", "role": None, "tags": [], "styles": [],
                     "name": None, "type_name": None, "attrs": {}, "children": ["child1"]},
            "child1": {"id": "child1", "span_id": "span2", "role": None, "tags": [], "styles": [],
                       "name": None, "type_name": None, "attrs": {}, "children": []},
        }

        block = load_block("root", blocks, spans)

        assert len(block.children) == 1
        assert block.children[0].content == "child"
        assert block.children[0].parent is block

    def test_load_block_schema(self):
        """Load BlockSchema with name."""
        spans = {}
        blocks = {
            "schema1": {
                "id": "schema1",
                "span_id": None,
                "role": None,
                "tags": ["thinking"],
                "styles": ["xml"],
                "name": "thought",
                "type_name": "str",
                "attrs": {"required": True},
                "children": [],
            }
        }

        block = load_block("schema1", blocks, spans)

        assert isinstance(block, BlockSchema)
        assert block.name == "thought"

    def test_load_preserves_logprobs(self):
        """Logprobs are preserved after load."""
        spans = {
            "span1": {
                "id": "span1",
                "prefix_text": "",
                "content_text": "hello world",
                "postfix_text": "",
                "prefix_chunks": [],
                "content_chunks": [
                    {"content": "hello", "logprob": -0.1},
                    {"content": " world", "logprob": -0.2},
                ],
                "postfix_chunks": [],
            }
        }
        blocks = {
            "block1": {"id": "block1", "span_id": "span1", "role": None, "tags": [], "styles": [],
                       "name": None, "type_name": None, "attrs": {}, "children": []},
        }

        block = load_block("block1", blocks, spans)

        assert block.span.content[0].logprob == -0.1
        assert block.span.content[1].logprob == -0.2


class TestRoundTrip:
    """Test dump → load round trip preserves data."""

    def test_round_trip_simple_block(self):
        """Simple block survives round trip."""
        original = Block("hello world", role="user", tags=["greeting"])

        root_data, all_blocks, all_spans = dump_block(original)
        restored = load_block(root_data["id"], all_blocks, all_spans)

        assert restored.content == original.content
        assert restored.role == original.role
        assert restored.tags == original.tags

    def test_round_trip_nested_tree(self):
        """Nested tree survives round trip."""
        with Block("root", role="system") as original:
            with original("child1", role="user") as child1:
                child1 /= "grandchild"
            with original("child2", role="assistant") as child2:
                child2 /= "response"

        root_data, all_blocks, all_spans = dump_block(original)
        restored = load_block(root_data["id"], all_blocks, all_spans)

        assert restored.content == "root"
        assert len(restored.children) == 2
        assert restored.children[0].content == "child1"
        assert len(restored.children[0].children) == 1
        assert restored.children[0].children[0].content == "grandchild"

    def test_round_trip_block_schema(self):
        """BlockSchema survives round trip."""
        original = BlockSchema("thought", tags=["thinking"], style="xml")

        root_data, all_blocks, all_spans = dump_block(original)
        restored = load_block(root_data["id"], all_blocks, all_spans)

        assert isinstance(restored, BlockSchema)
        assert restored.name == "thought"
        assert "thinking" in restored.tags

    def test_round_trip_shared_block_text(self):
        """All blocks share same BlockText after load."""
        with Block("root") as original:
            original /= "child1"
            original /= "child2"

        root_data, all_blocks, all_spans = dump_block(original)
        restored = load_block(root_data["id"], all_blocks, all_spans)

        assert restored.block_text is restored.children[0].block_text
        assert restored.block_text is restored.children[1].block_text


class TestMerkleDeduplication:
    """Test that Merkle hashing enables deduplication."""

    def test_identical_blocks_same_hash(self):
        """Identical blocks produce identical hashes."""
        block1 = Block("same content", role="user")
        block2 = Block("same content", role="user")

        _, blocks1, _ = dump_block(block1)
        _, blocks2, _ = dump_block(block2)

        assert list(blocks1.keys()) == list(blocks2.keys())

    def test_identical_subtrees_reused(self):
        """When dumping multiple trees with same subtree, it appears once."""
        with Block() as tree1:
            tree1 /= Block("shared system prompt", role="system")
            tree1 /= Block("user message 1", role="user")

        with Block() as tree2:
            tree2 /= Block("shared system prompt", role="system")
            tree2 /= Block("user message 2", role="user")

        _, blocks1, spans1 = dump_block(tree1)
        _, blocks2, spans2 = dump_block(tree2)

        system_id_1 = [b["id"] for b in blocks1.values() if b.get("role") == "system"][0]
        system_id_2 = [b["id"] for b in blocks2.values() if b.get("role") == "system"][0]

        assert system_id_1 == system_id_2

    def test_different_children_different_hash(self):
        """Parent with different children has different hash."""
        with Block("parent") as parent1:
            parent1 /= "child A"

        with Block("parent") as parent2:
            parent2 /= "child B"

        root1, _, _ = dump_block(parent1)
        root2, _, _ = dump_block(parent2)

        assert root1["id"] != root2["id"]
